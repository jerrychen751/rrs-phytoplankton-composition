#!/usr/bin/env python3
"""
Validate the Kramer-trained SDP model on PVST BATS and NES-LTER datasets.

Compares predicted pigment concentrations to in-situ HPLC using:
  1. L3 (0.1-deg daily) Rrs via OPeNDAP — coarser matchup, no downloads
  2. L2 (swath-level, ~1 km) Rrs via CMR search + curl download — finer resolution

SST/SSS use climatological defaults (20 °C, 35 PSU) for the GSM bio-optical
inversion.  This introduces small errors but matches the multi_source_cv.py
fallback when ancillary files are absent.

Usage:
    python experiments/method_comparison/validate_kramer_sdp.py

Prerequisites:
    - HPLC SeaBASS files in experiments/{pvst_bats,nes_lter}_validation/inputs/
    - ~/.netrc with urs.earthdata.nasa.gov credentials (for L2 download + OPeNDAP)
"""

from __future__ import annotations

import datetime as dt
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.method_comparison.config import SOURCES, CONFIG
from experiments.method_comparison.evaluation import compute_gof
from models.sdp_pigments.core.prediction import run_sdp
from utils.cmr_download import search_cmr, get_bearer_token, download_granule
from utils.pace_l2 import (
    preprocess_rrs_spectrum,
    haversine_km,
    extract_l2_matchup,
)
from utils.seabass_loader import load_seabass_file

# ── Configuration ──────────────────────────────────────────────────────────

PIGMENTS = CONFIG["multi_source"]["pigments"]
SPECTRAL_CFG = CONFIG["multi_source"]["spectral"]

L3_TEMPORAL_WINDOW_DAYS = 3
L2_TIME_WINDOW_HOURS = 12  # search full day for L2 granules
L2_MAX_DISTANCE_KM = 5.0
L2_QC_EXCLUDE_BITS = [0, 1, 3, 4, 8, 9, 14, 16, 25, 26]
L2_MIN_FINITE_FRACTION = 0.95

CLIM_SST = 20.0   # °C — climatological default
CLIM_SSS = 35.0   # PSU — climatological default

VALIDATION_SOURCES = ["pvst_bats", "nes_lter"]

L3_OPENDAP_BASE = "https://oceandata.sci.gsfc.nasa.gov/opendap/PACE_OCI/L3SMI"

# Reverse map from run_sdp() display names → SDP internal names
DISPLAY_TO_SDP = {
    "T chla": "Tchla", "Zea": "Zea", "DV chla": "DVchla",
    "ButFuco": "ButFuco", "HexFuco": "HexFuco", "Allo": "Allo",
    "MV chlb": "MVchlb", "Neo": "Neo", "Viola": "Viola",
    "Fuco": "Fuco", "chl c1+c2": "Chlc12", "chl c3": "Chlc3",
    "Perid": "Perid",
}

# SeaBASS column name → SDP pigment name (same as multi_source_cv.py)
SEABASS_TO_SDP: dict[str, str] = {
    "tot_chl_a": "Tchla", "but-fuco": "ButFuco", "hex-fuco": "HexFuco",
    "allo": "Allo", "fuco": "Fuco", "perid": "Perid", "zea": "Zea",
    "dv_chl_a": "DVchla", "mv_chl_b": "MVchlb", "chl_c1c2": "Chlc12",
    "chl_c3": "Chlc3", "neo": "Neo", "viola": "Viola",
}


# ── Helpers ────────────────────────────────────────────────────────────────

def get_source(name: str) -> dict:
    """Look up a source config entry by name."""
    return next(s for s in SOURCES if s["name"] == name)


# ── 1. Load HPLC Ground Truth ─────────────────────────────────────────────
#
# Duplicates the minimal SeaBASS loading logic from multi_source_cv.py to
# keep this script self-contained (no matplotlib or heavy imports).


def load_ground_truth(source_name: str) -> pd.DataFrame:
    """
    Load surface-filtered HPLC pigments for a validation source.

    Reads each SeaBASS .sb file, filters to depth <= surface_depth_m,
    maps column names to SDP pigment names, and averages replicates
    per (station, date).  Returns DataFrame indexed by sample_id.
    """
    src = get_source(source_name)
    surface_depth_m = src["surface_depth_m"]
    parts = []

    for path_str in src["hplc_files"]:
        path = PROJECT_ROOT / path_str
        df, _ = load_seabass_file(path)

        if "depth" in df.columns:
            df = df[pd.to_numeric(df["depth"], errors="coerce") <= surface_depth_m].copy()
        if df.empty:
            continue

        df.columns = [c.lower() for c in df.columns]
        rename = {k: v for k, v in SEABASS_TO_SDP.items() if k in df.columns}
        df = df.rename(columns=rename)

        for col in ("lat", "lon"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        parts.append(df)

    if not parts:
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)
    available = [p for p in PIGMENTS if p in combined.columns]

    combined["station"] = combined["station"].astype(str)
    combined["date"] = pd.to_numeric(combined["date"], errors="coerce").astype("Int64")
    combined = combined.dropna(subset=["date", "lat", "lon"])

    grouped = (
        combined
        .groupby(["station", "date"], dropna=True)[available + ["lat", "lon"]]
        .mean()
        .reset_index()
    )
    grouped["sample_id"] = [
        f"{source_name}__{row['station']}__{int(row['date'])}"
        for _, row in grouped.iterrows()
    ]
    grouped = grouped.set_index("sample_id")

    print(f"  {source_name}: {len(combined)} surface rows -> "
          f"{len(grouped)} unique (station, date) groups")
    return grouped


# ── 2. L3 OPeNDAP Matchup ─────────────────────────────────────────────────


def _build_l3_url(date: dt.date) -> str:
    """Build OPeNDAP URL for a PACE L3 0.1-deg daily Rrs file."""
    mmdd = date.strftime("%m%d")
    yyyymmdd = date.strftime("%Y%m%d")
    return (
        f"{L3_OPENDAP_BASE}/{date.year}/{mmdd}/"
        f"PACE_OCI.{yyyymmdd}.L3m.DAY.RRS.V3_1.Rrs.0p1deg.nc"
    )


def match_l3_opendap(hplc_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Match HPLC observations to L3 Rrs via OPeNDAP.

    For each observation, tries the same-day L3 file first, then expands
    to +/-1 .. +/-N days.  Opens the remote file lazily via OPeNDAP,
    selects the nearest 0.1-deg pixel, and preprocesses the spectrum.

    Returns (rrs_df, matched_sample_ids).
    """
    n = len(hplc_df)
    matched_rrs: list[np.ndarray] = []
    matched_ids: list[str] = []

    for sample_id, row in hplc_df.iterrows():
        date_int = int(row["date"])
        obs_date = dt.datetime.strptime(str(date_int), "%Y%m%d").date()
        lat, lon = float(row["lat"]), float(row["lon"])

        # Search same-day first, then expand outward
        offsets = [0] + [
            d for abs_d in range(1, L3_TEMPORAL_WINDOW_DAYS + 1)
            for d in (-abs_d, abs_d)
        ]

        found = False
        for offset in offsets:
            try_date = obs_date + dt.timedelta(days=offset)
            url = _build_l3_url(try_date)

            try:
                with xr.open_dataset(url, engine="netcdf4") as ds:
                    pixel = ds["Rrs"].sel(lat=lat, lon=lon, method="nearest")
                    pixel.load()

                    spec = pixel.values
                    if np.all(np.isnan(spec)):
                        continue

                    native_wl = ds.wavelength.values
                    pixel_lat = float(pixel.lat.values)
                    pixel_lon = float(pixel.lon.values)
                    dist_km = haversine_km(lat, lon, pixel_lat, pixel_lon)

                    _, rrs_1nm = preprocess_rrs_spectrum(
                        native_wl, spec,
                        interp_nm=SPECTRAL_CFG["interp_nm"],
                        smooth_nm=SPECTRAL_CFG["smooth_nm"],
                        edge_trim_nm=SPECTRAL_CFG["edge_trim_nm"],
                        final_range_nm=SPECTRAL_CFG["final_range_nm"],
                    )

                    if np.isfinite(rrs_1nm).all():
                        matched_rrs.append(rrs_1nm)
                        matched_ids.append(str(sample_id))
                        if offset != 0:
                            print(f"    {sample_id}: same-day cloudy, "
                                  f"used {try_date} (offset {offset:+d}d)")
                        found = True
                        break

            except OSError:
                continue
            except ValueError as e:
                print(f"    {sample_id}: spectral preprocessing failed ({e})")
                continue

        if not found:
            print(f"    {sample_id}: no valid L3 match "
                  f"within +/-{L3_TEMPORAL_WINDOW_DAYS}d")

    print(f"  L3 OPeNDAP: {len(matched_rrs)}/{n} matched")

    if not matched_rrs:
        return pd.DataFrame(), []

    wavelengths = np.arange(400, 701, 1)
    rrs_df = pd.DataFrame(np.stack(matched_rrs), columns=wavelengths)
    return rrs_df, matched_ids


# ── 3. L2 CMR Search + Download Matchup ───────────────────────────────────


def match_l2(hplc_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Match HPLC observations to L2 Rrs via CMR search + authenticated download.

    For each observation:
    1. Search CMR for PACE L2 AOP granules on the observation date.
    2. Download each candidate to a temp dir (Bearer token auth).
    3. Use extract_l2_matchup() for nearest-pixel extraction + QC.
    4. Auto-delete temp files when done.

    Caches downloaded granules by URL so the same granule is only
    downloaded once even if multiple observations share it.
    """
    n = len(hplc_df)
    matched_rrs: list[np.ndarray] = []
    matched_ids: list[str] = []

    # Set up authenticated session
    try:
        token = get_bearer_token()
    except Exception as e:
        print(f"  Skipping L2: cannot obtain Earthdata token ({e})")
        return pd.DataFrame(), []

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"

    with tempfile.TemporaryDirectory(prefix="sdp_l2_") as tmpdir:
        tmp = Path(tmpdir)
        # Cache: URL -> local path (avoid re-downloading shared granules)
        cache: dict[str, Path] = {}

        for sample_id, row in hplc_df.iterrows():
            date_int = int(row["date"])
            obs_date = dt.datetime.strptime(str(date_int), "%Y%m%d").date()
            lat, lon = float(row["lat"]), float(row["lon"])
            obs_time = pd.Timestamp(obs_date, tz="UTC") + pd.Timedelta(hours=12)

            granules = search_cmr(lat, lon, obs_date)
            if not granules:
                print(f"    {sample_id}: no L2 granules found via CMR")
                continue

            found = False
            for gran in granules:
                url = gran["url"]
                fname = url.rsplit("/", 1)[-1]

                # Download if not cached
                if url not in cache:
                    local = tmp / fname
                    print(f"    Downloading {fname}...", end=" ", flush=True)
                    if download_granule(url, local, session):
                        print("OK")
                        cache[url] = local
                    else:
                        print("FAILED")
                        continue
                local_path = cache[url]

                try:
                    matchup = extract_l2_matchup(
                        local_path,
                        obs_index=0,
                        obs_time=obs_time,
                        obs_lat=lat,
                        obs_lon=lon,
                        station=str(sample_id),
                        exclude_bits=L2_QC_EXCLUDE_BITS,
                        min_finite_fraction=L2_MIN_FINITE_FRACTION,
                        interp_nm=SPECTRAL_CFG["interp_nm"],
                        smooth_nm=SPECTRAL_CFG["smooth_nm"],
                        edge_trim_nm=SPECTRAL_CFG["edge_trim_nm"],
                        final_range_nm=SPECTRAL_CFG["final_range_nm"],
                    )
                except Exception as e:
                    print(f"      {sample_id}: extraction error ({e})")
                    continue

                if matchup is None:
                    continue
                if matchup.distance_km > L2_MAX_DISTANCE_KM:
                    continue

                matched_rrs.append(matchup.rrs_400_700_1nm)
                matched_ids.append(str(sample_id))
                found = True
                break

            if not found:
                print(f"    {sample_id}: no valid L2 matchup "
                      f"({len(granules)} candidates tried)")

    print(f"  L2: {len(matched_rrs)}/{n} matched")

    if not matched_rrs:
        return pd.DataFrame(), []

    wavelengths = np.arange(400, 701, 1)
    rrs_df = pd.DataFrame(np.stack(matched_rrs), columns=wavelengths)
    return rrs_df, matched_ids


# ── 4. Predict + Evaluate ─────────────────────────────────────────────────


def predict_sdp(rrs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run SDP ensemble predictions on matched Rrs spectra.

    Uses climatological SST/SSS for the GSM bio-optical inversion that
    computes Rrs residuals internally.  Returns DataFrame with SDP
    pigment name columns (not display names).
    """
    wavelengths = np.arange(400, 701, 1)
    n = len(rrs_df)
    sst = np.full(n, CLIM_SST)
    sss = np.full(n, CLIM_SSS)

    pred_df = run_sdp(rrs_df, wavelengths, sst, sss, pigments=PIGMENTS)
    pred_df = pred_df.rename(columns=DISPLAY_TO_SDP)
    return pred_df


def evaluate(
    pred_df: pd.DataFrame,
    hplc_df: pd.DataFrame,
    matched_ids: list[str],
    dataset_name: str,
    level: str,
) -> pd.DataFrame:
    """
    Compute per-pigment goodness-of-fit metrics and print a summary table.

    Flags pigments with systematic over-prediction (>20 % bias) or
    under-prediction (<-20 % bias).
    """
    print(f"\n{'=' * 72}")
    print(f"  {dataset_name} -- {level} Validation  (n={len(matched_ids)})")
    print(f"{'=' * 72}")

    if not matched_ids:
        print("  No matchups -- skipping evaluation")
        return pd.DataFrame()

    obs = hplc_df.loc[matched_ids]

    rows = []
    for pig in PIGMENTS:
        if pig not in obs.columns or pig not in pred_df.columns:
            continue

        observed = obs[pig].values.astype(float)
        predicted = pred_df[pig].values.astype(float)
        mask = np.isfinite(observed) & np.isfinite(predicted)

        if mask.sum() < 3:
            rows.append({"pigment": pig, "n": int(mask.sum()),
                         "R2": np.nan, "RMSE": np.nan, "MAE": np.nan,
                         "pct_bias": np.nan, "median_pct_error": np.nan,
                         "mean_pct_error": np.nan})
            continue

        gof = compute_gof(predicted[mask], observed[mask])
        gof["pigment"] = pig
        gof["n"] = int(mask.sum())
        rows.append(gof)

    df = pd.DataFrame(rows)

    # ---- Pretty-print ----
    hdr = (f"  {'Pigment':<10} {'n':>3}  {'R2':>7}  {'Bias%':>8}  "
           f"{'MdPE%':>8}  {'RMSE':>8}  {'MAE':>8}")
    sep = (f"  {'---':<10} {'---':>3}  {'---':>7}  {'---':>8}  "
           f"{'---':>8}  {'---':>8}  {'---':>8}")
    print(hdr)
    print(sep)

    for _, r in df.iterrows():
        flag = ""
        b = r["pct_bias"]
        if np.isfinite(b):
            if b > 50:   flag = " ^^"
            elif b > 20: flag = " ^"
            elif b < -50: flag = " vv"
            elif b < -20: flag = " v"

        print(
            f"  {r['pigment']:<10} {int(r['n']):>3}  {r['R2']:>7.3f}  "
            f"{r['pct_bias']:>7.1f}%  {r['median_pct_error']:>7.1f}%  "
            f"{r['RMSE']:>8.4f}  {r['MAE']:>8.4f}{flag}"
        )

    valid = df.dropna(subset=["R2"])
    if not valid.empty:
        mean_r2 = valid["R2"].mean()
        print(f"\n  Mean R2: {mean_r2:.3f}")

        over = valid[valid["pct_bias"] > 20]
        under = valid[valid["pct_bias"] < -20]
        if not over.empty:
            print(f"  Over-predicted (>20% bias):  "
                  f"{', '.join(over['pigment'].tolist())}")
        if not under.empty:
            print(f"  Under-predicted (<-20% bias): "
                  f"{', '.join(under['pigment'].tolist())}")

    return df


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    all_results: dict[str, pd.DataFrame] = {}

    for source_name in VALIDATION_SOURCES:
        print(f"\n{'#' * 72}")
        print(f"  Dataset: {source_name}")
        print(f"{'#' * 72}")

        # 1. Ground truth
        hplc_df = load_ground_truth(source_name)
        if hplc_df.empty:
            print(f"  No HPLC data for {source_name}")
            continue
        print(f"  Ground truth: {len(hplc_df)} surface samples, "
              f"{sum(1 for p in PIGMENTS if p in hplc_df.columns)}/{len(PIGMENTS)} pigments")

        # 2. L3 OPeNDAP matchup + evaluation
        print(f"\n--- L3 OPeNDAP matchup (0.1 deg, +/-{L3_TEMPORAL_WINDOW_DAYS}d) ---")
        l3_rrs, l3_ids = match_l3_opendap(hplc_df)
        if not l3_rrs.empty:
            l3_pred = predict_sdp(l3_rrs)
            all_results[f"{source_name}_L3"] = evaluate(
                l3_pred, hplc_df, l3_ids, source_name, "L3",
            )

        # 3. L2 CMR + download matchup + evaluation
        print(f"\n--- L2 CMR+download matchup (~1 km, same-day) ---")
        l2_rrs, l2_ids = match_l2(hplc_df)
        if not l2_rrs.empty:
            l2_pred = predict_sdp(l2_rrs)
            all_results[f"{source_name}_L2"] = evaluate(
                l2_pred, hplc_df, l2_ids, source_name, "L2",
            )

    # ---- Final comparison ----
    if all_results:
        print(f"\n{'#' * 72}")
        print("  SUMMARY: L3 vs L2")
        print(f"{'#' * 72}")
        for key, df in all_results.items():
            valid = df.dropna(subset=["R2"])
            if valid.empty:
                continue
            mr2 = valid["R2"].mean()
            mb = valid["pct_bias"].mean()
            print(f"  {key:<25} n_pigments={len(valid):>2}  "
                  f"mean_R2={mr2:.3f}  mean_bias={mb:+.1f}%")


if __name__ == "__main__":
    main()
