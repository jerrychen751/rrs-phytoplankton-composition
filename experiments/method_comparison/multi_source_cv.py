#!/usr/bin/env python3
"""
Multi-Source 5-Fold Cross-Validation.

Combines in-situ HPLC pigment measurements from six geographic datasets
into one training corpus, then runs 5-fold CV to compare 7 regression methods (4 linear + 3 nonlinear)
for predicting phytoplankton pigment concentrations from satellite Rrs spectra.

Tchla conditioning (chl_conditioning.enabled=True):
  For non-Tchla pigments, log10(HPLC Tchla) is appended as feature 300.
  This tests whether knowing total chlorophyll (a proxy for trophic state and
  community structure) improves accessory pigment predictions. Tchla itself is
  predicted from spectral features alone (no circular dependency). At deployment,
  satellite chlor_a replaces HPLC Tchla.

Data sources (total ~260 surface HPLC samples before cloud filtering):
  - pax_shearwater     : coastal California, Sep 2024  (~30 stations)
  - pvst_sopace_km2419 : Pacific transoceanic, Nov–Dec 2024  (49 samples)
  - pvst_sopace_tn444  : Bay of Bengal, May–Jun 2025  (68 samples)
  - pvst_sopace_tn440  : W Pacific / Micronesia, Dec 2024–Jan 2025  (13 samples)
  - pvst_bats          : Bermuda BATS station, Jan–Aug 2025  (~17 surface)
  - nes_lter           : NE US Shelf cross-shelf transect, Feb 2024–Apr 2025  (~82 surface)

Why k-fold instead of LOOCV:
  With n≈150+ samples, LOOCV runs 150+ model fits per pigment per method —
  computationally wasteful and no longer necessary for stable estimates.
  5-fold CV gives 80/20 train/test splits that are realistic for a satellite
  operational model, and total compute is ~5x less than LOOCV at this scale.

SST/SSS for physics residuals:
  The Kramer GSM model requires SST and SSS to estimate pure-water absorption.
  If SST/SSS files are present (downloaded via download_sst_sss.py), they are
  used. If not, climatological defaults (SST=20°C, SSS=35 PSU) are used with
  a warning. The effect on 2nd-derivative features is small but non-zero.

Usage:
    python experiments/method_comparison/multi_source_cv.py

Prerequisites:
    1. Run download_l3_rrs.py (on login node or locally)
    2. Optionally run download_sst_sss.py for more accurate physics residuals

Outputs: experiments/method_comparison/outputs/multi_source_cv/
    - cv_comparison_table.csv
    - scatter_grid.png
    - R2_by_pigment.png
    - MAE_by_pigment.png
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

from sklearn.model_selection import KFold

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sdp_pigments.core.physics import get_rrs_residuals
from utils.pace_l2 import (
    preprocess_rrs_spectrum,
    haversine_km,
    extract_l2_matchup,
    extract_l2_neighborhood_matchup,
    select_candidate_granules,
    list_nc_files,
)
from utils.seabass_loader import load_seabass_file

from experiments.method_comparison.config import CONFIG, SOURCES
from experiments.method_comparison.sklearn_models import train_sklearn_model, predict_sklearn_model
from experiments.method_comparison.evaluation import compute_gof

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "multi_source_cv"

# ---------------------------------------------------------------------------
# Column name mappings
# ---------------------------------------------------------------------------

# SeaBASS files use various capitalization conventions; we lowercase all
# column names before applying this map.
SEABASS_TO_SDP: dict[str, str] = {
    "tot_chl_a": "Tchla",
    "but-fuco":  "ButFuco",
    "hex-fuco":  "HexFuco",
    "allo":      "Allo",
    "fuco":      "Fuco",
    "perid":     "Perid",
    "zea":       "Zea",
    "dv_chl_a":  "DVchla",
    "mv_chl_b":  "MVchlb",
    "chl_c1c2":  "Chlc12",
    "chl_c3":    "Chlc3",
    "neo":       "Neo",
    "viola":     "Viola",
}

# PAX Shearwater CSV uses mixed-case instrument names.
CSV_TO_SDP: dict[str, str] = {
    "Tot_Chl_a": "Tchla",
    "But-fuco":  "ButFuco",
    "Hex-fuco":  "HexFuco",
    "Allo":      "Allo",
    "Fuco":      "Fuco",
    "Perid":     "Perid",
    "Zea":       "Zea",
    "DV_Chl_a":  "DVchla",
    "MV_Chl_b":  "MVchlb",
    "Chl_c1c2":  "Chlc12",
    "Chl_c3":    "Chlc3",
    "Neo":       "Neo",
    "Viola":     "Viola",
}

# Meta columns to retain after loading (subset available in both formats).
META_COLS = ["station", "date", "lat", "lon"]


# ---------------------------------------------------------------------------
# 1. HPLC loading
# ---------------------------------------------------------------------------

def _make_sample_id(source_name: str, station: object, date: object) -> str:
    """
    Build a unique string key for one HPLC sample.

    Station numbers repeat across cruises and sources, so we prefix with the
    source name and include the date to guarantee uniqueness.
    """
    return f"{source_name}__{station}__{int(date)}"


def load_csv_hplc(source_cfg: dict, pigments: list[str]) -> pd.DataFrame:
    """
    Load PAX Shearwater HPLC CSV, average station replicates, assign sample IDs.

    The CSV has 3 technical replicates per station (3 rows → 1 averaged row).
    Returns a DataFrame indexed by unique sample_id strings, with columns for
    each SDP pigment plus lat, lon, date.
    """
    path = PROJECT_ROOT / source_cfg["hplc_files"][0]
    df = pd.read_csv(path)

    rename = {k: v for k, v in CSV_TO_SDP.items() if k in df.columns}
    df = df.rename(columns=rename)

    available_pigs = [p for p in pigments if p in df.columns]
    agg_cols = available_pigs + ["lat", "lon"]

    grouped = df.groupby("station")[agg_cols + ["date"]].agg(
        {**{c: "mean" for c in agg_cols}, "date": "first"}
    )

    grouped["sample_id"] = [
        _make_sample_id(source_cfg["name"], st, grouped.loc[st, "date"])
        for st in grouped.index
    ]
    grouped = grouped.set_index("sample_id")

    print(f"  {source_cfg['name']}: {len(df)} rows → {len(grouped)} stations after averaging")
    return grouped


def load_seabass_hplc(source_cfg: dict, pigments: list[str]) -> pd.DataFrame:
    """
    Load HPLC from one or more SeaBASS files, filter to surface, average replicates.

    Steps:
    1. Load each .sb file using load_seabass_file (handles missing=-9999 and
       below_detection_limit=-8888 → both become NaN).
    2. Filter rows to depth ≤ surface_depth_m.
    3. Lowercase all column names, map SeaBASS names → SDP names.
    4. Group by (station, date) and average pigments + (lat, lon).
    5. Assign a unique sample_id per group.

    The below-detection-limit (-8888) values are treated as NaN here.
    They're excluded from per-pigment CV folds since we can't use NaN targets.
    """
    surface_depth_m = source_cfg["surface_depth_m"]
    parts = []

    for path_str in source_cfg["hplc_files"]:
        path = PROJECT_ROOT / path_str
        df, _ = load_seabass_file(path)

        # Filter to surface
        if "depth" in df.columns:
            df = df[pd.to_numeric(df["depth"], errors="coerce") <= surface_depth_m].copy()

        if df.empty:
            print(f"    Warning: no surface samples in {path.name}")
            continue

        # Normalize to lowercase, then map to SDP names
        df.columns = [c.lower() for c in df.columns]
        rename = {k: v for k, v in SEABASS_TO_SDP.items() if k in df.columns}
        df = df.rename(columns=rename)

        # Ensure numeric lat/lon
        for col in ("lat", "lon"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        parts.append(df)

    if not parts:
        print(f"  {source_cfg['name']}: no surface samples found")
        return pd.DataFrame()

    combined = pd.concat(parts, ignore_index=True)

    # Average replicates per (station, date)
    available_pigs = [p for p in pigments if p in combined.columns]
    group_cols = ["station", "date"]
    agg_cols = available_pigs + ["lat", "lon"]

    # Coerce station and date to string/int for groupby stability
    combined["station"] = combined["station"].astype(str)
    combined["date"] = pd.to_numeric(combined["date"], errors="coerce").astype("Int64")
    combined = combined.dropna(subset=["date", "lat", "lon"])

    grouped = (
        combined
        .groupby(group_cols, dropna=True)[agg_cols]
        .mean()
        .reset_index()
    )

    grouped["sample_id"] = [
        _make_sample_id(source_cfg["name"], row["station"], row["date"])
        for _, row in grouped.iterrows()
    ]
    grouped = grouped.set_index("sample_id")

    n_raw = len(combined)
    print(
        f"  {source_cfg['name']}: {n_raw} surface rows → {len(grouped)} unique (station,date) groups"
    )
    return grouped


def load_kramer_hplc(source_cfg: dict, pigments: list[str]) -> pd.DataFrame:
    """
    Load the Kramer-et-al 2021 CSV which bundles in-situ Rrs with HPLC pigments.

    Unlike satellite sources, there is no matchup step: the Rrs spectrum (400–700 nm
    at 1 nm) is already co-located with the pigment measurements, and Sal/Temp are
    measured in-situ. This function returns a DataFrame in the same shape as
    load_csv_hplc/load_seabass_hplc, but also embeds rrs_0..rrs_300 and sst/sss
    columns so that load_all_data() can bypass load_l3_rrs_matchups().

    Row index = unique sample_id string; "station" column duplicates the index
    since that's what load_all_data()'s matched_ids lookup expects.
    """
    path = PROJECT_ROOT / source_cfg["hplc_files"][0]
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["Date/Time"]).dt.strftime("%Y%m%d").astype(int)
    df = df.rename(columns={"Latitude": "lat", "Longitude": "lon"})

    # Use row index as a station surrogate (no station numbers in this dataset)
    sample_ids = [_make_sample_id(source_cfg["name"], i, df.loc[i, "date"]) for i in df.index]
    df.index = sample_ids
    df.index.name = "sample_id"
    df["station"] = sample_ids

    # Rename Rrs400..Rrs700 → rrs_0..rrs_300 (matches the matchup_df column scheme).
    # Build all 301 columns at once via concat to avoid DataFrame fragmentation.
    rrs_renamed = pd.DataFrame(
        {f"rrs_{i}": df[f"Rrs{wl}"].values for i, wl in enumerate(range(400, 701))},
        index=df.index,
    )
    df = pd.concat([df, rrs_renamed], axis=1)

    # In-situ temperature and salinity stand in for SST/SSS
    df["sst"] = df["Temp"].values.astype(float)
    df["sss"] = df["Sal"].values.astype(float)

    available_pigs = [p for p in pigments if p in df.columns]
    keep = available_pigs + ["lat", "lon", "date", "station", "sst", "sss"] + \
           [f"rrs_{i}" for i in range(301)]
    print(f"  {source_cfg['name']}: {len(df)} samples, "
          f"{len(available_pigs)}/{len(pigments)} pigments present")
    return df[keep]


def load_all_hplc(sources: list[dict], pigments: list[str]) -> dict[str, pd.DataFrame]:
    """
    Load HPLC for every source. Returns {source_name: hplc_df}.
    """
    print("\n" + "=" * 60)
    print("Loading HPLC ground truth (all sources)")
    print("=" * 60)

    hplc_by_source: dict[str, pd.DataFrame] = {}
    for src in sources:
        if src["hplc_format"] == "csv":
            df = load_csv_hplc(src, pigments)
        elif src["hplc_format"] == "kramer":
            df = load_kramer_hplc(src, pigments)
        else:
            df = load_seabass_hplc(src, pigments)

        if not df.empty:
            df["source"] = src["name"]   # track origin for each sample
            hplc_by_source[src["name"]] = df

    total = sum(len(v) for v in hplc_by_source.values())
    print(f"\n  Total HPLC samples loaded: {total}")
    return hplc_by_source


# ---------------------------------------------------------------------------
# 2. L3 Rrs matchup
# ---------------------------------------------------------------------------

L3_DATE_RE = re.compile(r"PACE_OCI\.(\d{8})\.L3m")


def _parse_l3_date(filename: str) -> str | None:
    m = L3_DATE_RE.search(filename)
    return m.group(1) if m else None


def load_l3_rrs_matchups(
    rrs_dir: Path,
    hplc_df: pd.DataFrame,
    spectral_cfg: dict,
    temporal_window_days: int,
    compute_spatial_cv: bool = False,
    cv_window: int = 3,
    cv_wl_range: tuple[int, int] = (440, 560),
) -> pd.DataFrame:
    """
    Match HPLC stations to PACE OCI L3 0.1-deg daily Rrs pixels.

    For each HPLC station:
      1. Find the L3 file for the station's sampling date (±temporal_window_days).
      2. Select the nearest 0.1-degree pixel via xarray .sel(method='nearest').
      3. Skip if the pixel is all-NaN (cloud/land masked).
      4. Interpolate the non-uniform 172-band spectrum to 1-nm grid [400,700].

    Args:
        compute_spatial_cv: If True, extract a cv_window×cv_window box of L3
            pixels around each matchup and compute the median coefficient of
            variation (CV) of native-band Rrs across valid pixels. The result
            is stored in a ``spatial_cv`` column for downstream QC filtering.
        cv_window: Pixel box size for the spatial CV computation (default 3×3).

    xarray's .sel(method='nearest') does a binary search on the sorted
    coordinate array — O(log n) per lookup, much faster than a manual loop.

    Returns:
        DataFrame with columns: station, lat, lon, date, rrs_date, date_offset,
        pixel_lat, pixel_lon, pixel_dist_km, rrs_0 .. rrs_300.
    """
    rrs_files = sorted(rrs_dir.glob("*.nc"))
    date_to_file: dict[int, Path] = {}
    for f in rrs_files:
        d = _parse_l3_date(f.name)
        if d:
            date_to_file[int(d)] = f

    print(f"  Found {len(date_to_file)} L3 Rrs files in {rrs_dir.name}")
    if not date_to_file:
        print(f"  WARNING: no L3 files found — run download_l3_rrs.py first")
        return pd.DataFrame()

    matchups = []
    for sample_id, row in hplc_df.iterrows():
        date_int = int(row["date"])
        lat_s, lon_s = float(row["lat"]), float(row["lon"])

        offsets = [0] + [
            d for abs_d in range(1, temporal_window_days + 1)
            for d in (-abs_d, abs_d)
        ]

        spectrum = pixel_lat = pixel_lon = native_wl = used_date = None
        used_offset = 0
        spatial_cv_val = np.nan

        for offset in offsets:
            cand = date_int + offset
            if cand not in date_to_file:
                continue
            try:
                with xr.open_dataset(date_to_file[cand]) as ds:
                    pixel = ds["Rrs"].sel(lat=lat_s, lon=lon_s, method="nearest")
                    spec = pixel.values
                    if not np.all(np.isnan(spec)):
                        spectrum = spec
                        pixel_lat = float(pixel.lat.values)
                        pixel_lon = float(pixel.lon.values)
                        native_wl = ds.wavelength.values
                        used_date = cand
                        used_offset = offset

                        if compute_spatial_cv:
                            # Extract cv_window × cv_window box for spatial
                            # homogeneity check (Bailey & Werdell 2006).
                            # Restrict to cv_wl_range to avoid noisy UV/NIR bands
                            # where Rrs ≈ 0 and CV blows up.
                            lat_arr = ds.lat.values
                            lon_arr = ds.lon.values
                            li = int(np.argmin(np.abs(lat_arr - pixel_lat)))
                            lo = int(np.argmin(np.abs(lon_arr - pixel_lon)))
                            r = cv_window // 2
                            wl_mask = (native_wl >= cv_wl_range[0]) & (native_wl <= cv_wl_range[1])
                            win_da = ds["Rrs"].isel(
                                lat=slice(max(0, li - r), min(len(lat_arr), li + r + 1)),
                                lon=slice(max(0, lo - r), min(len(lon_arr), lo + r + 1)),
                            )
                            # Move wavelength to axis 0 regardless of file dim order
                            arr = win_da.values
                            wl_dim = list(win_da.dims).index("wavelength")
                            if wl_dim != 0:
                                arr = np.moveaxis(arr, wl_dim, 0)
                            arr = arr[wl_mask]  # (n_wl_subset, ≤win, ≤win)
                            flat = arr.reshape(arr.shape[0], -1)
                            n_valid = np.sum(np.isfinite(flat), axis=1)
                            min_valid = (cv_window * cv_window + 1) // 2
                            with np.errstate(divide="ignore", invalid="ignore"):
                                mu = np.nanmean(flat, axis=1)
                                sd = np.nanstd(flat, axis=1)
                                cvs = sd / np.maximum(np.abs(mu), 1e-10)
                            cvs[(n_valid < min_valid) | ~np.isfinite(cvs)] = np.nan
                            ok = cvs[np.isfinite(cvs)]
                            spatial_cv_val = float(np.median(ok)) if len(ok) > 0 else np.nan

                        break
            except OSError as e:
                # Corrupt or unreadable NetCDF/HDF5 file (e.g. from a failed
                # write on Lustre). Remove it from the date map so other
                # stations don't attempt it again, and try the next offset.
                print(f"    WARNING: corrupt L3 file {date_to_file[cand].name} — {e}")
                date_to_file.pop(cand)

        if spectrum is None:
            continue

        if used_offset != 0:
            print(f"    {sample_id}: same-day cloudy, using {used_date} (offset {used_offset:+d}d)")

        dist_km = haversine_km(lat_s, lon_s, pixel_lat, pixel_lon)

        try:
            _, rrs_1nm = preprocess_rrs_spectrum(
                native_wl, spectrum,
                interp_nm=spectral_cfg["interp_nm"],
                smooth_nm=spectral_cfg["smooth_nm"],
                edge_trim_nm=spectral_cfg["edge_trim_nm"],
                final_range_nm=spectral_cfg["final_range_nm"],
            )
        except ValueError as e:
            print(f"    {sample_id}: spectral preprocessing failed ({e}), skipping")
            continue

        if not np.isfinite(rrs_1nm).all():
            continue

        entry: dict = {
            "station": sample_id,
            "lat": lat_s, "lon": lon_s,
            "date": date_int,
            "rrs_date": used_date,
            "date_offset": used_offset,
            "pixel_lat": pixel_lat, "pixel_lon": pixel_lon,
            "pixel_dist_km": dist_km,
        }
        if compute_spatial_cv:
            entry["spatial_cv"] = spatial_cv_val
        for i, v in enumerate(rrs_1nm):
            entry[f"rrs_{i}"] = v
        matchups.append(entry)

    result = pd.DataFrame(matchups)
    n_total = len(hplc_df)
    n_matched = len(result)
    print(f"  {n_matched}/{n_total} matched to valid L3 pixels")
    return result


# ---------------------------------------------------------------------------
# 2b. L2 Rrs matchup
# ---------------------------------------------------------------------------


def load_l2_rrs_matchups(
    l2_rrs_dir: Path,
    hplc_df: pd.DataFrame,
    spectral_cfg: dict,
    l2_cfg: dict,
) -> pd.DataFrame:
    """
    Match HPLC stations to pre-downloaded PACE OCI L2 swath-level Rrs pixels.

    For each HPLC station:
      1. Construct obs_time as noon UTC on the observation date (HPLC files
         have dates but not times; noon is a reasonable default that, with a
         12-hour window, covers the full day).
      2. select_candidate_granules() filters nearby-in-time granules.
      3. extract_l2_matchup() extracts the nearest pixel + QC + spectral
         preprocessing.
      4. Accept the first match within max_distance_km.

    Optimization: observations are grouped by candidate granule so each large
    L2 file (~400 MB) is only opened for observations that still need a match,
    avoiding redundant extraction attempts on already-matched stations.

    Returns:
        DataFrame with same columns as L3 version: station, lat, lon, date,
        pixel_lat, pixel_lon, pixel_dist_km, rrs_0..rrs_300.
    """
    all_granules = list_nc_files(l2_rrs_dir)
    print(f"  Found {len(all_granules)} L2 granules in {l2_rrs_dir.name}")
    if not all_granules:
        print(f"  WARNING: no L2 files found — run download_l2_rrs.py first")
        return pd.DataFrame()

    time_window_hours = l2_cfg["time_window_hours"]
    max_distance_km = l2_cfg["max_distance_km"]
    max_granules = l2_cfg["max_granules_per_obs"]
    qc_exclude_bits = l2_cfg["qc_exclude_bits"]
    min_finite_fraction = l2_cfg["min_finite_fraction"]

    # Build per-observation candidate lists, then invert to granule -> obs list
    # so we open each large file only once.
    obs_list: list[dict] = []
    for sample_id, row in hplc_df.iterrows():
        date_int = int(row["date"])
        obs_date = pd.Timestamp(str(date_int), tz="UTC") + pd.Timedelta(hours=12)
        obs_list.append({
            "sample_id": str(sample_id),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "date": date_int,
            "obs_time": obs_date,
        })

    # Map: granule path -> list of obs indices that could use it
    granule_to_obs: dict[Path, list[int]] = {}
    for obs_idx, obs in enumerate(obs_list):
        candidates = select_candidate_granules(
            all_granules, obs["obs_time"],
            time_window_hours=time_window_hours,
            max_granules=max_granules,
        )
        for gran_path in candidates:
            granule_to_obs.setdefault(gran_path, []).append(obs_idx)

    # Track which observations already have a match (first valid wins)
    matched: dict[int, dict] = {}

    for gran_path, obs_indices in granule_to_obs.items():
        # Only process observations that still need a match
        pending = [i for i in obs_indices if i not in matched]
        if not pending:
            continue

        for obs_idx in pending:
            obs = obs_list[obs_idx]
            try:
                matchup = extract_l2_matchup(
                    gran_path,
                    obs_index=obs_idx,
                    obs_time=obs["obs_time"],
                    obs_lat=obs["lat"],
                    obs_lon=obs["lon"],
                    station=obs["sample_id"],
                    exclude_bits=qc_exclude_bits,
                    min_finite_fraction=min_finite_fraction,
                    interp_nm=spectral_cfg["interp_nm"],
                    smooth_nm=spectral_cfg["smooth_nm"],
                    edge_trim_nm=spectral_cfg["edge_trim_nm"],
                    final_range_nm=spectral_cfg["final_range_nm"],
                )
            except Exception as e:
                print(f"    {obs['sample_id']}: L2 extraction error ({e})")
                continue

            if matchup is None:
                continue
            if matchup.distance_km > max_distance_km:
                continue

            entry: dict = {
                "station": obs["sample_id"],
                "lat": obs["lat"],
                "lon": obs["lon"],
                "date": obs["date"],
                "pixel_lat": matchup.pixel_lat,
                "pixel_lon": matchup.pixel_lon,
                "pixel_dist_km": matchup.distance_km,
            }
            for i, v in enumerate(matchup.rrs_400_700_1nm):
                entry[f"rrs_{i}"] = v
            matched[obs_idx] = entry

    matchups = [matched[i] for i in sorted(matched)]
    result = pd.DataFrame(matchups)
    n_total = len(hplc_df)
    n_matched = len(result)
    print(f"  {n_matched}/{n_total} matched to valid L2 pixels")
    return result


# ---------------------------------------------------------------------------
# 2c. L2-interp (neighborhood-averaged) Rrs matchup
# ---------------------------------------------------------------------------


def load_l2_interp_rrs_matchups(
    l2_rrs_dir: Path,
    hplc_df: pd.DataFrame,
    spectral_cfg: dict,
    l2_interp_cfg: dict,
) -> pd.DataFrame:
    """
    Match HPLC stations to neighborhood-averaged L2 swath-level Rrs spectra.

    Like load_l2_rrs_matchups(), but instead of extracting a single nearest
    pixel, this computes the median Rrs spectrum from all valid pixels within
    a radius of each station. This reduces single-pixel noise while preserving
    finer spatial resolution than L3 (~3 km effective footprint vs ~11 km).

    The median is computed on native PACE bands (before interpolation to 1 nm),
    then the result is preprocessed once. This ordering is important:
      - Spatial outlier rejection works best on raw (unsmoothed) data.
      - One preprocess_rrs_spectrum() call per station instead of N per pixel.
      - Avoids smoothing-of-smoothing ambiguity.

    Uses the same granule-grouping optimization as the L2 loader: each ~400 MB
    file is opened once for all stations that could match it.

    Returns:
        DataFrame with same columns as L3/L2 loaders (station, lat, lon, date,
        pixel_dist_km, rrs_0..rrs_300) plus diagnostics: n_valid_pixels,
        n_pixels_in_radius.
    """
    all_granules = list_nc_files(l2_rrs_dir)
    print(f"  Found {len(all_granules)} L2 granules in {l2_rrs_dir.name}")
    if not all_granules:
        print(f"  WARNING: no L2 files found — run download_l2_rrs.py first")
        return pd.DataFrame()

    time_window_hours = l2_interp_cfg["time_window_hours"]
    max_distance_km = l2_interp_cfg["max_distance_km"]
    max_granules = l2_interp_cfg["max_granules_per_obs"]
    qc_exclude_bits = l2_interp_cfg["qc_exclude_bits"]
    min_finite_fraction = l2_interp_cfg["min_finite_fraction"]
    radius_km = l2_interp_cfg["radius_km"]
    min_valid_pixels = l2_interp_cfg["min_valid_pixels"]

    # Build per-observation candidate lists, then invert to granule -> obs list.
    obs_list: list[dict] = []
    for sample_id, row in hplc_df.iterrows():
        date_int = int(row["date"])
        obs_date = pd.Timestamp(str(date_int), tz="UTC") + pd.Timedelta(hours=12)
        obs_list.append({
            "sample_id": str(sample_id),
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "date": date_int,
            "obs_time": obs_date,
        })

    # Map: granule path -> list of obs indices that could use it
    granule_to_obs: dict[Path, list[int]] = {}
    for obs_idx, obs in enumerate(obs_list):
        candidates = select_candidate_granules(
            all_granules, obs["obs_time"],
            time_window_hours=time_window_hours,
            max_granules=max_granules,
        )
        for gran_path in candidates:
            granule_to_obs.setdefault(gran_path, []).append(obs_idx)

    # Track which observations already have a match (first valid wins)
    matched: dict[int, dict] = {}

    for gran_path, obs_indices in granule_to_obs.items():
        pending = [i for i in obs_indices if i not in matched]
        if not pending:
            continue

        for obs_idx in pending:
            obs = obs_list[obs_idx]
            try:
                matchup = extract_l2_neighborhood_matchup(
                    gran_path,
                    obs_index=obs_idx,
                    obs_time=obs["obs_time"],
                    obs_lat=obs["lat"],
                    obs_lon=obs["lon"],
                    station=obs["sample_id"],
                    exclude_bits=qc_exclude_bits,
                    min_finite_fraction=min_finite_fraction,
                    radius_km=radius_km,
                    min_valid_pixels=min_valid_pixels,
                    max_distance_km=max_distance_km,
                )
            except Exception as e:
                print(f"    {obs['sample_id']}: L2-interp extraction error ({e})")
                continue

            if matchup is None:
                continue

            # Preprocess the native-band median to 400-700 @ 1nm.
            try:
                _, rrs_1nm = preprocess_rrs_spectrum(
                    matchup.wavelengths_nm,
                    matchup.rrs_median_native,
                    interp_nm=spectral_cfg["interp_nm"],
                    smooth_nm=spectral_cfg["smooth_nm"],
                    edge_trim_nm=spectral_cfg["edge_trim_nm"],
                    final_range_nm=spectral_cfg["final_range_nm"],
                )
            except ValueError as e:
                print(f"    {obs['sample_id']}: spectral preprocessing failed ({e})")
                continue

            if not np.isfinite(rrs_1nm).all():
                continue

            entry: dict = {
                "station": obs["sample_id"],
                "lat": obs["lat"],
                "lon": obs["lon"],
                "date": obs["date"],
                "pixel_lat": matchup.center_lat,
                "pixel_lon": matchup.center_lon,
                "pixel_dist_km": matchup.center_distance_km,
                "n_pixels_in_radius": matchup.n_pixels_in_radius,
                "n_valid_pixels": matchup.n_valid_pixels,
            }
            for i, v in enumerate(rrs_1nm):
                entry[f"rrs_{i}"] = v
            matched[obs_idx] = entry

    matchups = [matched[i] for i in sorted(matched)]
    result = pd.DataFrame(matchups)
    n_total = len(hplc_df)
    n_matched = len(result)

    if n_matched > 0:
        med_pix = int(np.median(result["n_valid_pixels"]))
        print(f"  {n_matched}/{n_total} matched to valid L2-interp neighborhoods "
              f"(median {med_pix} valid pixels per matchup)")
    else:
        print(f"  {n_matched}/{n_total} matched (no valid L2-interp neighborhoods)")

    return result


# ---------------------------------------------------------------------------
# 3. SST / SSS sampling  (with climatological fallback)
# ---------------------------------------------------------------------------

SST_DATE_RE = re.compile(r"AQUA_MODIS\.(\d{8})\.L3m")
SSS_DATE_RE = re.compile(r"SMAP_L3_SSS_(\d{8})_")

# Climatological defaults used when SST/SSS files are absent.
# Pure-water absorption a_w(T) has a ~0.001 m⁻¹/°C sensitivity in the blue,
# so a 5°C error shifts a_w by 0.005 m⁻¹ — small relative to phytoplankton
# absorption but non-zero. Using 20°C and 35 PSU is "open-ocean mid-latitude".
CLIM_SST_C = 20.0
CLIM_SSS_PSU = 35.0


def _build_date_file_map(directory: Path, pattern: re.Pattern) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    for f in sorted(directory.glob("*.nc")) + sorted(directory.glob("*.nc4")):
        m = pattern.search(f.name)
        if m:
            mapping[int(m.group(1))] = f
    return mapping


def _sample_nearest_valid(
    da: xr.DataArray, lat: float, lon: float,
    lat_coord: str, lon_coord: str, max_cells: int = 3,
) -> float:
    """
    Sample the nearest non-NaN value from a gridded DataArray.

    SMAP SSS has extensive NaN masking within ~75 km of coastlines (L-band
    can't measure salinity close to land). This function searches progressively
    larger neighborhoods to find a valid value.
    """
    nearest = da.sel({lat_coord: lat, lon_coord: lon}, method="nearest")
    val = float(nearest.values)
    if np.isfinite(val):
        return val

    lat_arr = da[lat_coord].values
    lon_arr = da[lon_coord].values
    lat_idx = int(np.argmin(np.abs(lat_arr - lat)))
    lon_idx = int(np.argmin(np.abs(lon_arr - lon)))

    lat_lo = max(0, lat_idx - max_cells)
    lat_hi = min(len(lat_arr), lat_idx + max_cells + 1)
    lon_lo = max(0, lon_idx - max_cells)
    lon_hi = min(len(lon_arr), lon_idx + max_cells + 1)

    window = da.isel({lat_coord: slice(lat_lo, lat_hi), lon_coord: slice(lon_lo, lon_hi)})
    vals = window.values
    wlat = window[lat_coord].values
    wlon = window[lon_coord].values
    dlat = wlat - lat
    dlon = (wlon - lon) * np.cos(np.radians(lat))
    dlat2, dlon2 = np.meshgrid(dlat, dlon, indexing="ij")
    dist2 = np.where(np.isfinite(vals), dlat2**2 + dlon2**2, np.inf)
    if np.all(np.isinf(dist2)):
        return np.nan
    i, j = np.unravel_index(int(np.argmin(dist2)), vals.shape)
    return float(vals[i, j])


def sample_sst_sss(
    matchup_df: pd.DataFrame,
    sst_dir: Path,
    sss_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample SST (AQUA MODIS daily) and SSS (SMAP 8-day) at matchup locations.

    Falls back to climatological SST=20°C, SSS=35 PSU when files are absent.
    Returns arrays of length len(matchup_df); never filters rows.
    """
    n = len(matchup_df)
    sst_vals = np.full(n, CLIM_SST_C)
    sss_vals = np.full(n, CLIM_SSS_PSU)

    if not sst_dir.exists() or not sss_dir.exists():
        return sst_vals, sss_vals

    sst_map = _build_date_file_map(sst_dir, SST_DATE_RE)
    sss_map = _build_date_file_map(sss_dir, SSS_DATE_RE)

    if not sst_map and not sss_map:
        print(f"    No SST/SSS files in {sst_dir.parent.name} — using climatological defaults")
        return sst_vals, sss_vals

    sst_cache: dict[int, xr.DataArray] = {}
    sss_cache: dict[int, xr.DataArray] = {}
    open_dsets: list[xr.Dataset] = []

    for i, (_, row) in enumerate(matchup_df.iterrows()):
        date_int = int(row["date"])
        lat, lon = row["lat"], row["lon"]

        if date_int in sst_map and date_int not in sst_cache:
            ds = xr.open_dataset(sst_map[date_int])
            open_dsets.append(ds)
            sst_cache[date_int] = ds["sst"]
        if date_int in sst_cache:
            v = _sample_nearest_valid(sst_cache[date_int], lat, lon, "lat", "lon")
            if np.isfinite(v):
                sst_vals[i] = v

        if date_int in sss_map and date_int not in sss_cache:
            ds = xr.open_dataset(sss_map[date_int])
            open_dsets.append(ds)
            sss_cache[date_int] = ds["smap_sss"]
        if date_int in sss_cache:
            v = _sample_nearest_valid(sss_cache[date_int], lat, lon, "latitude", "longitude")
            if np.isfinite(v):
                sss_vals[i] = v

    for ds in open_dsets:
        ds.close()

    n_clim_sst = int(np.isclose(sst_vals, CLIM_SST_C).sum())
    n_clim_sss = int(np.isclose(sss_vals, CLIM_SSS_PSU).sum())
    if n_clim_sst or n_clim_sss:
        print(
            f"    {n_clim_sst}/{n} using climatological SST, "
            f"{n_clim_sss}/{n} using climatological SSS"
        )
    return sst_vals, sss_vals


# ---------------------------------------------------------------------------
# 4. Rrs → 2nd derivative features
# ---------------------------------------------------------------------------

def preprocess_rrs_to_features(
    rrs_df: pd.DataFrame,
    sst: np.ndarray,
    sss: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Convert satellite Rrs to SDP 2nd-derivative features.

    1. GSM physics inversion (get_rrs_residuals): fit the Garver–Siegel–Maritorena
       bio-optical model per sample, subtract to isolate pigment-specific signal.
    2. 2nd difference (np.diff n=2 along wavelength axis): sharpens narrow
       pigment absorption bands and suppresses broad scattering baselines.

    Input rrs_df: (n_samples, 301) for wavelengths 400–700 nm at 1-nm steps.
    Output X: (n_samples, 299) — two fewer columns due to the 2nd difference.
    """
    _, RrsD = get_rrs_residuals(rrs_df, sst, sss, wavelengths)
    X = np.diff(RrsD, 2, axis=0).T   # (301-2, n) → transpose → (n, 299)
    return X


# ---------------------------------------------------------------------------
# 5a. Strict QC filters
# ---------------------------------------------------------------------------

def apply_strict_qc(
    rrs_array: np.ndarray,
    hplc_df: pd.DataFrame,
    matched_ids: list[str],
    pigments: list[str],
    qc_cfg: dict,
    source_name: str,
) -> np.ndarray:
    """
    Apply pre-GSM quality control filters on raw Rrs and HPLC values.

    Filters are applied in cascade order (cheapest first):
      1. HPLC concentration bounds — reject negative or implausibly high values
      2. Negative Rrs rejection — any Rrs(λ) < 0 indicates atm. correction failure
      3. Rrs magnitude check — any Rrs(λ) > max_rrs_sr indicates glint/foam
      4. Spectral shape check — Rrs(670) > Rrs(490) indicates non-Case-1 water

    Args:
        rrs_array: Raw Rrs, shape (n_samples, 301) for 400–700 nm.
        hplc_df: HPLC DataFrame indexed by sample IDs.
        matched_ids: Sample IDs aligned with rrs_array rows.
        pigments: List of pigment names to check for bounds.
        qc_cfg: The CONFIG["strict_qc"] dict.
        source_name: For logging.

    Returns:
        Boolean keep mask, shape (n_samples,). True = sample passes QC.
    """
    n = len(matched_ids)
    keep = np.ones(n, dtype=bool)

    print(f"  Strict QC for {source_name} ({n} samples):")

    # --- 1. HPLC concentration bounds ---
    hplc_limits = qc_cfg.get("hplc_max_concentration", {})
    default_max = hplc_limits.get("default", 10.0)
    hplc_sub = hplc_df.loc[matched_ids]
    # NaN comparisons return False in numpy, so missing pigment values
    # pass through — they're excluded per-pigment later in run_all_cv().
    hplc_bad = np.zeros(n, dtype=bool)
    for pig in pigments:
        if pig not in hplc_sub.columns:
            continue
        vals = hplc_sub[pig].values.astype(float)
        max_val = hplc_limits.get(pig, default_max)
        hplc_bad |= (vals < 0) | (vals > max_val)
    n_before = int(keep.sum())
    keep &= ~hplc_bad
    n_after = int(keep.sum())
    print(f"    HPLC bounds:      {n_before} → {n_after} ({n_before - n_after} removed)")

    # --- 2. Negative Rrs rejection ---
    if qc_cfg.get("negative_rrs_reject", True):
        neg_rrs = np.any(rrs_array < 0, axis=1)
        n_before = int(keep.sum())
        keep &= ~neg_rrs
        n_after = int(keep.sum())
        print(f"    Negative Rrs:     {n_before} → {n_after} ({n_before - n_after} removed)")

    # --- 3. Rrs magnitude check ---
    max_rrs = qc_cfg.get("max_rrs_sr", 0.05)
    high_rrs = np.any(rrs_array > max_rrs, axis=1)
    n_before = int(keep.sum())
    keep &= ~high_rrs
    n_after = int(keep.sum())
    print(f"    Rrs magnitude:    {n_before} → {n_after} ({n_before - n_after} removed)")

    # --- 4. Spectral shape check: Rrs(670) > Rrs(490) ---
    if qc_cfg.get("spectral_shape_check", True):
        # Index 270 = 670 nm (400 + 270), index 90 = 490 nm (400 + 90)
        shape_bad = rrs_array[:, 270] > rrs_array[:, 90]
        n_before = int(keep.sum())
        keep &= ~shape_bad
        n_after = int(keep.sum())
        print(f"    Spectral shape:   {n_before} → {n_after} ({n_before - n_after} removed)")

    print(f"    Total kept: {int(keep.sum())}/{n}")
    return keep


def compute_gsm_mapd(
    rrs_df: pd.DataFrame,
    sst: np.ndarray,
    sss: np.ndarray,
    wavelengths: np.ndarray,
    wl_range: tuple[int, int] = (400, 600),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute GSM model fit quality and return residuals for reuse.

    Runs GSM bio-optical inversion via get_rrs_residuals(), then computes
    the Mean Absolute Percent Difference (MAPD) between measured and modeled
    Rrs over the specified wavelength range. MAPD quantifies how well the
    bio-optical model explains the observed spectrum — high MAPD means the
    residuals (our features) carry more noise than signal.

    Only uses the 400–600 nm range by default because Rrs approaches zero
    in the red/NIR, making percent differences unreliable there.

    Args:
        rrs_df: Measured Rrs, shape (n_samples, 301), columns = wavelengths.
        sst: Sea surface temperature array, shape (n_samples,).
        sss: Sea surface salinity array, shape (n_samples,).
        wavelengths: 1-D array of wavelengths (400–700 nm).
        wl_range: (min_wl, max_wl) for MAPD calculation.

    Returns:
        mapd: MAPD per sample (%), shape (n_samples,).
        RrsD: Above-surface residual matrix (n_wavelengths, n_samples) for
              downstream 2nd-derivative feature computation (avoids re-running
              the expensive GSM inversion).
    """
    _, RrsD, modRrs = get_rrs_residuals(rrs_df, sst, sss, wavelengths, return_modeled=True)
    # RrsD is a DataFrame (Rrs.T - modRrs); convert to ndarray for downstream
    # numpy indexing (column slicing with boolean masks).
    RrsD = RrsD.values if hasattr(RrsD, "values") else np.asarray(RrsD)

    # Select wavelength range for MAPD (avoid noisy red/NIR)
    wl_mask = (wavelengths >= wl_range[0]) & (wavelengths <= wl_range[1])
    measured = rrs_df.values[:, wl_mask]         # (n_samples, n_wl_subset)
    modeled = modRrs[wl_mask, :].T               # (n_wl_subset, n_samples).T → (n_samples, n_wl_subset)

    # MAPD: mean(|measured - modeled| / max(|measured|, 1e-10)) * 100
    denom = np.maximum(np.abs(measured), 1e-10)
    mapd = np.mean(np.abs(measured - modeled) / denom, axis=1) * 100.0

    return mapd, RrsD


# ---------------------------------------------------------------------------
# 5b. Full data loading orchestrator
# ---------------------------------------------------------------------------

def load_all_data(
    sources: list[dict],
    pigments: list[str],
    spectral_cfg: dict,
    temporal_window_days: int,
    rrs_level: str = "L3",
    strict_qc: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, list[str]]:
    """
    Run the full data pipeline across all sources.

    Per source:
      1. Load + surface-filter HPLC
      2. Match to satellite Rrs pixels (L3 or L2 depending on rrs_level)
      3. Sample SST/SSS (with climatological fallback)
      4. (Optional) Apply strict QC filters on raw Rrs and HPLC
      5. Preprocess Rrs → 2nd-derivative features

    Combine everything into a single feature matrix X and ground-truth dict.

    Args:
        rrs_level: "L3" for gridded 0.1-deg daily, "L2" for swath-level ~1 km,
                   or "L2-interp" for neighborhood-averaged L2 within a radius.
                   Kramer source always uses in-situ Rrs regardless.
        strict_qc: If True, apply strict QC filters (negative Rrs, HPLC bounds,
                   spectral shape, GSM MAPD threshold, tighter temporal window).

    Returns:
        X            : feature matrix, shape (n_total, 299)
        y_dict       : {pigment: observed array, shape (n_total,)}
        source_labels: integer source index per sample, shape (n_total,) —
                       used for stratified k-fold splitting
        sample_ids   : list of unique string IDs for diagnostics
    """
    wavelengths = np.arange(400, 701)
    rrs_cols = [f"rrs_{i}" for i in range(301)]

    hplc_by_source = load_all_hplc(sources, pigments)

    all_X_parts: list[np.ndarray] = []
    all_y_parts: dict[str, list[np.ndarray]] = {p: [] for p in pigments}
    all_source_labels: list[int] = []
    all_sample_ids: list[str] = []

    for src_idx, src in enumerate(sources):
        name = src["name"]
        if name not in hplc_by_source:
            continue

        hplc_df = hplc_by_source[name]

        print(f"\n--- Source: {name} ---")

        if src["hplc_format"] == "kramer":
            # Rrs (400–700 nm, 1 nm) and SST/SSS are embedded in the HPLC CSV.
            # Skip satellite matchup entirely and use the in-situ values directly.
            matchup_df = hplc_df[
                ["station", "lat", "lon", "date"] + rrs_cols
            ].copy().reset_index(drop=True)
            sst_vals = hplc_df["sst"].values.astype(float)
            sss_vals = hplc_df["sss"].values.astype(float)
            matched_ids = hplc_df["station"].tolist()
            print(f"  Using in-situ Rrs directly ({len(matched_ids)} samples, no satellite matchup)")
        else:
            sst_dir = PROJECT_ROOT / src["sst_dir"]
            sss_dir = PROJECT_ROOT / src["sss_dir"]

            if rrs_level in ("L2", "L2-interp"):
                l2_rrs_dir_str = src.get("l2_rrs_dir")
                if not l2_rrs_dir_str:
                    print(f"  Skipping {name}: no l2_rrs_dir configured")
                    continue
                l2_rrs_dir = PROJECT_ROOT / l2_rrs_dir_str

                if rrs_level == "L2-interp":
                    l2i_cfg = CONFIG["multi_source"]["l2_interp"]
                    matchup_df = load_l2_interp_rrs_matchups(
                        l2_rrs_dir, hplc_df, spectral_cfg, l2i_cfg,
                    )
                    if matchup_df.empty:
                        print(f"  Skipping {name}: no L2-interp matchups (run download_l2_rrs.py?)")
                        continue
                else:
                    l2_cfg = CONFIG["multi_source"]["l2"]
                    matchup_df = load_l2_rrs_matchups(
                        l2_rrs_dir, hplc_df, spectral_cfg, l2_cfg,
                    )
                    if matchup_df.empty:
                        print(f"  Skipping {name}: no L2 matchups (run download_l2_rrs.py?)")
                        continue
            else:
                rrs_dir = PROJECT_ROOT / src["rrs_dir"]
                # Strict QC can tighten the temporal window (e.g. ±3d → ±1d)
                tw = temporal_window_days
                if strict_qc:
                    tw = CONFIG.get("strict_qc", {}).get("temporal_window_days", tw)
                sqc = CONFIG.get("strict_qc", {})
                cv_win = sqc.get("spatial_cv_window", 3)
                cv_wl = tuple(sqc.get("spatial_cv_wl_range", [440, 560]))
                matchup_df = load_l3_rrs_matchups(
                    rrs_dir, hplc_df, spectral_cfg, tw,
                    compute_spatial_cv=strict_qc,
                    cv_window=cv_win,
                    cv_wl_range=cv_wl,
                )
                if matchup_df.empty:
                    print(f"  Skipping {name}: no L3 matchups (run download_l3_rrs.py?)")
                    continue

            # Spatial quality filters (strict QC, satellite-matched data only)
            if strict_qc:
                qc_cfg = CONFIG["strict_qc"]
                n_before = len(matchup_df)

                # Pixel distance filter
                max_dist = qc_cfg.get("max_pixel_dist_km")
                if max_dist is not None and "pixel_dist_km" in matchup_df.columns:
                    matchup_df = matchup_df[matchup_df["pixel_dist_km"] <= max_dist]
                    n_after = len(matchup_df)
                    print(f"    Pixel dist ≤{max_dist} km: {n_before} → {n_after} ({n_before - n_after} removed)")
                    n_before = n_after

                # Spatial CV filter (NaN CV = too few valid neighbours → keep)
                max_cv = qc_cfg.get("max_spatial_cv")
                if max_cv is not None and "spatial_cv" in matchup_df.columns:
                    cv_bad = matchup_df["spatial_cv"].notna() & (matchup_df["spatial_cv"] > max_cv)
                    matchup_df = matchup_df[~cv_bad]
                    n_after = len(matchup_df)
                    print(f"    Spatial CV ≤{max_cv}: {n_before} → {n_after} ({n_before - n_after} removed)")

                matchup_df = matchup_df.reset_index(drop=True)
                if matchup_df.empty:
                    print(f"  {name}: all matchups rejected by spatial QC — skipping")
                    continue

            print(f"  Sampling SST/SSS for {len(matchup_df)} matchups...")
            sst_vals, sss_vals = sample_sst_sss(matchup_df, sst_dir, sss_dir)
            matched_ids = matchup_df["station"].tolist()

        rrs_array = matchup_df[rrs_cols].values

        if strict_qc:
            qc_cfg = CONFIG["strict_qc"]

            # Pre-GSM filters (cheap, on raw Rrs + HPLC)
            keep = apply_strict_qc(rrs_array, hplc_df, matched_ids, pigments, qc_cfg, name)
            rrs_array = rrs_array[keep]
            matched_ids = [mid for mid, k in zip(matched_ids, keep) if k]
            sst_vals, sss_vals = sst_vals[keep], sss_vals[keep]

            if not matched_ids:
                print(f"  {name}: all samples rejected by strict QC — skipping")
                continue

            # GSM residual check (expensive, but we reuse the residuals)
            print(f"  Computing GSM MAPD for {len(matched_ids)} samples...")
            rrs_df_src = pd.DataFrame(rrs_array, columns=wavelengths)
            wl_range = tuple(qc_cfg.get("gsm_mapd_wl_range", [400, 600]))
            mapd, RrsD = compute_gsm_mapd(rrs_df_src, sst_vals, sss_vals, wavelengths, wl_range)

            threshold = qc_cfg.get("gsm_mapd_threshold", 33.0)
            print(f"    MAPD stats: median={np.median(mapd):.1f}%, mean={np.mean(mapd):.1f}%, max={np.max(mapd):.1f}%")
            gsm_keep = mapd <= threshold
            n_reject = int((~gsm_keep).sum())
            print(f"    GSM MAPD ≤{threshold}%: {len(mapd)} → {int(gsm_keep.sum())} ({n_reject} removed)")

            rrs_array = rrs_array[gsm_keep]
            matched_ids = [mid for mid, k in zip(matched_ids, gsm_keep) if k]
            sst_vals, sss_vals = sst_vals[gsm_keep], sss_vals[gsm_keep]
            RrsD = RrsD[:, gsm_keep]

            if not matched_ids:
                print(f"  {name}: all samples rejected by GSM MAPD — skipping")
                continue

            # 2nd-derivative features from pre-computed residuals (no double GSM call)
            X_src = np.diff(RrsD, 2, axis=0).T
        else:
            print(f"  Preprocessing Rrs → 2nd derivative features...")
            rrs_df_src = pd.DataFrame(rrs_array, columns=wavelengths)
            X_src = preprocess_rrs_to_features(rrs_df_src, sst_vals, sss_vals, wavelengths)

        all_X_parts.append(X_src)
        all_sample_ids.extend(matched_ids)
        all_source_labels.extend([src_idx] * len(matched_ids))

        for pig in pigments:
            if pig in hplc_df.columns:
                y_src = hplc_df.loc[matched_ids, pig].values.astype(float)
            else:
                y_src = np.full(len(matched_ids), np.nan)
            all_y_parts[pig].append(y_src)

        print(f"  {name}: {len(matched_ids)} samples added  X.shape={X_src.shape}")

    if not all_X_parts:
        raise RuntimeError(
            f"No matchups from any source. Check that {rrs_level} Rrs files are downloaded."
        )

    X = np.vstack(all_X_parts)
    y_dict = {pig: np.concatenate(all_y_parts[pig]) for pig in pigments}
    source_labels = np.array(all_source_labels)

    print(f"\n{'='*60}")
    print(f"Combined dataset: {X.shape[0]} samples × {X.shape[1]} features")
    src_names = [src["name"] for src in sources]
    for idx, name in enumerate(src_names):
        n_src = int((source_labels == idx).sum())
        if n_src:
            print(f"  {name}: {n_src}")
    print(f"{'='*60}")

    return X, y_dict, source_labels, all_sample_ids


# ---------------------------------------------------------------------------
# 6. k-fold CV runners
# ---------------------------------------------------------------------------

def run_kfold_sklearn(
    X: np.ndarray,
    y: np.ndarray,
    method_name: str,
    config: dict,
    kfold: KFold,
) -> tuple[np.ndarray, np.ndarray]:
    """
    5-Fold CV for a sklearn method on one pigment.

    For each fold, train on the 80% training set and predict on the 20% test
    set. The predictions are accumulated into a single array of length n, where
    each element is the out-of-fold prediction for that sample.

    KFold shuffles once (random_state fixed in config); inner hyperparameter
    selection (e.g., RidgeCV LOO, KernelRidge GridSearch) is re-run inside
    train_sklearn_model on each fold's training set.

    Returns:
        (predictions, observed): both shape (n,).
    """
    n = len(X)
    predictions = np.full(n, np.nan)

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(X)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test = X[test_idx]

        model_info = train_sklearn_model(X_train, y_train, method_name, config)
        pred = predict_sklearn_model(X_test, model_info)
        predictions[test_idx] = pred

    return predictions, y.copy()


# ---------------------------------------------------------------------------
# 7. Main CV orchestrator
# ---------------------------------------------------------------------------

def run_all_cv(
    X: np.ndarray,
    y_dict: dict[str, np.ndarray],
    config: dict,
    tchla: np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """
    Run k-fold CV for all methods × all pigments.

    If tchla is provided and chl_conditioning is enabled in config, non-Tchla
    pigments get an augmented feature matrix with log10(Tchla) appended. Tchla
    itself is always predicted from spectral features alone to avoid circularity.

    Args:
        X: Spectral feature matrix, shape (n_total, 299).
        y_dict: {pigment_name: observed values, shape (n_total,)}.
        config: Full CONFIG dict.
        tchla: HPLC Tchla array for Tchla conditioning, shape (n_total,).
               If None, no conditioning is applied.

    Returns:
        gof_table     : DataFrame, one row per pigment, columns per method metric.
        all_predictions: nested dict [method][pigment] = prediction array.
    """
    ms_cfg = config["multi_source"]
    pigments = ms_cfg["pigments"]
    methods = config["sklearn"]["methods"]

    chl_cfg = config.get("chl_conditioning", {})
    chl_enabled = chl_cfg.get("enabled", False) and tchla is not None
    chl_target = chl_cfg.get("target_pigment", "Tchla")

    k = config["cv"]["kfold_k"]
    seed = config["cv"]["kfold_seed"]
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)

    all_predictions: dict[str, dict[str, np.ndarray]] = {m: {} for m in methods}
    rows = []

    for pig in pigments:
        y = y_dict.get(pig)
        if y is None or np.all(np.isnan(y)):
            print(f"\n  Skipping {pig}: no ground truth")
            continue

        # Determine whether to augment features with Tchla for this pigment
        use_chl = chl_enabled and pig != chl_target

        if use_chl:
            # Require both valid target AND valid Tchla for conditioning
            valid_mask = np.isfinite(y) & np.isfinite(tchla)
            log_tchla = np.log10(np.clip(tchla, 1e-6, None))
            X_pig = np.column_stack([X[valid_mask], log_tchla[valid_mask]])
        else:
            valid_mask = np.isfinite(y)
            X_pig = X[valid_mask]

        n_valid = int(valid_mask.sum())
        if n_valid < k * 2:
            print(f"\n  Skipping {pig}: only {n_valid} valid samples (need >= {k*2})")
            continue

        y_pig = y[valid_mask]
        feat_label = f"{X_pig.shape[1]} features" + (" [+Tchla]" if use_chl else "")
        row: dict = {"pigment": pig, "n_samples": n_valid}

        for method in methods:
            print(f"\n  {method} | {pig} (n={n_valid}, {feat_label})")
            t0 = time.time()
            preds, obs = run_kfold_sklearn(X_pig, y_pig, method, config, kfold)
            elapsed = time.time() - t0
            gof = compute_gof(preds, obs)
            row[f"{method}_R2"]       = gof["R2"]
            row[f"{method}_RMSE"]     = gof["RMSE"]
            row[f"{method}_MAE"]      = gof["MAE"]
            row[f"{method}_pct_bias"] = gof["pct_bias"]
            all_predictions[method][pig] = preds
            print(f"    R²={gof['R2']:.3f}  MAE={gof['MAE']:.4f}  ({elapsed:.1f}s)")

        rows.append(row)

    return pd.DataFrame(rows), all_predictions


# ---------------------------------------------------------------------------
# 8. Plotting
# ---------------------------------------------------------------------------

METHOD_COLORS = {
    "SDP":        "#4C72B0",
    "PLS":        "#DD8452",
    "ElasticNet": "#55A868",
    "HistGBT":    "#C44E52",
    "Ridge":      "#CCB974",
    "KernelRidge":"#64B5CD",
    "SVR":        "#8172B3",
}


def plot_scatter_grid(
    all_predictions: dict[str, dict[str, np.ndarray]],
    y_dict: dict[str, np.ndarray],
    pigments: list[str],
) -> None:
    """Grid of scatter plots: methods (rows) × pigments (columns)."""
    methods = [m for m in all_predictions if all_predictions[m]]
    if not methods or not pigments:
        return

    fig, axes = plt.subplots(
        len(methods), len(pigments),
        figsize=(3.5 * len(pigments), 3.5 * len(methods)),
        squeeze=False,
    )

    for row, method in enumerate(methods):
        for col, pig in enumerate(pigments):
            ax = axes[row, col]
            pred = all_predictions[method].get(pig)
            obs = y_dict.get(pig)

            if pred is None or obs is None:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
                continue

            valid = np.isfinite(obs)
            if valid.sum() < 2:
                ax.text(0.5, 0.5, "n<2", transform=ax.transAxes, ha="center")
                continue

            obs_v = obs[valid]
            pred_v = pred   # already filtered to valid samples during CV

            color = METHOD_COLORS.get(method, "#333333")
            ax.scatter(obs_v, pred_v, alpha=0.6, s=22, c=color,
                       edgecolors="k", linewidths=0.3)

            lo = min(obs_v.min(), pred_v.min(), 0)
            hi = max(obs_v.max(), pred_v.max())
            margin = (hi - lo) * 0.05
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=0.8)
            ax.set_xlim(lo - margin, hi + margin)
            ax.set_ylim(lo - margin, hi + margin)

            gof = compute_gof(pred_v, obs_v)
            ax.text(0.05, 0.92, f"R$^2$={gof['R2']:.2f}\nn={len(pred_v)}",
                    transform=ax.transAxes, fontsize=7, va="top")

            if row == 0:
                ax.set_title(pig, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{method}\nPredicted", fontsize=9)
            if row == len(methods) - 1:
                ax.set_xlabel("Observed", fontsize=9)

    fig.suptitle("Multi-Source 5-Fold CV: Predicted vs Observed", fontsize=13)
    fig.tight_layout()
    out = OUTPUT_DIR / "scatter_grid.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_metric_bars(gof_table: pd.DataFrame, metric: str, ylabel: str) -> None:
    """Grouped bar chart: one group per pigment, one bar per method."""
    method_cols = [c for c in gof_table.columns if c.endswith(f"_{metric}")]
    methods = [c.replace(f"_{metric}", "") for c in method_cols]
    pigments = gof_table["pigment"].tolist()

    x = np.arange(len(pigments))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(max(10, len(pigments) * 1.2), 5))
    for i, method in enumerate(methods):
        vals = gof_table[f"{method}_{metric}"].values
        offset = (i - len(methods) / 2 + 0.5) * width
        color = METHOD_COLORS.get(method, "#333333")
        ax.bar(x + offset, vals, width, label=method, color=color,
               alpha=0.8, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(pigments, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Multi-Source 5-Fold CV: {ylabel} by Pigment")
    ax.legend()
    fig.tight_layout()

    out = OUTPUT_DIR / f"{metric}_by_pigment.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def main() -> None:
    global OUTPUT_DIR  # allow dynamic override based on --rrs-level

    parser = argparse.ArgumentParser(description="Multi-Source 5-Fold CV")
    parser.add_argument(
        "--rrs-level", choices=["L3", "L2", "L2-interp"], default="L3",
        help="Satellite Rrs resolution: L3 (0.1-deg gridded), L2 (~1 km swath), "
             "or L2-interp (neighborhood-averaged L2 within 3 km radius)",
    )
    parser.add_argument(
        "--no-tchla-conditioning", action="store_true",
        help="Disable Tchla conditioning even if enabled in config",
    )
    parser.add_argument(
        "--strict-qc", action="store_true",
        help="Apply strict QC filters (negative Rrs, GSM residual, HPLC bounds, spectral shape)",
    )
    args = parser.parse_args()
    rrs_level = args.rrs_level

    # Override config if --no-tchla-conditioning is passed
    if args.no_tchla_conditioning:
        CONFIG["chl_conditioning"]["enabled"] = False

    # Set output directory based on Rrs level, Tchla conditioning, and QC mode
    chl_on = CONFIG.get("chl_conditioning", {}).get("enabled", False)
    level_suffix = {"L2": "_l2", "L2-interp": "_l2_interp"}.get(rrs_level, "")
    suffix = f"multi_source_cv{level_suffix}"
    if not chl_on:
        suffix += "_no_tchla"
    if args.strict_qc:
        suffix += "_strict_qc"
    OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / suffix

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    ms_cfg = CONFIG["multi_source"]
    pigments = ms_cfg["pigments"]

    print("\n" + "=" * 60)
    qc_label = ", strict QC" if args.strict_qc else ""
    print(f"Multi-Source 5-Fold Cross-Validation  (Rrs level: {rrs_level}{qc_label})")
    print("=" * 60)

    X, y_dict, source_labels, sample_ids = load_all_data(
        sources=SOURCES,
        pigments=pigments,
        spectral_cfg=ms_cfg["spectral"],
        temporal_window_days=ms_cfg["temporal_window_days"],
        rrs_level=rrs_level,
        strict_qc=args.strict_qc,
    )

    n_total = X.shape[0]
    n_methods = len(CONFIG["sklearn"]["methods"])
    k = CONFIG["cv"]["kfold_k"]
    print(f"\nRunning {k}-fold CV: {n_total} samples × {n_methods} methods × {len(pigments)} pigments")

    chl_cfg = CONFIG.get("chl_conditioning", {})
    if chl_cfg.get("enabled", False):
        chl_pig = chl_cfg.get("target_pigment", "Tchla")
        tchla = y_dict.get(chl_pig)
        n_chl_valid = int(np.isfinite(tchla).sum()) if tchla is not None else 0
        print(f"Tchla conditioning: ON  ({n_chl_valid}/{n_total} samples have valid Tchla)")
    else:
        tchla = None
        print("Tchla conditioning: OFF")

    gof_table, all_predictions = run_all_cv(X, y_dict, CONFIG, tchla=tchla)

    csv_path = OUTPUT_DIR / "cv_comparison_table.csv"
    gof_table.to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path}")

    print(f"\n{'='*60}")
    print(f"R² Summary ({k}-Fold CV)")
    print(f"{'='*60}")
    r2_cols = ["pigment", "n_samples"] + [c for c in gof_table.columns if c.endswith("_R2")]
    print(gof_table[r2_cols].to_string(index=False))

    print("\nGenerating plots...")
    plot_pigments = [
        pig for pig in pigments
        if pig in y_dict and np.isfinite(y_dict[pig]).sum() >= k * 2
    ]
    plot_scatter_grid(all_predictions, y_dict, plot_pigments)
    plot_metric_bars(gof_table, "R2", "R$^2$")
    plot_metric_bars(gof_table, "MAE", "MAE (µg/L)")

    total_elapsed = time.time() - total_start
    print(f"\nDone. Total time: {total_elapsed / 60:.1f} min")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
