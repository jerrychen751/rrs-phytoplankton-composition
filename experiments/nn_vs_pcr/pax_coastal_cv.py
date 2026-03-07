#!/usr/bin/env python3
"""
PAX Coastal Internal Cross-Validation.

Train and test entirely within the PAX coastal domain (~30 satellite-matched
HPLC samples from coastal California) to isolate the methodological comparison
(NN vs linear regression) without the domain-shift confound of the previous
Kramer → PAX experiment.

Uses Leave-One-Out Cross-Validation (LOOCV): train on N-1, predict the held-out
1, repeat N times. Standard in chemometrics for small spectral datasets —
deterministic and maximally data-efficient.

6 methods compared:
  - PCR (PCA + LinearRegression — what SDP does)
  - PLS (supervised dimensionality reduction)
  - ElasticNet (L1+L2 regularized linear)
  - HistGBT (gradient boosted trees)
  - SpectralCNN (1D conv NN)
  - TightPCAMLP (PCA + small MLP)

Usage:
    python experiments/nn_vs_pcr/pax_coastal_cv.py

Outputs: experiments/nn_vs_pcr/outputs/pax_coastal_cv/
    - pax_cv_comparison_table.csv
    - scatter_grid.png
    - R2_by_pigment.png
    - MAE_by_pigment.png
"""

from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

# --- Limit BLAS/LAPACK thread count BEFORE importing numpy/torch ---
# Our data is tiny (~30 × 299 matrices). At this scale, the overhead of
# spawning and synchronizing multiple threads exceeds any parallelism
# benefit. Without this cap, NumPy's SVD and PyTorch's matmuls each
# spawn 6-8 threads per call, and running many ensemble members in a loop
# drives CPU to 600%+.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no display needed
import matplotlib.pyplot as plt
import xarray as xr

import torch
torch.set_num_threads(2)  # Cap PyTorch's intra-op thread pool too

# Add project root to path so we can import project modules
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.sdp_pigments.core.physics import get_rrs_residuals
from utils.pace_l2 import preprocess_rrs_spectrum, haversine_km

from experiments.nn_vs_pcr.config import CONFIG
from experiments.nn_vs_pcr.nn_trainer import train_nn_ensemble, predict_nn_ensemble
from experiments.nn_vs_pcr.sklearn_models import train_sklearn_model, predict_sklearn_model
from experiments.nn_vs_pcr.evaluation import compute_gof

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "pax_coastal_cv"

# Column name mapping: PAX HPLC CSV names → SDP internal names.
# The PAX CSV uses instrument-style names (e.g., "Tot_Chl_a"), while SDP
# uses abbreviated names (e.g., "Tchla") matching Kramer et al. (2021).
HPLC_TO_SDP = {
    "Tot_Chl_a": "Tchla",
    "But-fuco": "ButFuco",
    "Hex-fuco": "HexFuco",
    "Allo": "Allo",
    "Fuco": "Fuco",
    "Perid": "Perid",
    "Zea": "Zea",
    "DV_Chl_a": "DVchla",
    "MV_Chl_b": "MVchlb",
    "Chl_c1c2": "Chlc12",
    "Chl_c3": "Chlc3",
    "Neo": "Neo",
    "Viola": "Viola",
}


# ---------------------------------------------------------------------------
# 1. HPLC loading
# ---------------------------------------------------------------------------

def load_and_average_hplc(hplc_path: Path) -> pd.DataFrame:
    """
    Load PAX HPLC CSV, rename columns to SDP names, average replicates.

    The PAX dataset has 3 technical replicates per station (108 rows for
    36 stations). Averaging them gives one ground-truth value per station,
    which is appropriate since each station maps to a single L3 pixel (~11 km).

    Pandas groupby().mean() ignores NaN by default, so if one replicate is
    missing a pigment value, the average is computed from the other two.

    Returns:
        DataFrame indexed by station number, with columns for each SDP
        pigment name plus lat, lon, and date.
    """
    df = pd.read_csv(hplc_path)

    # Rename pigment columns to SDP internal names
    rename_map = {k: v for k, v in HPLC_TO_SDP.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Identify which of the 13 SDP pigments are present in this dataset
    pigments = CONFIG["data"]["pigments"]
    available = [p for p in pigments if p in df.columns]

    # Average replicates per station.
    agg_cols = available + ["lat", "lon"]
    grouped = df.groupby("station")[agg_cols + ["date"]].agg({
        **{col: "mean" for col in agg_cols},
        "date": "first",
    })

    print(f"  Loaded {len(df)} HPLC samples -> {len(grouped)} stations (averaged replicates)")
    print(f"  Available pigments: {available}")

    return grouped


# ---------------------------------------------------------------------------
# 2. L3 Rrs matchup extraction
# ---------------------------------------------------------------------------

L3_DATE_RE = re.compile(r"PACE_OCI\.(\d{8})\.L3m")


def _parse_l3_date(filename: str) -> str | None:
    """
    Extract YYYYMMDD date string from an L3 filename.

    Example: "PACE_OCI.20240906.L3m.DAY.RRS.V3_1.Rrs.0p1deg.nc" → "20240906"
    """
    match = L3_DATE_RE.search(filename)
    return match.group(1) if match else None


def load_l3_rrs_matchups(
    rrs_dir: Path,
    hplc_df: pd.DataFrame,
    spectral_cfg: dict,
    temporal_window_days: int = 0,
) -> pd.DataFrame:
    """
    Match HPLC stations to L3 daily Rrs pixels.

    L3 mapped files are on a regular 0.1-degree lat/lon grid. We use xarray's
    built-in .sel(method="nearest") for the spatial lookup.

    For each station:
    1. Find the L3 file whose date matches the station's sampling date
    2. Select the nearest 0.1-degree pixel
    3. Interpolate the non-uniform 172-band spectrum to a uniform 1 nm grid
    4. Skip stations where the pixel is NaN (cloud/land masked)

    If temporal_window_days > 0 and the same-day pixel is NaN, tries nearby
    days (sorted by |offset| so the closest day wins).

    Args:
        rrs_dir: Directory containing PACE OCI L3 .nc files.
        hplc_df: HPLC DataFrame indexed by station.
        spectral_cfg: Dict with interp_nm, smooth_nm, edge_trim_nm, final_range_nm.
        temporal_window_days: Max days to search forward/backward.

    Returns:
        DataFrame with rrs columns plus metadata.
    """
    rrs_files = sorted(rrs_dir.glob("*.nc"))
    date_to_file: dict[int, Path] = {}
    for f in rrs_files:
        date_str = _parse_l3_date(f.name)
        if date_str:
            date_to_file[int(date_str)] = f

    print(f"  Found {len(date_to_file)} L3 Rrs files")
    if temporal_window_days > 0:
        print(f"  Temporal window: +/-{temporal_window_days} days")

    matchups = []

    for station, row in hplc_df.iterrows():
        date_int = int(row["date"])

        # Build candidate dates sorted by proximity to sample date
        candidate_offsets = [0] + [
            d for abs_d in range(1, temporal_window_days + 1)
            for d in (-abs_d, abs_d)
        ]

        spectrum = None
        pixel_lat = pixel_lon = None
        native_wl = None
        used_date = None
        used_offset = 0

        for offset in candidate_offsets:
            candidate_date = date_int + offset
            if candidate_date not in date_to_file:
                continue

            with xr.open_dataset(date_to_file[candidate_date]) as ds:
                pixel = ds["Rrs"].sel(
                    lat=row["lat"],
                    lon=row["lon"],
                    method="nearest",
                )
                spec = pixel.values
                if not np.all(np.isnan(spec)):
                    spectrum = spec
                    pixel_lat = float(pixel.lat.values)
                    pixel_lon = float(pixel.lon.values)
                    native_wl = ds.wavelength.values
                    used_date = candidate_date
                    used_offset = offset
                    break

        if spectrum is None:
            if date_int not in date_to_file:
                print(f"    Station {station}: no Rrs file for date {date_int}, skipping")
            else:
                window_str = f" (searched +/-{temporal_window_days}d)" if temporal_window_days > 0 else ""
                print(f"    Station {station}: pixel is all-NaN{window_str}, skipping")
            continue

        if used_offset != 0:
            print(f"    Station {station}: same-day cloudy, using {used_date} (offset {used_offset:+d}d)")

        dist_km = haversine_km(row["lat"], row["lon"], pixel_lat, pixel_lon)

        try:
            _, rrs_1nm = preprocess_rrs_spectrum(
                native_wl,
                spectrum,
                interp_nm=spectral_cfg["interp_nm"],
                smooth_nm=spectral_cfg["smooth_nm"],
                edge_trim_nm=spectral_cfg["edge_trim_nm"],
                final_range_nm=spectral_cfg["final_range_nm"],
            )
        except ValueError as e:
            print(f"    Station {station}: spectral preprocessing failed ({e}), skipping")
            continue

        if not np.isfinite(rrs_1nm).all():
            print(f"    Station {station}: processed spectrum has NaNs, skipping")
            continue

        matchup: dict = {
            "station": station,
            "lat": row["lat"],
            "lon": row["lon"],
            "date": date_int,
            "rrs_date": used_date,
            "date_offset": used_offset,
            "pixel_lat": pixel_lat,
            "pixel_lon": pixel_lon,
            "pixel_dist_km": dist_km,
        }
        for i, val in enumerate(rrs_1nm):
            matchup[f"rrs_{i}"] = val

        matchups.append(matchup)

    result = pd.DataFrame(matchups)
    print(f"  {len(result)} of {len(hplc_df)} stations matched to valid L3 pixels")

    if len(result) > 0:
        print(f"  Pixel distance range: {result['pixel_dist_km'].min():.1f} - {result['pixel_dist_km'].max():.1f} km")

    return result


# ---------------------------------------------------------------------------
# 3. SST / SSS sampling
# ---------------------------------------------------------------------------

SST_DATE_RE = re.compile(r"AQUA_MODIS\.(\d{8})\.L3m")
SSS_DATE_RE = re.compile(r"SMAP_L3_SSS_(\d{8})_")


def _sample_nearest_valid(
    da: xr.DataArray,
    lat: float,
    lon: float,
    lat_coord: str,
    lon_coord: str,
    max_cells: int = 3,
) -> float:
    """
    Sample the nearest non-NaN value from a gridded DataArray.

    SMAP SSS has extensive NaN masking near coastlines (L-band radiometry
    can't measure salinity within ~40-75 km of land). This function searches
    progressively larger neighborhoods until it finds a valid value.

    Args:
        da: 2-D DataArray with lat and lon coordinates.
        lat: Target latitude (degrees).
        lon: Target longitude (degrees).
        lat_coord: Name of latitude coordinate in the DataArray.
        lon_coord: Name of longitude coordinate in the DataArray.
        max_cells: Maximum number of grid cells to search in each direction.

    Returns:
        The nearest non-NaN value, or NaN if none found within the window.
    """
    # Fast path: single nearest cell
    nearest = da.sel({lat_coord: lat, lon_coord: lon}, method="nearest")
    val = float(nearest.values)
    if np.isfinite(val):
        return val

    # Slow path: search a neighborhood
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
    dlat_2d, dlon_2d = np.meshgrid(dlat, dlon, indexing="ij")
    dist2 = dlat_2d**2 + dlon_2d**2

    dist2 = np.where(np.isfinite(vals), dist2, np.inf)

    if np.all(np.isinf(dist2)):
        return np.nan

    flat_idx = int(np.argmin(dist2))
    i, j = np.unravel_index(flat_idx, vals.shape)
    return float(vals[i, j])


def _build_date_file_map(directory: Path, pattern: re.Pattern) -> dict[int, Path]:
    """
    Build a mapping from date integers to file paths using a regex pattern.
    """
    mapping: dict[int, Path] = {}
    for f in sorted(directory.glob("*.nc")) + sorted(directory.glob("*.nc4")):
        match = pattern.search(f.name)
        if match:
            mapping[int(match.group(1))] = f
    return mapping


def sample_sst_sss(
    matchup_df: pd.DataFrame,
    sst_dir: Path,
    sss_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample SST and SSS at matchup locations via nearest-neighbor lookup.

    Args:
        matchup_df: DataFrame with station, lat, lon, date columns.
        sst_dir: Directory containing daily AQUA MODIS SST .nc files.
        sss_dir: Directory containing SMAP SSS .nc4 files.

    Returns:
        (sst_values, sss_values, valid_mask) — each of length len(matchup_df).
        valid_mask is True where both SST and SSS are finite.
    """
    sst_map = _build_date_file_map(sst_dir, SST_DATE_RE)
    sss_map = _build_date_file_map(sss_dir, SSS_DATE_RE)

    print(f"  Found {len(sst_map)} SST files, {len(sss_map)} SSS files")

    n = len(matchup_df)
    sst_vals = np.full(n, np.nan)
    sss_vals = np.full(n, np.nan)

    sst_cache: dict[int, xr.DataArray] = {}
    sss_cache: dict[int, xr.DataArray] = {}
    open_datasets: list[xr.Dataset] = []

    for i, (_, row) in enumerate(matchup_df.iterrows()):
        date_int = int(row["date"])
        lat, lon = row["lat"], row["lon"]

        if date_int in sst_map:
            if date_int not in sst_cache:
                ds = xr.open_dataset(sst_map[date_int])
                open_datasets.append(ds)
                sst_cache[date_int] = ds["sst"]
            sst_vals[i] = _sample_nearest_valid(
                sst_cache[date_int], lat, lon, lat_coord="lat", lon_coord="lon",
            )

        if date_int in sss_map:
            if date_int not in sss_cache:
                ds = xr.open_dataset(sss_map[date_int])
                open_datasets.append(ds)
                sss_cache[date_int] = ds["smap_sss"]
            sss_vals[i] = _sample_nearest_valid(
                sss_cache[date_int], lat, lon, lat_coord="latitude", lon_coord="longitude",
            )

    for ds in open_datasets:
        ds.close()

    valid = np.isfinite(sst_vals) & np.isfinite(sss_vals)
    n_valid = int(valid.sum())
    n_missing_sst = int((~np.isfinite(sst_vals)).sum())
    n_missing_sss = int((~np.isfinite(sss_vals)).sum())

    print(f"  SST/SSS sampling: {n_valid} valid, {n_missing_sst} missing SST, {n_missing_sss} missing SSS")

    return sst_vals, sss_vals, valid


# ---------------------------------------------------------------------------
# 4. Preprocessing: Rrs → 2nd derivative features
# ---------------------------------------------------------------------------

def preprocess_rrs_to_features(
    rrs_df: pd.DataFrame,
    sst: np.ndarray,
    sss: np.ndarray,
    wavelengths: np.ndarray,
) -> np.ndarray:
    """
    Convert satellite Rrs to SDP-compatible 2nd derivative features.

    Steps:
    1. Physics residuals (get_rrs_residuals): fit GSM bio-optical model,
       subtract modeled Rrs to isolate pigment-specific spectral signals.
    2. 2nd derivative (np.diff with n=2): emphasizes narrow absorption features
       and suppresses broad baseline trends.

    Args:
        rrs_df: DataFrame with integer wavelength columns (400-700).
        sst: Sea surface temperature values, shape (n,).
        sss: Sea surface salinity values, shape (n,).
        wavelengths: np.arange(400, 701), shape (301,).

    Returns:
        X: Feature matrix, shape (n, 299).
    """
    _, RrsD = get_rrs_residuals(rrs_df, sst, sss, wavelengths)
    X = np.diff(RrsD, 2, axis=0).T  # (n_samples, 299)
    return X


# ---------------------------------------------------------------------------
# 5. Data loading orchestrator
# ---------------------------------------------------------------------------

def load_pax_data() -> tuple[np.ndarray, dict[str, np.ndarray], int]:
    """
    Full data pipeline: HPLC → L3 matchups → SST/SSS → features.

    Orchestrates the four loading/preprocessing steps to produce the final
    feature matrix X and pigment ground-truth dict y_dict.

    Returns:
        X: Feature matrix, shape (n_matchups, 299).
        y_dict: Dict mapping pigment name → observed concentrations array.
        n_matchups: Number of valid matchup samples.
    """
    data_cfg = CONFIG["data"]
    pigments = data_cfg["pigments"]

    # ---- Load HPLC ground truth ----
    print("\n" + "=" * 60)
    print("Loading HPLC ground truth")
    print("=" * 60)
    hplc_path = PROJECT_ROOT / data_cfg["hplc_path"]
    hplc_df = load_and_average_hplc(hplc_path)

    # ---- Extract L3 Rrs matchups ----
    print("\n" + "=" * 60)
    print("Extracting L3 Rrs matchups")
    print("=" * 60)
    rrs_dir = PROJECT_ROOT / data_cfg["rrs_dir"]
    matchup_df = load_l3_rrs_matchups(
        rrs_dir, hplc_df, data_cfg["spectral"],
        temporal_window_days=data_cfg["temporal_window_days"],
    )

    if matchup_df.empty:
        raise RuntimeError("No valid matchups found — check data paths and temporal window")

    # ---- Sample SST / SSS ----
    print("\n" + "=" * 60)
    print("Sampling SST / SSS")
    print("=" * 60)
    sst_dir = PROJECT_ROOT / data_cfg["sst_dir"]
    sss_dir = PROJECT_ROOT / data_cfg["sss_dir"]
    sst_vals, sss_vals, valid_mask = sample_sst_sss(matchup_df, sst_dir, sss_dir)

    # Filter to samples with complete SST + SSS
    matchup_df = matchup_df[valid_mask].reset_index(drop=True)
    sst_vals = sst_vals[valid_mask]
    sss_vals = sss_vals[valid_mask]
    n_matchups = len(matchup_df)
    print(f"  {n_matchups} matchups with complete Rrs + SST + SSS")

    if n_matchups == 0:
        raise RuntimeError("No matchups remaining after SST/SSS filtering")

    # Log temporal offsets
    if data_cfg["temporal_window_days"] > 0:
        offsets = matchup_df["date_offset"].values
        n_exact = int((offsets == 0).sum())
        n_shifted = int((offsets != 0).sum())
        print(f"  Temporal breakdown: {n_exact} same-day, {n_shifted} from adjacent days")

    # ---- Preprocess Rrs → 2nd derivative features ----
    print("\n" + "=" * 60)
    print("Preprocessing Rrs -> features")
    print("=" * 60)
    wavelengths = np.arange(400, 701)
    rrs_cols = [f"rrs_{i}" for i in range(301)]
    rrs_array = matchup_df[rrs_cols].values
    rrs_df = pd.DataFrame(rrs_array, columns=wavelengths)

    print("  Computing Rrs residuals...")
    X = preprocess_rrs_to_features(rrs_df, sst_vals, sss_vals, wavelengths)
    print(f"  X shape: {X.shape}")

    # ---- Collect HPLC ground truth for matched stations ----
    matched_stations = matchup_df["station"].values
    y_dict: dict[str, np.ndarray] = {}
    for pig in pigments:
        if pig in hplc_df.columns:
            y_dict[pig] = hplc_df.loc[matched_stations, pig].values
        else:
            y_dict[pig] = np.full(n_matchups, np.nan)

    return X, y_dict, n_matchups


# ---------------------------------------------------------------------------
# 6. LOOCV runners
# ---------------------------------------------------------------------------

def run_loocv_sklearn(
    X: np.ndarray,
    y: np.ndarray,
    method_name: str,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Leave-One-Out CV for a single sklearn method on one pigment.

    For each of the N samples, trains on the remaining N-1 and predicts
    the held-out sample. This is the standard LOOCV loop — deterministic,
    no randomness, and maximally data-efficient.

    The sklearn methods handle hyperparameter selection internally:
    - PCR/PLS: inner 5-fold CV on the N-1 training samples selects n_components
    - ElasticNet: ElasticNetCV does built-in CV for alpha and l1_ratio
    - HistGBT: early stopping on a validation fraction of the N-1 samples

    Args:
        X: Full feature matrix, shape (N, 299).
        y: Full target vector, shape (N,).
        method_name: One of "PCR", "PLS", "ElasticNet", "HistGBT".
        config: Full CONFIG dict.

    Returns:
        (predictions, observed): both shape (N,). predictions[i] is the
        model's prediction for sample i when trained without sample i.
    """
    n = len(X)
    predictions = np.zeros(n)
    observed = y.copy()

    for i in range(n):
        # Boolean mask: all samples except i
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        X_train, y_train = X[mask], y[mask]
        X_test = X[i:i+1]  # shape (1, 299) — keep 2D for sklearn

        model_info = train_sklearn_model(X_train, y_train, method_name, config)
        pred = predict_sklearn_model(X_test, model_info)
        predictions[i] = pred[0]

    return predictions, observed


def run_loocv_nn(
    X: np.ndarray,
    y: np.ndarray,
    arch_name: str,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Leave-One-Out CV for a single NN architecture on one pigment.

    Same LOOCV structure as run_loocv_sklearn, but each fold trains an
    ensemble of models (default 10 members) and takes the median prediction.
    The ensemble addresses NN training stochasticity — different random
    initializations and early-stopping splits produce different models, and
    the median is more robust than any single member.

    seed_base = 42 + fold_idx * 100 ensures that:
    1. Each fold uses different seeds (so different weight initializations)
    2. The gap of 100 between folds prevents seed overlap between ensemble
       members across folds (member i in fold j uses seed 42 + j*100 + i)

    Args:
        X: Full feature matrix, shape (N, 299).
        y: Full target vector, shape (N,).
        arch_name: One of "SpectralCNN", "TightPCAMLP".
        config: Full CONFIG dict.

    Returns:
        (predictions, observed): both shape (N,).
    """
    n = len(X)
    n_ensemble = config["nn"]["n_ensemble"]
    predictions = np.zeros(n)
    observed = y.copy()

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False

        X_train, y_train = X[mask], y[mask]
        X_test = X[i:i+1]

        # Train ensemble with fold-specific seed base
        seed_base = 42 + i * 100
        models = train_nn_ensemble(
            X_train, y_train, arch_name, config,
            n_ensemble=n_ensemble, seed_base=seed_base,
        )
        pred = predict_nn_ensemble(X_test, models, arch_name, config)
        predictions[i] = pred[0]

    return predictions, observed


# ---------------------------------------------------------------------------
# 7. Main CV orchestrator
# ---------------------------------------------------------------------------

def run_all_cv(
    X: np.ndarray,
    y_dict: dict[str, np.ndarray],
    config: dict,
) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
    """
    Run LOOCV for all methods × all pigments, collect results.

    Loops over every pigment and every method, running the appropriate LOOCV
    function (sklearn or NN). For each (method, pigment) pair, it stores the
    full prediction vector and computes GOF metrics.

    Args:
        X: Feature matrix, shape (N, 299).
        y_dict: Dict mapping pigment name → observed values, shape (N,).
        config: Full CONFIG dict.

    Returns:
        gof_table: DataFrame with one row per pigment, columns for each
            method's R2, RMSE, MAE, pct_bias.
        all_predictions: Nested dict all_predictions[method][pigment] = array.
    """
    pigments = config["data"]["pigments"]
    sklearn_methods = config["sklearn"]["methods"]
    nn_architectures = config["nn"]["architectures"]
    all_methods = sklearn_methods + nn_architectures

    # Store predictions for scatter plots
    all_predictions: dict[str, dict[str, np.ndarray]] = {m: {} for m in all_methods}

    rows = []

    for pig in pigments:
        y = y_dict.get(pig)
        if y is None or np.all(np.isnan(y)):
            print(f"\n  Skipping {pig}: no ground truth data")
            continue

        # Check for sufficient non-NaN samples
        valid_mask = np.isfinite(y)
        n_valid = int(valid_mask.sum())
        if n_valid < 5:
            print(f"\n  Skipping {pig}: only {n_valid} valid samples (need >= 5)")
            continue

        # Use only samples with valid ground truth
        X_pig = X[valid_mask]
        y_pig = y[valid_mask]

        row: dict = {"pigment": pig, "n_samples": n_valid}

        # --- Sklearn methods ---
        for method in sklearn_methods:
            print(f"\n  {method} | {pig} (n={n_valid})")
            t0 = time.time()
            preds, obs = run_loocv_sklearn(X_pig, y_pig, method, config)
            elapsed = time.time() - t0

            gof = compute_gof(preds, obs)
            row[f"{method}_R2"] = gof["R2"]
            row[f"{method}_RMSE"] = gof["RMSE"]
            row[f"{method}_MAE"] = gof["MAE"]
            row[f"{method}_pct_bias"] = gof["pct_bias"]

            all_predictions[method][pig] = preds
            print(f"    R²={gof['R2']:.3f}, MAE={gof['MAE']:.4f} ({elapsed:.1f}s)")

        # --- NN methods ---
        for arch in nn_architectures:
            print(f"\n  {arch} | {pig} (n={n_valid})")
            t0 = time.time()
            preds, obs = run_loocv_nn(X_pig, y_pig, arch, config)
            elapsed = time.time() - t0

            gof = compute_gof(preds, obs)
            row[f"{arch}_R2"] = gof["R2"]
            row[f"{arch}_RMSE"] = gof["RMSE"]
            row[f"{arch}_MAE"] = gof["MAE"]
            row[f"{arch}_pct_bias"] = gof["pct_bias"]

            all_predictions[arch][pig] = preds
            print(f"    R²={gof['R2']:.3f}, MAE={gof['MAE']:.4f} ({elapsed:.1f}s)")

        rows.append(row)

    gof_table = pd.DataFrame(rows)
    return gof_table, all_predictions


# ---------------------------------------------------------------------------
# 8. Plotting
# ---------------------------------------------------------------------------

# Color palette for 6 methods: chosen for distinguishability and consistency
# with the previous experiment's palette where possible.
METHOD_COLORS = {
    "PCR": "#4C72B0",         # steel blue
    "PLS": "#DD8452",         # muted orange
    "ElasticNet": "#55A868",  # sage green
    "HistGBT": "#C44E52",     # muted red
    "SpectralCNN": "#8172B3", # muted purple
    "TightPCAMLP": "#937860", # muted brown
}


def plot_scatter_grid(
    all_predictions: dict[str, dict[str, np.ndarray]],
    y_dict: dict[str, np.ndarray],
    pigments: list[str],
) -> None:
    """
    Grid of scatter plots: methods (rows) × pigments (columns).

    Each cell shows LOOCV predicted vs observed with a 1:1 reference line.
    Points above the line = over-prediction, below = under-prediction.
    R² and sample size are annotated in each cell.
    """
    methods = list(all_predictions.keys())
    # Only plot methods that have at least one pigment
    methods = [m for m in methods if len(all_predictions[m]) > 0]
    n_methods = len(methods)
    n_pigs = len(pigments)

    if n_methods == 0 or n_pigs == 0:
        print("  No data to plot scatter grid")
        return

    fig, axes = plt.subplots(
        n_methods, n_pigs,
        figsize=(3.5 * n_pigs, 3.5 * n_methods),
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

            # Filter to the same valid mask used during CV
            valid = np.isfinite(obs)
            if valid.sum() < 2:
                ax.text(0.5, 0.5, "n < 2", transform=ax.transAxes, ha="center")
                continue

            obs_v = obs[valid]
            pred_v = pred  # Already filtered during run_all_cv

            color = METHOD_COLORS.get(method, "#333333")
            ax.scatter(obs_v, pred_v, alpha=0.6, s=25, c=color,
                       edgecolors="k", linewidths=0.3)

            # 1:1 reference line
            lo = min(obs_v.min(), pred_v.min(), 0)
            hi = max(obs_v.max(), pred_v.max())
            margin = (hi - lo) * 0.05
            ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, linewidth=0.8)
            ax.set_xlim(lo - margin, hi + margin)
            ax.set_ylim(lo - margin, hi + margin)

            # Annotate R² and sample count
            gof = compute_gof(pred_v, obs_v)
            ax.text(
                0.05, 0.92,
                f"R$^2$={gof['R2']:.2f}\nn={len(pred_v)}",
                transform=ax.transAxes, fontsize=7, va="top",
            )

            if row == 0:
                ax.set_title(pig, fontsize=10)
            if col == 0:
                ax.set_ylabel(f"{method}\nPredicted", fontsize=9)
            if row == n_methods - 1:
                ax.set_xlabel("Observed", fontsize=9)

    fig.suptitle("PAX Coastal LOOCV: Predicted vs Observed", fontsize=13)
    fig.tight_layout()
    out = OUTPUT_DIR / "scatter_grid.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_metric_bars(
    gof_table: pd.DataFrame,
    metric: str,
    ylabel: str,
) -> None:
    """
    Grouped bar chart: one group per pigment, one bar per method.

    Makes it easy to visually compare how each method performs across
    pigments — spot which method wins for each pigment and whether any
    method consistently dominates.
    """
    method_cols = [c for c in gof_table.columns if c.endswith(f"_{metric}")]
    methods = [c.replace(f"_{metric}", "") for c in method_cols]
    pigments = gof_table["pigment"].tolist()

    x = np.arange(len(pigments))
    width = 0.8 / len(methods)

    fig, ax = plt.subplots(figsize=(max(10, len(pigments) * 1.2), 5))

    for i, method in enumerate(methods):
        col = f"{method}_{metric}"
        vals = gof_table[col].values
        offset = (i - len(methods) / 2 + 0.5) * width
        color = METHOD_COLORS.get(method, "#333333")
        ax.bar(x + offset, vals, width, label=method, color=color,
               alpha=0.8, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(pigments, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"PAX Coastal LOOCV: {ylabel} by Pigment")
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    # ---- Load and preprocess PAX data ----
    X, y_dict, n_matchups = load_pax_data()
    print(f"\n{'=' * 60}")
    print(f"Starting LOOCV with {n_matchups} samples, 6 methods, {len(CONFIG['data']['pigments'])} pigments")
    print(f"{'=' * 60}")

    # ---- Run all LOOCV ----
    gof_table, all_predictions = run_all_cv(X, y_dict, CONFIG)

    # ---- Save CSV ----
    csv_path = OUTPUT_DIR / "pax_cv_comparison_table.csv"
    gof_table.to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path}")

    # ---- Print R² summary ----
    print(f"\n{'=' * 60}")
    print("R² Summary (LOOCV)")
    print(f"{'=' * 60}")
    r2_cols = ["pigment"] + [c for c in gof_table.columns if c.endswith("_R2")]
    print(gof_table[r2_cols].to_string(index=False))

    # ---- Plots ----
    print("\nGenerating plots...")
    plot_pigments = [
        pig for pig in CONFIG["data"]["pigments"]
        if pig in y_dict and np.isfinite(y_dict[pig]).sum() >= 5
    ]

    plot_scatter_grid(all_predictions, y_dict, plot_pigments)
    plot_metric_bars(gof_table, "R2", "R$^2$")
    plot_metric_bars(gof_table, "MAE", "MAE (µg/L)")

    total_elapsed = time.time() - total_start
    print(f"\nAll done! Total time: {total_elapsed / 60:.1f} minutes")
    print(f"Results in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
