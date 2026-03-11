"""
- Rrs under netCDF groups (e.g., `geophysical_data/Rrs`)
- lat/lon are 2-D swath arrays (e.g., `navigation_data/latitude`, `longitude`)
- the model workflow requires *spectral preprocessing* (interpolation, smoothing,
  trimming) prior to taking a second derivative; see `docs/notes.md`

Expected inputs (under `io.input_dir/`):
  - `rrs/`: L2 AOP granules
  - `sst/`: gridded SST
  - `sss/`: gridded SSS
  - `validation.hplc`: in-situ HPLC dataset used for matchups

"""

import sys
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import xarray as xr
from utils.config_loader import (
    load_config_from_file,
    get_experiment_name,
    get_experiment_description,
    get_bbox,
    get_pigments,
    get_input_dir,
    get_output_dir,
    get_output_filename,
    resolve_path,
)

from utils.pace_l2 import (
    L2Matchup,
    extract_l2_matchup,
    list_nc_files as list_nc_paths,
    load_hplc_observations,
    select_candidate_granules,
)

def _is_pace_l2_rrs_file(path: Path) -> bool:
    name = path.name
    return ".L2." in name or ".L2_" in name


def _infer_lat_lon_names(da: xr.DataArray) -> tuple[str, str]:
    lat_candidates = ("lat", "latitude")
    lon_candidates = ("lon", "longitude")
    lat_name = next((n for n in lat_candidates if n in da.coords or n in da.dims), None)
    lon_name = next((n for n in lon_candidates if n in da.coords or n in da.dims), None)
    if lat_name is None or lon_name is None:
        raise KeyError(
            f"Unable to infer lat/lon coordinate names for {da.name}. "
            f"Found coords={list(da.coords)}, dims={list(da.dims)}"
        )
    return lat_name, lon_name


def _parse_time_attr(raw: str) -> pd.Timestamp:
    """
    Parse a time string from a NetCDF global attribute.

    Handles two common NASA conventions:
    - ISO 8601: ``2025-05-01T00:00:00.000Z``
    - Day-of-year: ``2025-118T12:00:00.000`` (used by SMAP SSS products)

    The day-of-year format has a hyphen between the 4-digit year and a
    3-digit ordinal day, which ``pd.Timestamp`` does not recognise. We
    detect it with a regex and convert via ``datetime.strptime``.
    """
    import re as _re
    from datetime import datetime

    s = str(raw).strip().rstrip("Z")
    # Day-of-year: "2025-118T12:00:00.000"
    m = _re.match(r"^(\d{4})-(\d{3})T(.+)$", s)
    if m:
        year, doy, time_part = m.group(1), m.group(2), m.group(3)
        # Truncate fractional seconds to 6 digits (microseconds) to
        # avoid strptime errors with >6 decimal places.
        if "." in time_part:
            base, frac = time_part.split(".", 1)
            time_part = f"{base}.{frac[:6]}"
        dt = datetime.strptime(f"{year}-{doy}T{time_part}", "%Y-%jT%H:%M:%S.%f")
        return pd.Timestamp(dt, tz="UTC")

    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts


def _parse_file_center_time(path: Path) -> Optional[pd.Timestamp]:
    """
    Parse the temporal midpoint of a gridded ancillary file from its
    ``time_coverage_start`` / ``time_coverage_end`` global attributes.

    Returns None if the attributes are missing or unparseable.
    """
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            t0_raw = ds.attrs.get("time_coverage_start")
            t1_raw = ds.attrs.get("time_coverage_end")
        if t0_raw is None or t1_raw is None:
            return None
        t0 = _parse_time_attr(t0_raw)
        t1 = _parse_time_attr(t1_raw)
        return t0 + (t1 - t0) / 2
    except Exception:
        return None


def _pick_nearest_file(
    file_times: list[tuple[Path, pd.Timestamp]],
    target: pd.Timestamp,
) -> Path:
    """
    From a list of (path, center_time) pairs, return the path whose
    center_time is closest to *target*.
    """
    if not file_times:
        raise FileNotFoundError("No ancillary files with parseable time coverage")
    best_path, best_dt = file_times[0][0], abs((file_times[0][1] - target).total_seconds())
    for p, t in file_times[1:]:
        dt = abs((t - target).total_seconds())
        if dt < best_dt:
            best_path, best_dt = p, dt
    return best_path


def _sample_gridded_field(
    da: xr.DataArray,
    *,
    lat: float,
    lon: float,
    time: Optional[pd.Timestamp],
) -> float:
    """
    Sample a gridded DataArray at a single (lat, lon) point using
    nearest-neighbor interpolation.  If the DataArray has a ``time``
    dimension and *time* is provided, the nearest timestep is selected
    first.
    """
    lat_name, lon_name = _infer_lat_lon_names(da)
    sel = da
    if time is not None and "time" in sel.dims:
        sel = sel.sel(time=np.datetime64(time.to_datetime64()), method="nearest")
    sampled = sel.interp({lat_name: lat, lon_name: lon}, method="nearest")
    return float(sampled.values)


def _run_l2_validation_pipeline(
    config: dict,
    *,
    config_file: Path,
    pigments: Optional[list[str]],
) -> xr.Dataset:
    validation = config.get("validation") or {}
    if not validation:
        raise ValueError(
            "PACE L2 inputs were detected, but your config does not include a 'validation' section. "
            "For L2 workflows, configure 'validation.hplc', 'validation.matching', 'validation.qc', "
            "and 'validation.spectral'."
        )

    input_dir = get_input_dir(config, project_root, config_dir=config_file.parent)
    rrs_dir = input_dir / "rrs"
    l2_granules = [p for p in list_nc_paths(rrs_dir) if _is_pace_l2_rrs_file(p)]
    if not l2_granules:
        raise FileNotFoundError(f"No PACE L2 AOP granules found under {rrs_dir}")

    # ---- Load in-situ observations
    hplc_cfg = validation.get("hplc") or {}
    hplc_path_raw = hplc_cfg.get("path")
    if not hplc_path_raw:
        raise ValueError("validation.hplc.path is required for the L2 validation pipeline")

    hplc_path = resolve_path(str(hplc_path_raw), project_root, config_dir=config_file.parent)

    columns_cfg = hplc_cfg.get("columns") or {}
    datetime_cfg = hplc_cfg.get("datetime") or {}

    obs_df = load_hplc_observations(
        hplc_path,
        lat_key=str(columns_cfg.get("lat", "lat")),
        lon_key=str(columns_cfg.get("lon", "lon")),
        date_key=str(columns_cfg.get("date", "date")),
        time_key=str(columns_cfg.get("time", "time")),
        station_key=str(columns_cfg.get("station", "station")),
        date_format=str(datetime_cfg.get("date_format", "%Y%m%d")),
        time_format=str(datetime_cfg.get("time_format", "%H:%M:%S")),
        timezone=str(datetime_cfg.get("timezone", "UTC")),
    )

    if obs_df.empty:
        raise ValueError(f"No valid observations found in {hplc_path}")

    # ---- Matching + QC + spectral parameters
    matching = validation.get("matching") or {}
    qc = validation.get("qc") or {}
    spectral = validation.get("spectral") or {}

    time_window_hours = float(matching.get("time_window_hours", 3))
    max_distance_km = float(matching.get("max_distance_km", 5))
    max_granules_per_obs = int(matching.get("max_granules_per_obs", 3))

    if "neighborhood" in matching:
        raise ValueError(
            "validation.matching.neighborhood is no longer supported. "
            "This pipeline uses the single nearest pixel only (no neighborhood aggregation)."
        )

    l2_flags_cfg = (qc.get("l2_flags") or {})
    exclude_bits = list(l2_flags_cfg.get("exclude_bits", [0, 1, 3, 4, 8, 9, 14, 16, 25, 26]))

    pix_validity = (qc.get("pixel_spectral_validity") or {})
    min_finite_fraction = float(pix_validity.get("min_finite_fraction", 0.95))

    interp_nm = int(spectral.get("interp_nm", 1))
    smooth_nm = int(spectral.get("smooth_nm", 5))
    edge_trim_nm = int(spectral.get("edge_trim_nm", 4))
    final_range_nm = list(spectral.get("final_range_nm", [400, 700]))
    if interp_nm != 1:
        raise ValueError(
            f"validation.spectral.interp_nm must be 1 for SDP inputs (got {interp_nm}). "
            "The model coefficients are defined on a 1 nm wavelength grid."
        )
    if smooth_nm != 5:
        raise ValueError(
            f"validation.spectral.smooth_nm must be 5 for SDP inputs (got {smooth_nm}). "
            "Second-derivative features are extremely noise-sensitive; the model expects 5 nm smoothing."
        )

    # ---- Build matchups: one per in-situ obs (best granule that passes QC)
    matchups: list[L2Matchup] = []
    for obs_idx, row in enumerate(obs_df.itertuples(index=False)):
        obs_time: pd.Timestamp = row.obs_time
        obs_lat = float(row.obs_lat)
        obs_lon = float(row.obs_lon)
        station = None if pd.isna(row.station) else str(row.station)

        candidates = select_candidate_granules(
            l2_granules,
            obs_time,
            time_window_hours=time_window_hours,
            max_granules=max_granules_per_obs,
        )
        if not candidates:
            continue

        best: Optional[L2Matchup] = None
        for granule in candidates:
            matchup = extract_l2_matchup(
                granule,
                obs_index=obs_idx,
                obs_time=obs_time,
                obs_lat=obs_lat,
                obs_lon=obs_lon,
                station=station,
                exclude_bits=exclude_bits,
                min_finite_fraction=min_finite_fraction,
                interp_nm=interp_nm,
                smooth_nm=smooth_nm,
                edge_trim_nm=edge_trim_nm,
                final_range_nm=final_range_nm,
            )
            if matchup is None:
                continue
            if matchup.distance_km > max_distance_km:
                continue
            best = matchup
            break

        if best is not None:
            matchups.append(best)

    if not matchups:
        raise RuntimeError("No valid matchups produced (all candidates failed QC / distance / time gates).")

    # ---- Sample SST/SSS at matchup points (required by the model)
    #
    # These ancillary files are typically 8-day composites with NO time
    # dimension — just (lat, lon).  We can't merge them with
    # open_mfdataset(combine="by_coords") because there's nothing to
    # concatenate along.  Instead, for each matchup we pick the file
    # whose time_coverage midpoint is closest to the observation time.
    sss_paths = list_nc_paths(input_dir / "sss")
    sst_paths = list_nc_paths(input_dir / "sst")
    if not sss_paths:
        raise FileNotFoundError(f"Missing SSS inputs under {input_dir / 'sss'} (need one or more .nc/.nc4 files)")
    if not sst_paths:
        raise FileNotFoundError(f"Missing SST inputs under {input_dir / 'sst'} (need one or more .nc/.nc4 files)")

    sss_file_times = [(p, t) for p in sss_paths if (t := _parse_file_center_time(p)) is not None]
    sst_file_times = [(p, t) for p in sst_paths if (t := _parse_file_center_time(p)) is not None]
    if not sss_file_times:
        raise ValueError(
            f"None of the SSS files under {input_dir / 'sss'} have parseable "
            "time_coverage_start/end attributes"
        )
    if not sst_file_times:
        raise ValueError(
            f"None of the SST files under {input_dir / 'sst'} have parseable "
            "time_coverage_start/end attributes"
        )

    sss_vals: list[float] = []
    sst_vals: list[float] = []
    keep: list[L2Matchup] = []
    for m in matchups:
        sss_file = _pick_nearest_file(sss_file_times, m.obs_time)
        sst_file = _pick_nearest_file(sst_file_times, m.obs_time)

        with xr.open_dataset(sss_file) as sss_ds, xr.open_dataset(sst_file) as sst_ds:
            if "smap_sss" not in sss_ds:
                raise KeyError(f"SSS file missing variable 'smap_sss'. Found: {list(sss_ds.data_vars)}")
            if "sst" not in sst_ds:
                raise KeyError(f"SST file missing variable 'sst'. Found: {list(sst_ds.data_vars)}")

            sss_val = _sample_gridded_field(sss_ds["smap_sss"], lat=m.pixel_lat, lon=m.pixel_lon, time=m.obs_time)
            sst_val = _sample_gridded_field(sst_ds["sst"], lat=m.pixel_lat, lon=m.pixel_lon, time=m.obs_time)

        if not (np.isfinite(sss_val) and np.isfinite(sst_val)):
            continue
        keep.append(m)
        sss_vals.append(sss_val)
        sst_vals.append(sst_val)

    if not keep:
        raise RuntimeError("All matchups were dropped due to missing SST/SSS values at matchup points.")

    matchups = keep
    sss_np = np.asarray(sss_vals, dtype=float)
    sst_np = np.asarray(sst_vals, dtype=float)

    # ---- Run SDP model
    wavelengths = np.arange(400, 701, 1)
    rrs = np.stack([m.rrs_400_700_1nm for m in matchups], axis=0)
    rrs_df = pd.DataFrame(rrs, columns=wavelengths)

    from models.sdp_pigments.core.prediction import run_sdp

    sdp = run_sdp(rrs_df, wavelengths, sst_np, sss_np, pigments=pigments)

    # ---- Build output Dataset
    data_vars = {}
    for col in sdp.columns:
        var_name = col.replace(" ", "_").replace("+", "_").replace("-", "_").lower()
        data_vars[var_name] = (["matchup"], sdp[col].to_numpy(dtype=float))

    ds = xr.Dataset(
        data_vars,
        coords={
            "matchup": np.arange(len(matchups), dtype=int),
            "wavelength": wavelengths.astype(int),
        },
    )

    ds["rrs"] = (["matchup", "wavelength"], rrs.astype(float))
    ds["obs_index"] = (["matchup"], np.asarray([m.obs_index for m in matchups], dtype=int))
    ds["obs_time"] = (["matchup"], np.asarray([np.datetime64(m.obs_time.to_datetime64()) for m in matchups]))
    ds["obs_lat"] = (["matchup"], np.asarray([m.obs_lat for m in matchups], dtype=float))
    ds["obs_lon"] = (["matchup"], np.asarray([m.obs_lon for m in matchups], dtype=float))
    ds["pixel_lat"] = (["matchup"], np.asarray([m.pixel_lat for m in matchups], dtype=float))
    ds["pixel_lon"] = (["matchup"], np.asarray([m.pixel_lon for m in matchups], dtype=float))
    ds["distance_km"] = (["matchup"], np.asarray([m.distance_km for m in matchups], dtype=float))
    ds["sss"] = (["matchup"], sss_np)
    ds["sst"] = (["matchup"], sst_np)

    ds.attrs["experiment"] = get_experiment_name(config)
    ds.attrs["config_file"] = str(config_file)
    return ds


if __name__ == "__main__":
    # Prompt for config file path
    config_input = input("Enter path to experiment config file (e.g., experiments/<name>/config.py): ").strip()
    if not config_input:
        raise ValueError("Config file path cannot be empty")
    
    config_path = Path(config_input)
    config_file = config_path if config_path.is_absolute() else (project_root / config_path)
    config_file = config_file.resolve()
    config = load_config_from_file(config_file)
    
    # Extract experiment name from config
    experiment = get_experiment_name(config)
    bbox = get_bbox(config)
    pigments = get_pigments(config)
    output_filename = get_output_filename(config)
    
    print(f"Experiment: {experiment}")
    description = get_experiment_description(config)
    if description:
        print(f"Description: {description}")
    
    # Data directory paths
    input_dir = get_input_dir(config, project_root, config_dir=config_file.parent)
    
    # Collect file paths
    rrs_paths = list_nc_paths(input_dir / "rrs")
    sss_paths = list_nc_paths(input_dir / "sss")
    sst_paths = list_nc_paths(input_dir / "sst")
    
    print(f"Found {len(rrs_paths)} Rrs, {len(sss_paths)} SSS, {len(sst_paths)} SST files")
    print(f"Bbox: {bbox}")
    print(f"Pigments: {'all' if pigments is None else ', '.join(pigments)}")

    is_l2 = any(_is_pace_l2_rrs_file(p) for p in rrs_paths)
    if not is_l2:
        raise ValueError(
            "This repo no longer supports mapped/gridded (L3m-style) Rrs inputs via `main.py`. "
            "Please provide PACE OCI L2 AOP granules under `io.input_dir/rrs/`."
        )

    result = _run_l2_validation_pipeline(config, config_file=config_file, pigments=pigments)
    
    # Save results
    output_dir = get_output_dir(config, project_root, config_dir=config_file.parent)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / output_filename
    result.to_netcdf(output_file)
    print(f"Results saved to {output_file}")
