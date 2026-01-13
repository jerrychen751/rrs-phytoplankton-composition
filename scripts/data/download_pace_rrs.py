#!/usr/bin/env python3
"""
Download and extract PACE OCI Level-2 Rrs matchups for in-situ HPLC validation.

The intended workflow is:

1) Provide an experiment YAML (see `config/template.yaml`).
2) Point `validation.hplc.path` to an in-situ SeaBASS-derived dataset (CSV or .sb).
3) For each in-situ observation:
   - Find candidate PACE OCI L2 AOP granules within ±N hours
   - Download only those candidate granules
   - Find the nearest pixel, verify temporal proximity using scanline time
   - Extract a 3×3 neighborhood, apply L2 flag screening and spectral QC
   - Compute a median neighborhood spectrum
   - Apply paper-aligned preprocessing:
       * interpolate to 1 nm grid
       * 5 nm moving-mean smoothing
       * drop first/last 4 nm
       * restrict to 400–700 nm
4) Write a compact structured NetCDF matchup dataset under ~/Downloads (configurable).

Notes on L2 data layout:
- PACE OCI L2 files are netCDF4 with groups like:
    * /geophysical_data     (contains Rrs, l2_flags)
    * /navigation_data      (contains latitude, longitude)
    * /scan_line_attributes (contains time per scan line)
    * /sensor_band_parameters (contains wavelength vectors)

Authentication:
- This script expects an Earthdata token (bearer token) in either:
    * EARTHDATA_TOKEN  (recommended)
    * EARTHACCESS_TOKEN (repo legacy)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import xarray as xr

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import get_bbox, get_time_range, load_config_from_file
from utils.seabass_loader import load_hplc_data as load_seabass_hplc


CMR_GRANULE_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
CMR_COLLECTION_SEARCH_URL = "https://cmr.earthdata.nasa.gov/search/collections.json"


def _get_token() -> Optional[str]:
    # Prefer EARTHDATA_TOKEN (matches earthaccess convention), but support the
    # repo legacy env var name as well.
    return os.getenv("EARTHDATA_TOKEN") or os.getenv("EARTHACCESS_TOKEN")


def _build_session() -> requests.Session:
    session = requests.Session()
    token = _get_token()
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})
    return session


def _expand_user(path: str) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(path)))


def _resolve_from_project_root(path: str) -> Path:
    p = _expand_user(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _as_utc_naive(dt: pd.Series) -> np.ndarray:
    """Convert pandas datetime series to timezone-naive UTC datetime64."""
    if not pd.api.types.is_datetime64_any_dtype(dt):
        raise TypeError("Expected a datetime-like pandas Series")
    if getattr(dt.dt, "tz", None) is None:
        # Treat naive timestamps as UTC.
        return dt.to_numpy(dtype="datetime64[ns]")
    return dt.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(dtype="datetime64[ns]")


def _format_cmr_time(dt: pd.Timestamp) -> str:
    # CMR expects ISO8601 with Z.
    if dt.tzinfo is None:
        dt = dt.tz_localize("UTC")
    return dt.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _deg_padding_from_km(km: float, multiplier: float = 1.5) -> float:
    # Rough conversion for small distances (1 degree lat ~ 111 km).
    return float(km) / 111.0 * multiplier


def _wrap_lon_diff(lon: np.ndarray, lon0: float) -> np.ndarray:
    """Return wrapped lon difference in degrees in [-180, 180]."""
    return ((lon - lon0 + 180.0) % 360.0) - 180.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    # Great-circle distance with mean Earth radius.
    r = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * r * math.asin(math.sqrt(a))


def _moving_mean(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    kernel = np.ones(int(window), dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def _interp_with_linear_extrap(x_new: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    1-D interpolation with linear extrapolation at both ends.

    This avoids rejecting matchups when only a small buffer outside the final
    wavelength range is missing (e.g. a few nm), which is common in practice for
    L2 swath products depending on band availability and flagging.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_new = np.asarray(x_new, dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    # np.interp requires monotonically increasing x; handle any duplicates by
    # averaging their y values.
    x_unique, inv = np.unique(x, return_inverse=True)
    if x_unique.size != x.size:
        sums = np.bincount(inv, weights=y)
        counts = np.bincount(inv)
        y = sums / counts
        x = x_unique

    if x.size < 2:
        raise ValueError("Need at least 2 points for interpolation/extrapolation")

    y_new = np.interp(x_new, x, y)

    left = x_new < x[0]
    if np.any(left):
        denom = x[1] - x[0]
        slope = 0.0 if denom == 0 else (y[1] - y[0]) / denom
        y_new[left] = y[0] + float(slope) * (x_new[left] - x[0])

    right = x_new > x[-1]
    if np.any(right):
        denom = x[-1] - x[-2]
        slope = 0.0 if denom == 0 else (y[-1] - y[-2]) / denom
        y_new[right] = y[-1] + float(slope) * (x_new[right] - x[-1])

    return y_new


def _sanitize_var_name(name: str) -> str:
    out = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip())
    out = re.sub(r"_+", "_", out).strip("_")
    if not out:
        return "unnamed"
    if out[0].isdigit():
        out = f"v_{out}"
    return out.lower()


def _get_nested(cfg: Mapping[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _search_collection_concept_id(session: requests.Session, short_name: str) -> Optional[str]:
    params = {"short_name": short_name, "page_size": 1}
    try:
        resp = session.get(CMR_COLLECTION_SEARCH_URL, params=params, timeout=30)
        resp.raise_for_status()
        feed = resp.json().get("feed", {})
        entries = feed.get("entry", [])
        if not entries:
            return None
        return entries[0].get("id")
    except Exception as exc:
        print(f"Error searching collections for '{short_name}': {exc}")
        return None


def _search_granules(
    session: requests.Session,
    collection_concept_id: str,
    temporal: Tuple[str, str],
    bounding_box: Tuple[float, float, float, float],
    page_size: int = 2000,
) -> List[dict]:
    bbox_str = ",".join(str(float(x)) for x in bounding_box)
    temporal_str = f"{temporal[0]},{temporal[1]}"

    params = {
        "collection_concept_id": collection_concept_id,
        "temporal": temporal_str,
        "bounding_box": bbox_str,
        "page_size": page_size,
    }

    entries: List[dict] = []
    headers = {}
    # If session has auth headers already, they will be used automatically. We
    # still need a local header map for pagination token updates.
    token = _get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    while True:
        resp = session.get(CMR_GRANULE_SEARCH_URL, params=params, headers=headers, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("feed", {}).get("entry", [])
        entries.extend(batch)

        search_after = resp.headers.get("CMR-Search-After")
        if search_after and batch:
            headers["CMR-Search-After"] = search_after
            continue
        break

    return entries


def _get_download_urls(granule: dict) -> List[str]:
    # Reuse the existing link-selection approach from other scripts in this repo:
    # prefer data download links; avoid OPeNDAP; keep .nc/.nc4.
    urls: List[str] = []
    for link in granule.get("links", []) or []:
        rel = (link.get("rel") or "").lower()
        href = link.get("href") or ""
        title = (link.get("title") or "").lower()
        if not href:
            continue
        if "/opendap/" in href.lower():
            continue
        if not (href.endswith(".nc") or href.endswith(".nc4")):
            continue
        is_data = "data#" in rel or "edsc" in rel or rel.endswith("/data")
        if is_data and ("download" in title or not title):
            urls.append(href)

    if urls:
        return sorted(set(urls))

    # Fallback: any .nc data-ish link.
    for link in granule.get("links", []) or []:
        rel = (link.get("rel") or "").lower()
        href = link.get("href") or ""
        if not href:
            continue
        if "/opendap/" in href.lower():
            continue
        if not (href.endswith(".nc") or href.endswith(".nc4")):
            continue
        if "data" in rel:
            urls.append(href)

    return sorted(set(urls))


def _download_file(session: requests.Session, url: str, output_path: Path, chunk_size: int = 1024 * 1024) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if output_path.exists():
            # If sizes match, skip.
            try:
                head = session.head(url, timeout=30, allow_redirects=True)
                if head.status_code == 200:
                    remote = int(head.headers.get("Content-Length", "0") or "0")
                    local = output_path.stat().st_size
                    if remote > 0 and local == remote:
                        return True
            except Exception:
                pass

        with session.get(url, stream=True, timeout=300) as resp:
            resp.raise_for_status()
            tmp = output_path.with_suffix(output_path.suffix + ".part")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
            tmp.replace(output_path)
        return True
    except Exception as exc:
        print(f"  Error downloading {url}: {exc}")
        return False


def _open_group(path: Path, group: str) -> xr.Dataset:
    try:
        return xr.open_dataset(path, group=group, mask_and_scale=True, decode_cf=True, decode_timedelta=False)
    except ValueError as exc:
        hint = (
            "xarray could not determine a backend to read this netCDF4/HDF5 file. "
            "Install `netCDF4` (or `h5netcdf`) in your environment."
        )
        raise RuntimeError(f"Unable to open {path.name} group '{group}': {exc}\n{hint}") from exc


def _extract_wavelengths(geophys: xr.Dataset, path: Path) -> Tuple[str, np.ndarray]:
    if "Rrs" not in geophys:
        raise KeyError("geophysical_data is missing variable 'Rrs'")
    rrs = geophys["Rrs"]
    # Identify the spectral dimension name.
    candidate_dims = [d for d in rrs.dims if d not in {"number_of_lines", "pixels_per_line"}]
    if len(candidate_dims) != 1:
        raise ValueError(f"Unexpected Rrs dims {rrs.dims}; cannot determine spectral dimension")
    wl_dim = candidate_dims[0]

    # Try direct coordinate/variable lookup first.
    for container in (geophys.coords, geophys.variables):
        if wl_dim in container:
            wl = np.asarray(container[wl_dim].values, dtype=float)
            return wl_dim, wl

    # Fall back to sensor_band_parameters group.
    band = _open_group(path, "sensor_band_parameters")
    try:
        for cand in (wl_dim, "wavelength_3d", "wavelength"):
            if cand in band:
                wl = np.asarray(band[cand].values, dtype=float)
                return wl_dim, wl
    finally:
        band.close()

    raise KeyError(f"Unable to locate wavelength coordinate for dim '{wl_dim}'")


@dataclass(frozen=True)
class MatchConfig:
    time_window_seconds: int
    max_distance_km: float
    neighborhood_size: int
    min_valid_pixels: int
    search_padding_deg: float
    max_granules_per_obs: int
    exclude_mask_u32: np.uint32
    pixel_min_finite_fraction: float
    max_cv: float
    cv_wl_lo: float
    cv_wl_hi: float
    interp_nm: int
    smooth_nm: int
    edge_trim_nm: int
    final_wl_lo: int
    final_wl_hi: int


def _build_match_config(cfg: Mapping[str, Any]) -> MatchConfig:
    time_window_hours = float(_get_nested(cfg, ["validation", "matching", "time_window_hours"], 3))
    time_window_seconds = int(round(time_window_hours * 3600))

    max_distance_km = float(_get_nested(cfg, ["validation", "matching", "max_distance_km"], 5))

    box_size = int(_get_nested(cfg, ["validation", "matching", "neighborhood", "box_size"], 3))
    min_valid_pixels = int(_get_nested(cfg, ["validation", "matching", "neighborhood", "min_valid_pixels"], 6))

    search_padding_deg = _get_nested(cfg, ["validation", "matching", "search_padding_deg"], None)
    if search_padding_deg is None:
        search_padding_deg = _deg_padding_from_km(max_distance_km)
    search_padding_deg = float(search_padding_deg)

    max_granules_per_obs = int(_get_nested(cfg, ["validation", "matching", "max_granules_per_obs"], 3))

    exclude_bits = _get_nested(cfg, ["validation", "qc", "l2_flags", "exclude_bits"], [0, 1, 3, 4, 8, 9, 14, 16, 25, 26])
    exclude_mask = 0
    for bit in exclude_bits:
        exclude_mask |= 1 << int(bit)
    exclude_mask_u32 = np.uint32(exclude_mask)

    pixel_min_finite_fraction = float(
        _get_nested(cfg, ["validation", "qc", "pixel_spectral_validity", "min_finite_fraction"], 0.95)
    )

    max_cv = float(_get_nested(cfg, ["validation", "qc", "spectral_homogeneity", "max_cv"], 0.15))
    cv_range = _get_nested(cfg, ["validation", "qc", "spectral_homogeneity", "cv_wavelength_range_nm"], [405, 570])
    cv_wl_lo = float(cv_range[0])
    cv_wl_hi = float(cv_range[1])

    interp_nm = int(_get_nested(cfg, ["validation", "spectral", "interp_nm"], 1))
    smooth_nm = int(_get_nested(cfg, ["validation", "spectral", "smooth_nm"], 5))
    edge_trim_nm = int(_get_nested(cfg, ["validation", "spectral", "edge_trim_nm"], 4))
    final_range = _get_nested(cfg, ["validation", "spectral", "final_range_nm"], [400, 700])
    final_wl_lo = int(final_range[0])
    final_wl_hi = int(final_range[1])

    if box_size % 2 != 1:
        raise ValueError(f"Neighborhood box_size must be odd; got {box_size}")
    if box_size != 3:
        raise ValueError(f"This script currently supports only 3x3 neighborhoods; got box_size={box_size}")
    if min_valid_pixels < 1 or min_valid_pixels > box_size * box_size:
        raise ValueError(f"min_valid_pixels must be in [1, {box_size*box_size}]; got {min_valid_pixels}")
    if final_wl_hi <= final_wl_lo:
        raise ValueError(f"Invalid final_range_nm: {final_range}")

    return MatchConfig(
        time_window_seconds=time_window_seconds,
        max_distance_km=max_distance_km,
        neighborhood_size=box_size,
        min_valid_pixels=min_valid_pixels,
        search_padding_deg=search_padding_deg,
        max_granules_per_obs=max_granules_per_obs,
        exclude_mask_u32=exclude_mask_u32,
        pixel_min_finite_fraction=pixel_min_finite_fraction,
        max_cv=max_cv,
        cv_wl_lo=cv_wl_lo,
        cv_wl_hi=cv_wl_hi,
        interp_nm=interp_nm,
        smooth_nm=smooth_nm,
        edge_trim_nm=edge_trim_nm,
        final_wl_lo=final_wl_lo,
        final_wl_hi=final_wl_hi,
    )


def _load_hplc_targets(cfg: Mapping[str, Any]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    path_str = _get_nested(cfg, ["validation", "hplc", "path"], None)
    if not path_str:
        raise ValueError("Missing config key: validation.hplc.path")
    path = _resolve_from_project_root(str(path_str))
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".sb":
        df = load_seabass_hplc(path)
        # Best-effort normalization.
        if "datetime" not in df.columns:
            if "date" in df.columns and "time" in df.columns:
                df["datetime"] = pd.to_datetime(
                    df["date"].astype(str) + " " + df["time"].astype(str),
                    errors="coerce",
                    utc=True,
                )
        columns = {"lat": "lat", "lon": "lon", "date": "date", "time": "time", "station": "station"}
        return df, columns

    df = pd.read_csv(path)

    columns_cfg = _get_nested(cfg, ["validation", "hplc", "columns"], {})
    lat_col = str(columns_cfg.get("lat", "lat"))
    lon_col = str(columns_cfg.get("lon", "lon"))
    date_col = str(columns_cfg.get("date", "date"))
    time_col = str(columns_cfg.get("time", "time"))
    station_col = str(columns_cfg.get("station", "station"))

    for required in (lat_col, lon_col, date_col, time_col):
        if required not in df.columns:
            raise ValueError(f"HPLC CSV is missing required column '{required}'. Columns: {list(df.columns)}")

    dt_cfg = _get_nested(cfg, ["validation", "hplc", "datetime"], {})
    date_fmt = dt_cfg.get("date_format", "%Y%m%d")
    time_fmt = dt_cfg.get("time_format", "%H:%M:%S")

    dt = pd.to_datetime(
        df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip(),
        format=f"{date_fmt} {time_fmt}",
        errors="coerce",
        utc=True,
    )
    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(f"Failed to parse {bad} datetime values from HPLC CSV (check date/time formats).")

    df = df.copy()
    df["datetime"] = dt
    # Normalize primary matching columns.
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    if df[lat_col].isna().any() or df[lon_col].isna().any():
        raise ValueError("HPLC CSV contains NaNs in lat/lon after coercion.")

    columns = {"lat": lat_col, "lon": lon_col, "date": date_col, "time": time_col, "station": station_col}
    return df, columns


def _filter_targets_to_experiment(df: pd.DataFrame, columns: Mapping[str, str], cfg: Mapping[str, Any]) -> pd.DataFrame:
    west, south, east, north = get_bbox(cfg)
    lat_col = columns["lat"]
    lon_col = columns["lon"]

    df = df.copy()
    df = df[(df[lat_col] >= south) & (df[lat_col] <= north)]

    # Handle lon wrap naively for [-180, 180] ranges; for this repo's cases this is fine.
    df = df[(df[lon_col] >= west) & (df[lon_col] <= east)]

    tr = get_time_range(cfg)
    if tr is not None:
        start = pd.to_datetime(str(tr[0]), utc=True)
        end = pd.to_datetime(str(tr[1]), utc=True) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        df = df[(df["datetime"] >= start) & (df["datetime"] <= end)]

    return df


def _build_station_bbox(
    lat: float,
    lon: float,
    exp_bbox: Tuple[float, float, float, float],
    padding_deg: float,
) -> Tuple[float, float, float, float]:
    west, south, east, north = exp_bbox
    lon_min = _clip(lon - padding_deg, west, east)
    lon_max = _clip(lon + padding_deg, west, east)
    lat_min = _clip(lat - padding_deg, south, north)
    lat_max = _clip(lat + padding_deg, south, north)
    return (lon_min, lat_min, lon_max, lat_max)


def _granule_time_start(granule: Mapping[str, Any]) -> Optional[pd.Timestamp]:
    ts = granule.get("time_start")
    if not ts:
        return None
    try:
        return pd.to_datetime(ts, utc=True)
    except Exception:
        return None


def _choose_candidate_granules(
    session: requests.Session,
    cfg: Mapping[str, Any],
    match_cfg: MatchConfig,
    targets: pd.DataFrame,
    columns: Mapping[str, str],
) -> Tuple[Dict[int, List[dict]], Dict[str, dict]]:
    exp_bbox = get_bbox(cfg)
    lat_col = columns["lat"]
    lon_col = columns["lon"]

    short_names: List[str] = _get_nested(cfg, ["pace", "l2_aop_short_names"], ["PACE_OCI_L2_AOP", "PACE_OCI_L2_AOP_NRT"])

    # Resolve concept IDs once per short_name.
    concept_ids: Dict[str, str] = {}
    for short_name in short_names:
        cid = _search_collection_concept_id(session, short_name)
        if cid:
            concept_ids[short_name] = cid
        else:
            print(f"Warning: could not resolve CMR concept_id for short_name '{short_name}'")

    if not concept_ids:
        raise RuntimeError(
            "Unable to locate any PACE OCI L2 AOP collections in CMR.\n"
            f"Tried short_names: {short_names}\n"
            "Check that your network/auth is working and that short_names are correct."
        )

    per_obs: Dict[int, List[dict]] = {}
    selected_unique: Dict[str, dict] = {}

    for obs_id, row in targets.iterrows():
        obs_time = pd.Timestamp(row["datetime"]).tz_convert("UTC")
        t0 = obs_time - pd.Timedelta(seconds=match_cfg.time_window_seconds)
        t1 = obs_time + pd.Timedelta(seconds=match_cfg.time_window_seconds)

        station_bbox = _build_station_bbox(
            float(row[lat_col]),
            float(row[lon_col]),
            exp_bbox=exp_bbox,
            padding_deg=match_cfg.search_padding_deg,
        )

        all_candidates: List[dict] = []
        for short_name, concept_id in concept_ids.items():
            try:
                granules = _search_granules(
                    session,
                    collection_concept_id=concept_id,
                    temporal=(_format_cmr_time(t0), _format_cmr_time(t1)),
                    bounding_box=station_bbox,
                )
            except Exception as exc:
                print(f"Warning: CMR granule search failed for {short_name}: {exc}")
                continue
            for g in granules:
                g = dict(g)
                g["_collection_short_name"] = short_name
                all_candidates.append(g)

        # Deduplicate candidates by CMR granule id.
        by_id: Dict[str, dict] = {}
        for g in all_candidates:
            gid = g.get("id") or g.get("producer_granule_id") or g.get("title") or json.dumps(g, sort_keys=True)
            if gid not in by_id:
                by_id[gid] = g

        # Rank by absolute temporal proximity (using time_start metadata).
        ranked: List[Tuple[float, dict]] = []
        for g in by_id.values():
            t_start = _granule_time_start(g)
            if t_start is None:
                continue
            dt = abs((t_start - obs_time).total_seconds())
            ranked.append((dt, g))
        ranked.sort(key=lambda x: x[0])

        chosen = [g for _, g in ranked[: match_cfg.max_granules_per_obs]]
        per_obs[int(obs_id)] = chosen

        for g in chosen:
            gid = g.get("id") or g.get("producer_granule_id") or g.get("title")
            if not gid:
                continue
            selected_unique[str(gid)] = g

    return per_obs, selected_unique


def _scanline_time(scan: xr.Dataset, line_index: int) -> Optional[pd.Timestamp]:
    if "time" in scan:
        t = scan["time"].values
        if line_index < 0 or line_index >= t.shape[0]:
            return None
        sec = float(t[line_index])
        if not np.isfinite(sec) or sec < 0:
            return None
        return pd.to_datetime(sec, unit="s", origin="unix", utc=True)

    # Fallback: compute from year/day/msec if present.
    for key in ("year", "day", "msec"):
        if key not in scan:
            return None

    def as_int(value: np.ndarray, unit: str) -> int:
        scalar = value[line_index]
        if np.issubdtype(np.asarray(scalar).dtype, np.timedelta64):
            denom = np.timedelta64(1, unit)
            return int(np.asarray(scalar) / denom)
        return int(scalar)

    year_raw = np.asarray(scan["year"].values)
    day_raw = np.asarray(scan["day"].values)
    msec_raw = np.asarray(scan["msec"].values)

    year = int(float(year_raw[line_index]))
    day_of_year = as_int(day_raw, unit="D")
    msec_of_day = as_int(msec_raw, unit="ms")

    if year <= 0 or day_of_year <= 0 or day_of_year > 366:
        return None
    if msec_of_day < 0 or msec_of_day >= 24 * 60 * 60 * 1000:
        return None

    base = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
    return base + pd.Timedelta(days=day_of_year - 1, milliseconds=msec_of_day)


def _extract_matchup_from_file(
    file_path: Path,
    obs_lat: float,
    obs_lon: float,
    obs_time: pd.Timestamp,
    match_cfg: MatchConfig,
) -> Dict[str, Any]:
    """
    Attempt to extract a validated matchup spectrum from one granule.

    Returns a dict with:
    - ok: bool
    - reason: str (if !ok)
    - rrs: np.ndarray[float] (400-700, 1nm) if ok
    - plus metadata fields (pixel indices, dt, etc.)
    """

    nav = _open_group(file_path, "navigation_data")
    scan = _open_group(file_path, "scan_line_attributes")
    geo = _open_group(file_path, "geophysical_data")

    try:
        if "latitude" not in nav or "longitude" not in nav:
            return {"ok": False, "reason": "missing_navigation_latlon"}
        lat = np.asarray(nav["latitude"].values, dtype=float)
        lon = np.asarray(nav["longitude"].values, dtype=float)
        if lat.ndim != 2 or lon.ndim != 2:
            return {"ok": False, "reason": "unexpected_navigation_shape"}

        # Nearest pixel using squared distance in degrees (fast), then compute km distance.
        lon_diff = _wrap_lon_diff(lon, obs_lon)
        d2 = (lat - obs_lat) ** 2 + lon_diff**2
        try:
            flat = int(np.nanargmin(d2))
        except ValueError:
            return {"ok": False, "reason": "no_finite_navigation_points"}
        i, j = np.unravel_index(flat, lat.shape)

        nearest_lat = float(lat[i, j])
        nearest_lon = float(lon[i, j])
        distance_km = _haversine_km(obs_lat, obs_lon, nearest_lat, nearest_lon)
        if distance_km > match_cfg.max_distance_km:
            return {"ok": False, "reason": "nearest_pixel_too_far", "distance_km": distance_km}

        pix_time = _scanline_time(scan, int(i))
        if pix_time is None:
            return {"ok": False, "reason": "missing_scanline_time"}

        dt_seconds = float((pix_time - obs_time).total_seconds())
        if abs(dt_seconds) > match_cfg.time_window_seconds:
            return {"ok": False, "reason": "time_window_exceeded", "dt_seconds": dt_seconds}

        if "Rrs" not in geo or "l2_flags" not in geo:
            return {"ok": False, "reason": "missing_rrs_or_l2_flags"}

        line_dim, pix_dim = nav["latitude"].dims

        wl_dim, wl = _extract_wavelengths(geo, file_path)
        if wl.ndim != 1:
            return {"ok": False, "reason": "unexpected_wavelength_shape"}

        # Define the working wavelength range early so QC can be evaluated only
        # on the spectral region that will actually be used for the matchup
        # spectrum (and not rejected due to missing values outside that range).
        final_lo = match_cfg.final_wl_lo
        final_hi = match_cfg.final_wl_hi
        work_lo = final_lo - match_cfg.edge_trim_nm
        work_hi = final_hi + match_cfg.edge_trim_nm
        work_mask = (wl >= work_lo) & (wl <= work_hi)
        if not np.any(work_mask):
            return {"ok": False, "reason": "no_wavelengths_in_working_range"}
        work_count = int(np.count_nonzero(work_mask))

        half = match_cfg.neighborhood_size // 2
        i0 = max(int(i) - half, 0)
        i1 = min(int(i) + half + 1, lat.shape[0])
        j0 = max(int(j) - half, 0)
        j1 = min(int(j) + half + 1, lat.shape[1])

        rrs_patch = geo["Rrs"].isel({line_dim: slice(i0, i1), pix_dim: slice(j0, j1)})
        flags_patch = geo["l2_flags"].isel({line_dim: slice(i0, i1), pix_dim: slice(j0, j1)})

        rrs_vals = np.asarray(rrs_patch.values, dtype=float)
        flags_vals = np.asarray(flags_patch.values)
        if rrs_vals.ndim != 3 or flags_vals.ndim != 2:
            return {"ok": False, "reason": "unexpected_rrs_or_flags_shape"}

        n_lines, n_pix, n_wl = rrs_vals.shape
        rrs_flat = rrs_vals.reshape(n_lines * n_pix, n_wl)

        flags_u32 = np.asarray(flags_vals, dtype=np.uint32).reshape(n_lines * n_pix)
        flags_ok = (flags_u32 & match_cfg.exclude_mask_u32) == 0

        finite_frac = np.isfinite(rrs_flat[:, work_mask]).sum(axis=1) / float(work_count)
        spectral_ok = finite_frac >= match_cfg.pixel_min_finite_fraction

        pixel_ok = flags_ok & spectral_ok
        n_ok = int(pixel_ok.sum())
        if n_ok < match_cfg.min_valid_pixels:
            return {"ok": False, "reason": "too_few_valid_pixels", "valid_pixels": n_ok}

        # Spectral homogeneity check (SeaBASS-style): median of band-specific CVs.
        wl_mask = (wl >= match_cfg.cv_wl_lo) & (wl <= match_cfg.cv_wl_hi)
        if not np.any(wl_mask):
            return {"ok": False, "reason": "no_wavelengths_in_cv_range"}
        vals_cv = rrs_flat[pixel_ok][:, wl_mask]
        mean = np.nanmean(vals_cv, axis=0)
        std = np.nanstd(vals_cv, axis=0)
        eps = 1e-12
        good = np.isfinite(mean) & np.isfinite(std) & (np.abs(mean) > eps)
        if not np.any(good):
            return {"ok": False, "reason": "cv_invalid_mean"}
        cv = std[good] / mean[good]
        cv_median = float(np.nanmedian(cv))
        if not np.isfinite(cv_median):
            return {"ok": False, "reason": "cv_nan"}
        if cv_median > match_cfg.max_cv:
            return {"ok": False, "reason": "spectrally_heterogeneous", "cv_median": cv_median}

        # Neighborhood median spectrum across valid pixels.
        median_native = np.nanmedian(rrs_flat[pixel_ok], axis=0)

        # Paper-aligned preprocessing.
        wl_work = wl[work_mask]
        spec_work = median_native[work_mask]
        finite = np.isfinite(spec_work)
        if finite.sum() < 2:
            return {"ok": False, "reason": "median_spectrum_too_sparse"}
        wl_finite = wl_work[finite]
        spec_finite = spec_work[finite]

        # Require that the data span the final (kept) range, but allow small
        # gaps in the buffer region that will be trimmed away after smoothing.
        wl_min = float(wl_finite.min())
        wl_max = float(wl_finite.max())
        left_gap_nm = max(0.0, wl_min - float(work_lo))
        right_gap_nm = max(0.0, float(work_hi) - wl_max)
        if left_gap_nm > match_cfg.edge_trim_nm or right_gap_nm > match_cfg.edge_trim_nm:
            return {
                "ok": False,
                "reason": "median_spectrum_missing_edges",
                "left_gap_nm": left_gap_nm,
                "right_gap_nm": right_gap_nm,
                "wl_min": wl_min,
                "wl_max": wl_max,
                "work_lo": float(work_lo),
                "work_hi": float(work_hi),
            }

        grid = np.arange(work_lo, work_hi + 1, match_cfg.interp_nm, dtype=float)
        spec_interp = _interp_with_linear_extrap(grid, wl_finite, spec_finite)

        window = int(round(match_cfg.smooth_nm / match_cfg.interp_nm))
        spec_smooth = _moving_mean(spec_interp, window=window)

        trim = match_cfg.edge_trim_nm
        if trim <= 0 or trim * 2 >= spec_smooth.size:
            return {"ok": False, "reason": "invalid_edge_trim"}
        spec_trim = spec_smooth[trim:-trim]
        wl_trim = grid[trim:-trim]

        # Ensure final wavelength window is correct and stable.
        if int(round(wl_trim[0])) != final_lo or int(round(wl_trim[-1])) != final_hi:
            return {"ok": False, "reason": "unexpected_final_wavelength_range"}

        if not np.all(np.isfinite(spec_trim)):
            return {"ok": False, "reason": "final_spectrum_contains_nan"}

        return {
            "ok": True,
            "rrs": spec_trim.astype(np.float32),
            "wavelength": wl_trim.astype(np.int32),
            "pixel_i": int(i),
            "pixel_j": int(j),
            "distance_km": float(distance_km),
            "pixel_time": pix_time,
            "dt_seconds": float(dt_seconds),
            "valid_pixels": n_ok,
            "cv_median": cv_median,
            "patch_i0": int(i0),
            "patch_i1": int(i1),
            "patch_j0": int(j0),
            "patch_j1": int(j1),
        }
    finally:
        nav.close()
        scan.close()
        geo.close()


def _build_output_dir(cfg: Mapping[str, Any]) -> Path:
    downloads_dir = _get_nested(cfg, ["io", "downloads_dir"], "~/Downloads/rrs-SDP-pigments")
    downloads_base = _expand_user(str(downloads_dir))

    exp_name = _get_nested(cfg, ["experiment", "name"], None)
    if not exp_name:
        raise ValueError("Config missing experiment.name")

    output_subdir = _get_nested(cfg, ["io", "pace_l2_rrs", "output_subdir"], "pace_l2_rrs")
    return downloads_base / str(exp_name) / str(output_subdir)


def _write_matchups_dataset(
    output_path: Path,
    targets: pd.DataFrame,
    columns: Mapping[str, str],
    spectra_400_700: np.ndarray,
    wl_400_700: np.ndarray,
    result_meta: Dict[int, Dict[str, Any]],
) -> None:
    lat_col = columns["lat"]
    lon_col = columns["lon"]
    station_col = columns.get("station")

    obs_time = _as_utc_naive(targets["datetime"])
    lat = targets[lat_col].to_numpy(dtype=float)
    lon = targets[lon_col].to_numpy(dtype=float)

    ds = xr.Dataset(
        coords={
            "obs": np.arange(targets.shape[0], dtype=np.int32),
            "wavelength": wl_400_700.astype(np.int32),
        },
        data_vars={
            "Rrs": (("obs", "wavelength"), spectra_400_700.astype(np.float32)),
            "obs_time": ("obs", obs_time),
            "obs_lat": ("obs", lat.astype(np.float32)),
            "obs_lon": ("obs", lon.astype(np.float32)),
        },
        attrs={
            "title": "PACE OCI L2 Rrs matchups for in-situ validation",
            "rrs_processing": (
                "Median over valid pixels in a 3x3 neighborhood; "
                "interpolate to 1 nm; 5 nm moving mean; trim first/last 4 nm; "
                "restrict to 400-700 nm."
            ),
        },
    )

    if station_col and station_col in targets.columns:
        ds["station"] = ("obs", targets[station_col].astype(str).to_numpy())

    # Include all numeric pigment columns (ground truth) if present.
    excluded_cols = {"datetime", lat_col, lon_col}
    if station_col:
        excluded_cols.add(station_col)
    excluded_cols.add(columns.get("date", ""))
    excluded_cols.add(columns.get("time", ""))
    for col in targets.columns:
        if col in excluded_cols:
            continue
        if pd.api.types.is_numeric_dtype(targets[col]):
            ds[_sanitize_var_name(col)] = ("obs", pd.to_numeric(targets[col], errors="coerce").to_numpy(dtype=np.float32))

    # QC/meta fields
    accepted = np.zeros(targets.shape[0], dtype=bool)
    reason = np.array([""] * targets.shape[0], dtype=object)
    pace_file = np.array([""] * targets.shape[0], dtype=object)
    pixel_i = np.full(targets.shape[0], -1, dtype=np.int32)
    pixel_j = np.full(targets.shape[0], -1, dtype=np.int32)
    dt_seconds = np.full(targets.shape[0], np.nan, dtype=np.float32)
    distance_km = np.full(targets.shape[0], np.nan, dtype=np.float32)
    valid_pixels = np.full(targets.shape[0], 0, dtype=np.int16)
    cv_median = np.full(targets.shape[0], np.nan, dtype=np.float32)
    pixel_time = np.full(targets.shape[0], np.datetime64("NaT"), dtype="datetime64[ns]")

    for obs_id, meta in result_meta.items():
        accepted[obs_id] = bool(meta.get("ok", False))
        reason[obs_id] = str(meta.get("reason", ""))
        pace_file[obs_id] = str(meta.get("pace_file", ""))
        pixel_i[obs_id] = int(meta.get("pixel_i", -1))
        pixel_j[obs_id] = int(meta.get("pixel_j", -1))
        dt_seconds[obs_id] = float(meta.get("dt_seconds", np.nan))
        distance_km[obs_id] = float(meta.get("distance_km", np.nan))
        valid_pixels[obs_id] = int(meta.get("valid_pixels", 0))
        cv_median[obs_id] = float(meta.get("cv_median", np.nan))
        pt = meta.get("pixel_time")
        if isinstance(pt, pd.Timestamp):
            pixel_time[obs_id] = np.datetime64(pt.tz_convert("UTC").tz_localize(None).to_datetime64())

    ds["accepted"] = ("obs", accepted)
    ds["reject_reason"] = ("obs", reason.astype(str))
    ds["pace_file"] = ("obs", pace_file.astype(str))
    ds["pace_pixel_i"] = ("obs", pixel_i)
    ds["pace_pixel_j"] = ("obs", pixel_j)
    ds["pace_dt_seconds"] = ("obs", dt_seconds)
    ds["pace_distance_km"] = ("obs", distance_km)
    ds["valid_pixel_count"] = ("obs", valid_pixels)
    ds["cv_median"] = ("obs", cv_median)
    ds["pace_pixel_time"] = ("obs", pixel_time)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {"Rrs": {"zlib": True, "complevel": 4}}
    ds.to_netcdf(output_path, encoding=encoding)


def run(cfg_path: Path, dry_run: bool = False) -> Path:
    cfg = load_config_from_file(cfg_path)
    match_cfg = _build_match_config(cfg)

    # Load and filter in-situ targets.
    df_hplc, columns = _load_hplc_targets(cfg)
    df_hplc = _filter_targets_to_experiment(df_hplc, columns, cfg)
    if df_hplc.empty:
        raise RuntimeError("No in-situ observations remain after filtering to experiment bbox/time_range.")

    # Match-key de-duplication: replicate rows in HPLC datasets often share the same
    # lat/lon/time; we search/download once per unique tuple.
    lat_col = columns["lat"]
    lon_col = columns["lon"]
    df_hplc = df_hplc.reset_index(drop=True).copy()
    df_hplc["obs_id"] = np.arange(df_hplc.shape[0], dtype=int)

    unique_cols = [lat_col, lon_col, "datetime"]
    df_unique = df_hplc.drop_duplicates(subset=unique_cols).reset_index(drop=True).copy()
    df_unique["unique_id"] = np.arange(df_unique.shape[0], dtype=int)

    # Map each original obs row to a unique target row.
    df_map = df_hplc.merge(df_unique[unique_cols + ["unique_id"]], on=unique_cols, how="left", validate="many_to_one")
    if df_map["unique_id"].isna().any():
        raise RuntimeError("Internal error: failed to map observations to unique targets.")

    out_dir = _build_output_dir(cfg)

    output_filename = _get_nested(cfg, ["io", "pace_l2_rrs", "output_filename"], "pace_l2_rrs_matchups.nc")
    output_path = out_dir / str(output_filename)

    keep_raw = bool(_get_nested(cfg, ["io", "pace_l2_rrs", "keep_raw_granules"], False))
    raw_subdir = _get_nested(cfg, ["io", "pace_l2_rrs", "raw_granules_subdir"], "raw_granules")
    raw_dir = out_dir / str(raw_subdir)

    print("=" * 72)
    print("PACE OCI L2 Rrs Matchup Builder")
    print("=" * 72)
    print(f"Config: {cfg_path}")
    print(f"Observations: {df_hplc.shape[0]} rows ({df_unique.shape[0]} unique lat/lon/time)")
    print(f"Output dir: {out_dir}")
    print(f"Output file: {output_path}")
    print(f"Raw granules cache: {raw_dir} (keep={keep_raw})")

    if dry_run:
        print("Dry-run: stopping before creating output dirs / CMR search / downloads.")
        return output_path

    # Create output directories only for real runs.
    out_dir.mkdir(parents=True, exist_ok=True)

    session = _build_session()
    if not _get_token():
        raise RuntimeError(
            "Missing Earthdata token. Set EARTHDATA_TOKEN (preferred) or EARTHACCESS_TOKEN.\n"
            "Example:\n"
            "  export EARTHDATA_TOKEN='...'\n"
        )

    # Candidate selection (per unique target).
    per_obs, unique_granules = _choose_candidate_granules(
        session=session,
        cfg=cfg,
        match_cfg=match_cfg,
        targets=df_unique.set_index("unique_id"),
        columns=columns,
    )

    print(f"Selected {len(unique_granules)} unique granules for download across {len(per_obs)} targets.")

    # Download unique granules.
    downloaded: Dict[str, Path] = {}
    for gid, granule in unique_granules.items():
        urls = _get_download_urls(granule)
        if not urls:
            print(f"Warning: no download URL found for granule {gid}")
            continue
        url = urls[0]
        filename = os.path.basename(url.split("?")[0])
        if not filename:
            filename = (granule.get("producer_granule_id") or granule.get("title") or gid) + ".nc"
        dest = raw_dir / filename
        ok = _download_file(session, url, dest)
        if ok:
            downloaded[gid] = dest
        time.sleep(0.25)

    if not downloaded:
        raise RuntimeError("No granules were downloaded; cannot build matchups.")

    # Extract per-unique-target matchups.
    final_wl = np.arange(match_cfg.final_wl_lo, match_cfg.final_wl_hi + 1, match_cfg.interp_nm, dtype=np.int32)
    spectra_unique = np.full((df_unique.shape[0], final_wl.size), np.nan, dtype=np.float32)
    results_unique: Dict[int, Dict[str, Any]] = {}

    for unique_id, row in df_unique.set_index("unique_id").iterrows():
        obs_lat = float(row[columns["lat"]])
        obs_lon = float(row[columns["lon"]])
        obs_time = pd.Timestamp(row["datetime"]).tz_convert("UTC")

        candidates = per_obs.get(int(unique_id), [])
        if not candidates:
            results_unique[int(unique_id)] = {"ok": False, "reason": "no_candidate_granules"}
            continue

        chosen_result: Optional[Dict[str, Any]] = None
        for cand in candidates:
            gid = cand.get("id") or cand.get("producer_granule_id") or cand.get("title")
            if not gid or str(gid) not in downloaded:
                continue
            fpath = downloaded[str(gid)]
            res = _extract_matchup_from_file(fpath, obs_lat, obs_lon, obs_time, match_cfg)
            res["pace_file"] = fpath.name
            if res.get("ok"):
                chosen_result = res
                break
            # Keep last failure for visibility.
            chosen_result = res

        if not chosen_result:
            results_unique[int(unique_id)] = {"ok": False, "reason": "no_downloaded_candidate"}
            continue

        results_unique[int(unique_id)] = chosen_result
        if chosen_result.get("ok"):
            spectra_unique[int(unique_id), :] = chosen_result["rrs"]

    # Expand unique results back to full HPLC rows.
    spectra_full = np.full((df_hplc.shape[0], final_wl.size), np.nan, dtype=np.float32)
    results_full: Dict[int, Dict[str, Any]] = {}

    for obs_id, row in df_map.iterrows():
        uid = int(row["unique_id"])
        spectra_full[int(obs_id), :] = spectra_unique[uid, :]
        meta = dict(results_unique.get(uid, {"ok": False, "reason": "missing_unique_result"}))
        results_full[int(obs_id)] = meta

    _write_matchups_dataset(
        output_path=output_path,
        targets=df_hplc,
        columns=columns,
        spectra_400_700=spectra_full,
        wl_400_700=final_wl,
        result_meta=results_full,
    )

    # Optionally delete raw downloads.
    if not keep_raw:
        for path in raw_dir.glob("*.nc*"):
            try:
                path.unlink()
            except OSError:
                pass
        try:
            raw_dir.rmdir()
        except OSError:
            pass

    accepted = sum(1 for meta in results_full.values() if meta.get("ok"))
    print(f"Done. Accepted {accepted}/{df_hplc.shape[0]} observations.")
    print(f"Wrote: {output_path}")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to experiment YAML (reads experiment.bbox/time_range and validation.hplc.path)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print plan without downloading")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(args.config, dry_run=bool(args.dry_run))


if __name__ == "__main__":
    main()
