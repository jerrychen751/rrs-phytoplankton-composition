"""PACE OCI Level-2 (swath) utilities for SDP validation workflows.

This module focuses on the pieces that are *structural prerequisites* for the
SDP pigments model:

- Extracting the nearest-pixel spectrum from PACE OCI L2 AOP granules.
- Applying the QC gates described in `docs/notes.md`:
    - l2_flags masking
    - per-pixel spectral validity (finite fraction)
- Performing the spectral preprocessing used by Kramer et al.:
    - interpolate to 1 nm grid
    - smooth with a 5 nm moving mean
    - trim edges to avoid smoothing artifacts

The model code expects above-surface Rrs on a fixed 400–700 nm 1 nm grid.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import xarray as xr


PACE_L2_TIMESTAMP_RE = re.compile(r"PACE_OCI\.(\d{8}T\d{6})\.L2\.")


def parse_pace_l2_granule_time(path: Path) -> Optional[pd.Timestamp]:
    """Parse a UTC timestamp from a PACE L2 filename, if present."""
    match = PACE_L2_TIMESTAMP_RE.search(path.name)
    if not match:
        return None
    stamp = match.group(1)
    ts = pd.to_datetime(stamp, format="%Y%m%dT%H%M%S", utc=True)
    return ts


def _wrap_lon_delta_deg(delta: np.ndarray) -> np.ndarray:
    """Wrap longitude deltas into [-180, 180]."""
    return (delta + 180.0) % 360.0 - 180.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance (km) between two lat/lon points (degrees)."""
    r_km = 6371.0088
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r_km * c


def list_nc_files(directory: Path) -> list[Path]:
    """Collect NetCDF files, supporting both .nc and .nc4 extensions."""
    files: set[Path] = set()
    for pattern in ("*.nc", "*.nc4"):
        for path in directory.glob(pattern):
            files.add(path.resolve())
    return sorted(files)


def _coerce_wavelength_vector(values: np.ndarray, target_len: int) -> np.ndarray:
    """Coerce wavelength arrays from common PACE L2 layouts into a 1-D vector."""
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        if arr.size != target_len:
            raise ValueError(f"Unexpected wavelength length {arr.size} (expected {target_len})")
        return arr

    # Common cases:
    # - (bands, 1) or (1, bands)
    # - (lines, pixels, bands) -> take first pixel
    if arr.ndim == 2:
        if arr.shape[0] == target_len:
            return arr[:, 0]
        if arr.shape[1] == target_len:
            return arr[0, :]
    if arr.ndim >= 3 and arr.shape[-1] == target_len:
        # e.g., (lines, pixels, bands)
        return arr.reshape(-1, target_len)[0, :]

    raise ValueError(f"Unable to coerce wavelength array of shape {arr.shape} to len={target_len}")


def _get_l2_rrs_wavelengths(path: Path, geophys: xr.Dataset, rrs_var: str = "Rrs") -> np.ndarray:
    """Return the wavelength centers (nm) for the Rrs spectral dimension."""
    if rrs_var not in geophys:
        raise KeyError(f"Missing variable '{rrs_var}' in geophysical_data group: {path}")
    rrs = geophys[rrs_var]
    band_dim = rrs.dims[-1]
    n_bands = int(rrs.sizes[band_dim])

    # Prefer an in-group wavelength variable if present.
    # Try wavelength_3d first — in PACE OCI L2 files, the Rrs band dimension
    # is named "wavelength_3d" (172 visible/NIR bands), while the full sensor
    # "wavelength" array has 286 bands and won't match.
    for candidate in ("wavelength_3d", "wavelength", band_dim):
        if candidate in geophys:
            wl = geophys[candidate].values
            try:
                return _coerce_wavelength_vector(wl, n_bands)
            except ValueError:
                pass

    # Fallback: sensor band parameters group.
    try:
        with xr.open_dataset(
            path,
            group="sensor_band_parameters",
            mask_and_scale=True,
            decode_cf=True,
            decode_timedelta=False,
        ) as band:
            for candidate in ("wavelength_3d", "wavelength"):
                if candidate in band:
                    try:
                        return _coerce_wavelength_vector(band[candidate].values, n_bands)
                    except ValueError:
                        pass
    except Exception:
        # Let the error below be the actionable one.
        pass

    raise KeyError(
        "Unable to locate wavelength coordinate for Rrs. "
        "Tried geophysical_data:{wavelength,wavelength_3d} and sensor_band_parameters:{wavelength,wavelength_3d}."
    )


def moving_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving mean with NaN-aware handling (requires a full window)."""
    if window <= 0 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")

    values = np.asarray(values, dtype=float)
    kernel = np.ones(window, dtype=float)

    finite = np.isfinite(values)
    sum_ = np.convolve(np.where(finite, values, 0.0), kernel, mode="same")
    count = np.convolve(finite.astype(float), kernel, mode="same")

    out = np.full_like(values, np.nan, dtype=float)
    full = count >= float(window)
    out[full] = sum_[full] / count[full]
    return out


def preprocess_rrs_spectrum(
    wavelengths_nm: np.ndarray,
    rrs: np.ndarray,
    *,
    interp_nm: int,
    smooth_nm: int,
    edge_trim_nm: int,
    final_range_nm: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate + smooth a single Rrs spectrum to a fixed 1 nm grid.

    This follows the paper workflow described in `docs/notes.md`:
    - interpolate to 1 nm
    - 5 nm moving mean
    - trim edges to remove smoothing artifacts

    Implementation detail: to make the *final* [400, 700] values valid after
    smoothing, we interpolate on an *extended* grid (final_range +/- edge_trim)
    and then trim back to final_range.
    """
    wl = np.asarray(wavelengths_nm, dtype=float)
    y = np.asarray(rrs, dtype=float)
    if wl.shape != y.shape:
        raise ValueError(f"wavelengths and rrs must have the same shape; got {wl.shape} vs {y.shape}")

    if len(final_range_nm) != 2:
        raise ValueError("final_range_nm must be a (min, max) pair")
    final_min, final_max = int(final_range_nm[0]), int(final_range_nm[1])
    if final_min >= final_max:
        raise ValueError(f"Invalid final_range_nm: {final_range_nm}")

    extended_min = final_min - int(edge_trim_nm)
    extended_max = final_max + int(edge_trim_nm)
    target = np.arange(extended_min, extended_max + 1, int(interp_nm), dtype=float)

    finite = np.isfinite(wl) & np.isfinite(y)
    if finite.sum() < 2:
        raise ValueError("Not enough finite wavelength/value pairs to interpolate")

    wl = wl[finite]
    y = y[finite]

    order = np.argsort(wl)
    wl = wl[order]
    y = y[order]

    # De-duplicate wavelength centers (keep the first occurrence).
    uniq_wl, uniq_idx = np.unique(wl, return_index=True)
    wl = uniq_wl
    y = y[uniq_idx]
    if wl.size < 2:
        raise ValueError("Not enough unique wavelength points to interpolate")

    interp = np.full_like(target, np.nan, dtype=float)
    in_range = (target >= wl.min()) & (target <= wl.max())
    if in_range.any():
        interp[in_range] = np.interp(target[in_range], wl, y)

    smoothed = moving_mean(interp, window=int(smooth_nm))

    keep = (target >= final_min) & (target <= final_max)
    out_wl = target[keep].astype(int)
    out_rrs = smoothed[keep]

    if out_wl[0] != final_min or out_wl[-1] != final_max:
        raise AssertionError("Internal error: output wavelength grid does not match requested final_range_nm")
    return out_wl.astype(float), out_rrs


def _haversine_km_vec(
    lat1: float,
    lon1: float,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Great-circle distance (km) from one point to an array of lat/lon points.

    Vectorized version of haversine_km(): uses NumPy broadcasting so that
    all distances are computed in a single pass through C-level array ops,
    avoiding a Python loop over N pixels.

    Args:
        lat1: Scalar latitude of the reference point (degrees).
        lon1: Scalar longitude of the reference point (degrees).
        lat2: Array of latitudes (degrees), any shape.
        lon2: Array of longitudes (degrees), same shape as lat2.

    Returns:
        Array of distances in km, same shape as lat2/lon2.
    """
    r_km = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(((lon2 - lon1 + 180.0) % 360.0) - 180.0)

    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return r_km * c


@dataclass(frozen=True)
class L2NeighborhoodMatchup:
    """
    Result of a neighborhood-averaged L2 matchup.

    Instead of a single nearest pixel, this holds the median spectrum
    computed from all valid pixels within a radius of the station. The
    spectrum is in native PACE bands (not yet interpolated to 1 nm).
    """
    obs_index: int
    obs_time: pd.Timestamp
    obs_lat: float
    obs_lon: float
    station: Optional[str]

    granule_path: Path
    granule_time: Optional[pd.Timestamp]

    center_i: int
    center_j: int
    center_lat: float
    center_lon: float
    center_distance_km: float

    wavelengths_nm: np.ndarray    # native PACE band centers
    rrs_median_native: np.ndarray  # median spectrum in native bands

    n_pixels_in_radius: int   # total pixels within radius (before QC)
    n_valid_pixels: int       # pixels that passed QC + finite checks


@dataclass(frozen=True)
class L2Matchup:
    obs_index: int
    obs_time: pd.Timestamp
    obs_lat: float
    obs_lon: float
    station: Optional[str]

    granule_path: Path
    granule_time: Optional[pd.Timestamp]

    pixel_i: int
    pixel_j: int
    pixel_lat: float
    pixel_lon: float
    distance_km: float

    rrs_400_700_1nm: np.ndarray  # shape: (301,)


def _nearest_pixel(lat: np.ndarray, lon: np.ndarray, target_lat: float, target_lon: float) -> tuple[int, int]:
    """Return (i, j) index of the nearest pixel in a 2-D lat/lon grid."""
    if lat.shape != lon.shape or lat.ndim != 2:
        raise ValueError(f"Expected lat/lon 2-D arrays with same shape; got {lat.shape} vs {lon.shape}")

    finite = np.isfinite(lat) & np.isfinite(lon)
    if not finite.any():
        raise ValueError("Latitude/longitude arrays contain no finite values")

    dlat = lat - float(target_lat)
    dlon = _wrap_lon_delta_deg(lon - float(target_lon))
    # Basic degree-space weighting for longitude.
    scale_lon = math.cos(math.radians(float(target_lat)))
    dist2 = dlat**2 + (dlon * scale_lon) ** 2
    dist2 = np.where(finite, dist2, np.inf)
    flat = int(np.argmin(dist2))
    i, j = np.unravel_index(flat, lat.shape)
    return int(i), int(j)


def extract_l2_matchup(
    granule_path: Path,
    *,
    obs_index: int,
    obs_time: pd.Timestamp,
    obs_lat: float,
    obs_lon: float,
    station: Optional[str],
    exclude_bits: Sequence[int],
    min_finite_fraction: float,
    interp_nm: int,
    smooth_nm: int,
    edge_trim_nm: int,
    final_range_nm: Sequence[int],
) -> Optional[L2Matchup]:
    """Attempt to build a single observation->granule matchup.

    Returns None if the granule cannot produce a valid spectrum due to QC filters.
    """
    granule_path = granule_path.expanduser().resolve()
    granule_time = parse_pace_l2_granule_time(granule_path)

    exclude_mask = 0
    for bit in exclude_bits:
        exclude_mask |= 1 << int(bit)

    with xr.open_dataset(
        granule_path,
        group="navigation_data",
        mask_and_scale=True,
        decode_cf=True,
        decode_timedelta=False,
    ) as nav, xr.open_dataset(
        granule_path,
        group="geophysical_data",
        mask_and_scale=True,
        decode_cf=True,
        decode_timedelta=False,
    ) as geophys:
        if "latitude" not in nav or "longitude" not in nav:
            raise KeyError(f"navigation_data missing latitude/longitude: {granule_path.name}")
        if "Rrs" not in geophys or "l2_flags" not in geophys:
            raise KeyError(f"geophysical_data missing Rrs/l2_flags: {granule_path.name}")

        lat2d = np.asarray(nav["latitude"].values, dtype=float)
        lon2d = np.asarray(nav["longitude"].values, dtype=float)
        i, j = _nearest_pixel(lat2d, lon2d, obs_lat, obs_lon)

        pixel_lat = float(lat2d[i, j])
        pixel_lon = float(lon2d[i, j])
        distance_km = haversine_km(obs_lat, obs_lon, pixel_lat, pixel_lon)

        rrs_da = geophys["Rrs"]
        flags_da = geophys["l2_flags"]
        wl_native = _get_l2_rrs_wavelengths(granule_path, geophys, rrs_var="Rrs")

        # ------------------------------------------------------------------
        # Flow (per user request):
        # 1) Candidate granules already selected by time (outside this function)
        # 2) Apply QC directly (no neighborhood aggregation)
        # 3) Spectral preprocessing: interpolate -> smooth (5 nm) -> edge-safe trim
        # ------------------------------------------------------------------

        pixel_rrs = np.asarray(
            rrs_da.isel({rrs_da.dims[0]: int(i), rrs_da.dims[1]: int(j)}).values,
            dtype=float,
        )
        pixel_flags = flags_da.isel({flags_da.dims[0]: int(i), flags_da.dims[1]: int(j)}).values

        if not np.isfinite(pixel_flags):
            return None
        flags_int = int(np.int64(pixel_flags))
        if (flags_int & exclude_mask) != 0:
            return None

        finite_fraction = float(np.isfinite(pixel_rrs).mean())
        if finite_fraction < float(min_finite_fraction):
            return None

        rrs_native = pixel_rrs

    # Spectral preprocessing to 400–700 @ 1 nm on the nearest-pixel spectrum.
    _, rrs_400_700 = preprocess_rrs_spectrum(
        wl_native,
        rrs_native,
        interp_nm=interp_nm,
        smooth_nm=smooth_nm,
        edge_trim_nm=edge_trim_nm,
        final_range_nm=final_range_nm,
    )

    if rrs_400_700.shape != (301,):
        raise AssertionError(f"Expected 301 wavelengths (400-700 inclusive); got {rrs_400_700.shape}")
    if not np.isfinite(rrs_400_700).all():
        # Treat partially missing spectra as unusable for SDP (derivatives explode).
        return None

    return L2Matchup(
        obs_index=int(obs_index),
        obs_time=obs_time,
        obs_lat=float(obs_lat),
        obs_lon=float(obs_lon),
        station=station,
        granule_path=granule_path,
        granule_time=granule_time,
        pixel_i=int(i),
        pixel_j=int(j),
        pixel_lat=float(pixel_lat),
        pixel_lon=float(pixel_lon),
        distance_km=float(distance_km),
        rrs_400_700_1nm=rrs_400_700.astype(float),
    )


def extract_l2_neighborhood_matchup(
    granule_path: Path,
    *,
    obs_index: int,
    obs_time: pd.Timestamp,
    obs_lat: float,
    obs_lon: float,
    station: Optional[str],
    exclude_bits: Sequence[int],
    min_finite_fraction: float,
    radius_km: float,
    min_valid_pixels: int,
    max_distance_km: float,
) -> Optional[L2NeighborhoodMatchup]:
    """
    Extract a neighborhood-averaged L2 spectrum within a radius of a station.

    Instead of returning the single nearest pixel (like extract_l2_matchup),
    this function:
      1. Finds the nearest pixel as the center point.
      2. Extracts a bounding box of pixels around that center.
      3. Computes haversine distance from the station to each pixel.
      4. Applies QC masks (l2_flags, finite fraction) to each pixel.
      5. Takes the nanmedian across all valid pixels within the radius.

    The returned spectrum is in **native PACE bands** (not interpolated to 1 nm).
    The caller is responsible for running preprocess_rrs_spectrum() on the result,
    which is intentional: median of raw spectra then preprocess is better than
    preprocessing each pixel then averaging (avoids smoothing-of-smoothing and
    lets spatial outlier rejection work on unsmoothed data).

    Args:
        granule_path: Path to a PACE OCI L2 AOP NetCDF file.
        obs_index: Integer index of this observation (for bookkeeping).
        obs_time: UTC timestamp of the in-situ observation.
        obs_lat: Latitude of the in-situ station (degrees).
        obs_lon: Longitude of the in-situ station (degrees).
        station: Optional station identifier string.
        exclude_bits: l2_flags bit positions that disqualify a pixel.
        min_finite_fraction: Minimum fraction of spectral bands that must be
            finite for a pixel to be considered valid.
        radius_km: Search radius around the station (km).
        min_valid_pixels: Minimum number of QC-passing pixels required for a
            valid matchup. Below this, the median is too unstable.
        max_distance_km: Maximum distance from station to nearest pixel center.
            If the nearest pixel is farther than this, the station is outside
            the granule's useful coverage and we return None.

    Returns:
        L2NeighborhoodMatchup with native-band median spectrum, or None if
        the matchup fails QC (too few valid pixels, station too far, etc.).
    """
    granule_path = granule_path.expanduser().resolve()
    granule_time = parse_pace_l2_granule_time(granule_path)

    exclude_mask = 0
    for bit in exclude_bits:
        exclude_mask |= 1 << int(bit)

    with xr.open_dataset(
        granule_path,
        group="navigation_data",
        mask_and_scale=True,
        decode_cf=True,
        decode_timedelta=False,
    ) as nav, xr.open_dataset(
        granule_path,
        group="geophysical_data",
        mask_and_scale=True,
        decode_cf=True,
        decode_timedelta=False,
    ) as geophys:
        if "latitude" not in nav or "longitude" not in nav:
            raise KeyError(f"navigation_data missing latitude/longitude: {granule_path.name}")
        if "Rrs" not in geophys or "l2_flags" not in geophys:
            raise KeyError(f"geophysical_data missing Rrs/l2_flags: {granule_path.name}")

        lat2d = np.asarray(nav["latitude"].values, dtype=float)
        lon2d = np.asarray(nav["longitude"].values, dtype=float)

        # Step 1: Find nearest pixel as center reference.
        center_i, center_j = _nearest_pixel(lat2d, lon2d, obs_lat, obs_lon)
        center_lat = float(lat2d[center_i, center_j])
        center_lon = float(lon2d[center_i, center_j])
        center_dist_km = haversine_km(obs_lat, obs_lon, center_lat, center_lon)

        if center_dist_km > max_distance_km:
            return None

        # Step 2: Compute bounding box in index space.
        # PACE OCI pixels are ~1 km at nadir, wider at swath edges.
        # Using 0.8 km as a conservative lower bound on pixel spacing
        # ensures the bounding box is large enough to capture all pixels
        # within radius_km even at nadir.
        margin = int(math.ceil(radius_km / 0.8))
        n_lines, n_pixels = lat2d.shape
        i_lo = max(0, center_i - margin)
        i_hi = min(n_lines, center_i + margin + 1)
        j_lo = max(0, center_j - margin)
        j_hi = min(n_pixels, center_j + margin + 1)

        # Step 3: Extract subarrays within bounding box.
        sub_lat = lat2d[i_lo:i_hi, j_lo:j_hi]
        sub_lon = lon2d[i_lo:i_hi, j_lo:j_hi]

        # Step 4: Compute haversine distance from station to each pixel.
        dist_km = _haversine_km_vec(obs_lat, obs_lon, sub_lat, sub_lon)
        within_radius = dist_km <= radius_km

        rrs_da = geophys["Rrs"]
        flags_da = geophys["l2_flags"]
        wl_native = _get_l2_rrs_wavelengths(granule_path, geophys, rrs_var="Rrs")
        n_bands = len(wl_native)

        # Extract Rrs and flags subarrays (dims: lines, pixels, [bands]).
        sub_rrs = np.asarray(
            rrs_da.isel({
                rrs_da.dims[0]: slice(i_lo, i_hi),
                rrs_da.dims[1]: slice(j_lo, j_hi),
            }).values,
            dtype=float,
        )  # shape: (sub_lines, sub_pixels, n_bands)
        sub_flags = flags_da.isel({
            flags_da.dims[0]: slice(i_lo, i_hi),
            flags_da.dims[1]: slice(j_lo, j_hi),
        }).values  # shape: (sub_lines, sub_pixels)

        # Step 5: Build QC mask.
        # flags_clean: pixel-level l2_flags check (no excluded bits set).
        # Replace NaN with 0 before int cast to avoid undefined behavior
        # (NaN.astype(int64) produces garbage values in NumPy).
        flags_finite = np.isfinite(sub_flags)
        safe_flags = np.where(flags_finite, sub_flags, 0).astype(np.int64)
        flags_clean = flags_finite & ((safe_flags & exclude_mask) == 0)

        # finite_ok: per-pixel spectral completeness check.
        finite_fraction = np.isfinite(sub_rrs).mean(axis=-1)
        finite_ok = finite_fraction >= min_finite_fraction

        valid_mask = within_radius & flags_clean & finite_ok

        n_in_radius = int(within_radius.sum())
        n_valid = int(valid_mask.sum())

        if n_valid < min_valid_pixels:
            return None

        # Step 6: Extract valid spectra and compute median.
        # Reshape sub_rrs to (n_sub_pixels, n_bands) then select valid rows.
        flat_rrs = sub_rrs.reshape(-1, n_bands)
        flat_valid = valid_mask.ravel()
        valid_spectra = flat_rrs[flat_valid]  # shape: (n_valid, n_bands)

        median_spectrum = np.nanmedian(valid_spectra, axis=0)

    return L2NeighborhoodMatchup(
        obs_index=int(obs_index),
        obs_time=obs_time,
        obs_lat=float(obs_lat),
        obs_lon=float(obs_lon),
        station=station,
        granule_path=granule_path,
        granule_time=granule_time,
        center_i=int(center_i),
        center_j=int(center_j),
        center_lat=float(center_lat),
        center_lon=float(center_lon),
        center_distance_km=float(center_dist_km),
        wavelengths_nm=wl_native,
        rrs_median_native=median_spectrum,
        n_pixels_in_radius=n_in_radius,
        n_valid_pixels=n_valid,
    )


def select_candidate_granules(
    granules: Sequence[Path],
    obs_time: pd.Timestamp,
    *,
    time_window_hours: float,
    max_granules: int,
) -> list[Path]:
    """Pick up to `max_granules` granules closest in time to an observation."""
    candidates: list[tuple[float, Path]] = []
    for path in granules:
        ts = parse_pace_l2_granule_time(path)
        if ts is None:
            continue
        dt_hours = abs((ts - obs_time).total_seconds()) / 3600.0
        if dt_hours <= float(time_window_hours):
            candidates.append((dt_hours, path))

    candidates.sort(key=lambda pair: pair[0])
    return [p for _, p in candidates[: int(max_granules)]]


def coerce_timestamp(value: Any, timezone: str) -> pd.Timestamp:
    """Coerce an input datetime into a tz-aware UTC pandas Timestamp."""
    ts = pd.to_datetime(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize(timezone)
    return ts.tz_convert("UTC")


def load_hplc_observations(
    path: Path,
    *,
    lat_key: str = "lat",
    lon_key: str = "lon",
    date_key: str = "date",
    time_key: str = "time",
    station_key: str = "station",
    date_format: str = "%Y%m%d",
    time_format: str = "%H:%M:%S",
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Load an in-situ HPLC dataset (CSV or SeaBASS) into a normalized DataFrame.

    Returned DataFrame includes:
      - `obs_time` (UTC Timestamp)
      - `obs_lat`, `obs_lon`
      - `station` (optional)
    and preserves any additional pigment columns for later analysis.
    """
    path = path.expanduser().resolve()
    frames: list[pd.DataFrame] = []

    def _load_one(file_path: Path) -> pd.DataFrame:
        if file_path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        # SeaBASS files are often .sb or .txt in practice; we treat non-CSV as SeaBASS-ish.
        from utils.seabass_loader import load_seabass_file

        df, _ = load_seabass_file(file_path)
        return df

    if path.is_dir():
        for file_path in sorted(path.glob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in {".csv", ".sb", ".txt"}:
                continue
            frames.append(_load_one(file_path))
    else:
        frames.append(_load_one(path))

    if not frames:
        raise ValueError(f"No HPLC files found under {path}")
    df = pd.concat(frames, ignore_index=True)

    missing = [k for k in (lat_key, lon_key, date_key, time_key) if k not in df.columns]
    if missing:
        raise ValueError(f"HPLC dataset missing required columns {missing}. Found: {list(df.columns)}")

    obs_lat = pd.to_numeric(df[lat_key], errors="coerce")
    obs_lon = pd.to_numeric(df[lon_key], errors="coerce")
    date_str = df[date_key].astype(str)
    time_str = df[time_key].astype(str)

    parsed = pd.to_datetime(
        date_str + " " + time_str,
        format=f"{date_format} {time_format}",
        errors="coerce",
    )
    if parsed.dt.tz is None:
        parsed = parsed.dt.tz_localize(timezone)
    obs_time = parsed.dt.tz_convert("UTC")

    out = df.copy()
    out["obs_time"] = obs_time
    out["obs_lat"] = obs_lat
    out["obs_lon"] = obs_lon
    if station_key in out.columns:
        out["station"] = out[station_key]
    else:
        out["station"] = None

    out = out.dropna(subset=["obs_time", "obs_lat", "obs_lon"])
    return out
