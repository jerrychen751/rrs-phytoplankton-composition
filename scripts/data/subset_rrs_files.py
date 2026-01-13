#!/usr/bin/env python3
"""Subset previously downloaded *mapped* PACE OCI RRS NetCDF files in-place.

This utility trims each Remote Sensing Reflectance (Rrs) granule to the
specified latitude/longitude bounding box and, optionally, a set of target
wavelengths.

This is part of the legacy "L3 mapped Rrs" workflow. The current validation
workflow uses PACE OCI *L2* AOP granules and produces per-station matchup
datasets via `scripts/data/download_pace_rrs.py`, which does not require this
post-processing step.

Example usage:
    python subset_rrs_files.py \
        --input-dir /path/to/rrs \
        --lat-range 33 35 \
        --lon-range -121 -119 \
        --wavelengths 443 486 551 \
        --inplace
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "experiments" / "seabass_validation" / "inputs" / "rrs"
DEFAULT_LAT_RANGE = (33.0, 35.0)
DEFAULT_LON_RANGE = (-121.0, -119.0)
DEFAULT_TOLERANCE_NM = 2.5


def ensure_lat_slice(latitudes: xr.DataArray, lat_range: Tuple[float, float]) -> Tuple[float, float]:
    """Return slice bounds compatible with dataset latitude ordering."""
    lat_min, lat_max = sorted(lat_range)
    values = latitudes.values
    if values.ndim != 1:
        raise ValueError("Latitude coordinate must be 1-D for slicing")
    ascending = bool(np.all(np.diff(values) > 0))
    return (lat_min, lat_max) if ascending else (lat_max, lat_min)


def select_wavelengths(
    available: Sequence[float] | np.ndarray,
    requested: Iterable[float] | None,
    tolerance_nm: float = DEFAULT_TOLERANCE_NM,
) -> Tuple[List[float], List[float]]:
    """Return (selected, missing) wavelength lists."""
    if not requested:
        return sorted(float(w) for w in available), []

    available_arr = np.asarray(available, dtype=float)
    selected: set[float] = set()
    missing: List[float] = []
    for wl in requested:
        idx = int(np.abs(available_arr - wl).argmin())
        closest = float(available_arr[idx])
        if abs(closest - wl) <= tolerance_nm:
            selected.add(closest)
        else:
            missing.append(float(wl))
    return sorted(selected), missing


def subset_dataset(
    ds: xr.Dataset,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    wavelengths: Iterable[float] | None,
    tolerance_nm: float,
) -> Tuple[xr.Dataset, List[float]]:
    """Subset an Rrs dataset and report missing wavelengths."""
    if "Rrs" not in ds.data_vars:
        raise KeyError("Dataset does not contain an 'Rrs' data variable")
    if "wavelength" not in ds.coords:
        raise KeyError("Dataset is missing a 'wavelength' coordinate")

    lat_slice = ensure_lat_slice(ds["lat"], lat_range)
    lon_min, lon_max = min(lon_range), max(lon_range)

    selected_wavelengths, missing = select_wavelengths(ds["wavelength"].values, wavelengths, tolerance_nm)
    if not selected_wavelengths:
        raise ValueError("No overlapping wavelengths found within tolerance")

    subset = ds[["Rrs"]].sel(
        wavelength=selected_wavelengths,
        lat=slice(*lat_slice),
        lon=slice(lon_min, lon_max),
    )

    subset = subset.drop_vars("palette", errors="ignore")
    drop_dims = [dim for dim in subset.dims if dim not in {"lat", "lon", "wavelength"}]
    if drop_dims:
        subset = subset.drop_dims(drop_dims)

    return subset, missing


def subset_rrs_file(
    file_path: Path,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    wavelengths: Iterable[float] | None = None,
    tolerance_nm: float = DEFAULT_TOLERANCE_NM,
    inplace: bool = True,
    output_suffix: str = "_subset",
    compression_level: int = 4,
) -> Path:
    """Subset a single NetCDF file and return the resulting path."""
    file_path = file_path.resolve()
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    try:
        ds = xr.open_dataset(file_path)
    except ValueError as exc:
        hint = (
            "xarray could not determine a backend. Install the 'netCDF4' or 'h5netcdf' package "
            "(e.g., pip install netCDF4)."
        )
        raise RuntimeError(f"Unable to open {file_path.name}: {exc}\n{hint}") from exc

    try:
        subset, missing = subset_dataset(ds, lat_range, lon_range, wavelengths, tolerance_nm)
    finally:
        ds.close()

    encoding = {"Rrs": {"zlib": True, "complevel": compression_level}}

    if inplace:
        target_path = file_path
    else:
        target_path = file_path.with_name(f"{file_path.stem}{output_suffix}{file_path.suffix}")
    tmp_path = target_path.with_suffix(target_path.suffix + ".tmp")

    subset.to_netcdf(tmp_path, encoding=encoding)
    subset.close()

    tmp_path.replace(target_path)

    if inplace:
        status = f"Overwrote {file_path.name}"
    else:
        status = f"Wrote {target_path.name}"
    if missing:
        status += f" (missing wavelengths: {', '.join(f'{wl:.1f}' for wl in missing)})"
    print(status)

    return target_path


def collect_files(input_dir: Path, pattern: str, recursive: bool) -> List[Path]:
    """Return sorted list of NetCDF files under input_dir."""
    input_dir = input_dir.expanduser().resolve()
    iterator = input_dir.rglob(pattern) if recursive else input_dir.glob(pattern)
    files = sorted(p for p in iterator if p.is_file() and p.suffix.lower() == ".nc")
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing downloaded RRS NetCDF files",
    )
    parser.add_argument(
        "--lat-range",
        metavar=("LAT_MIN", "LAT_MAX"),
        type=float,
        nargs=2,
        default=DEFAULT_LAT_RANGE,
        help="Latitude bounds for subsetting",
    )
    parser.add_argument(
        "--lon-range",
        metavar=("LON_MIN", "LON_MAX"),
        type=float,
        nargs=2,
        default=DEFAULT_LON_RANGE,
        help="Longitude bounds for subsetting",
    )
    parser.add_argument(
        "--wavelengths",
        type=float,
        nargs="*",
        default=None,
        help="Optional list of wavelengths (nm) to retain. Omit to keep all available",
    )
    parser.add_argument(
        "--tolerance-nm",
        type=float,
        default=DEFAULT_TOLERANCE_NM,
        help="Maximum deviation when matching requested wavelengths",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.nc",
        help="Glob pattern for files (supports subdirectories if recursive)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only inspect the top-level input directory",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Write subset to a new file instead of overwriting the source",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_subset",
        help="Suffix for new files when --keep-original is set",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="zlib compression level (0-9) used when writing NetCDF",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    input_dir: Path = args.input_dir
    lat_range = (min(args.lat_range), max(args.lat_range))
    lon_range = (min(args.lon_range), max(args.lon_range))
    recursive = not args.no_recursive
    wavelengths = args.wavelengths

    files = collect_files(input_dir, args.pattern, recursive)
    if not files:
        print(f"No NetCDF files found under {input_dir}")
        return 1

    print(f"Found {len(files)} file(s). Beginning subsetting...")
    errors = 0
    for file_path in files:
        try:
            subset_rrs_file(
                file_path=file_path,
                lat_range=lat_range,
                lon_range=lon_range,
                wavelengths=wavelengths,
                tolerance_nm=args.tolerance_nm,
                inplace=not args.keep_original,
                output_suffix=args.output_suffix,
                compression_level=args.compression_level,
            )
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"Failed to subset {file_path}: {exc}")

    if errors:
        print(f"Completed with {errors} error(s).")
        return 2

    print("Subsetting complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
