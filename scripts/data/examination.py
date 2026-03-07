#!/usr/bin/env python3
"""Examine representative input files for rrs/sss/sst datasets.

Run this script directly to produce a detailed, human-readable report
about data resolution, coverage, quality, missing values, and basic
statistics for one sample file from each input category.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import sys
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import xarray as xr
except ImportError as exc:  # pragma: no cover - used for user feedback
    raise SystemExit(
        "xarray is required to run this script. Please install it in your environment."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = REPO_ROOT / "experiments/initial_test/inputs"
DEFAULT_CATEGORIES = ("rrs", "sss", "sst")

LAT_NAMES = ("lat", "latitude", "nav_lat", "y")
LON_NAMES = ("lon", "longitude", "nav_lon", "x")
TIME_NAMES = ("time", "times", "date", "day")

FLAG_HINTS = ("flag", "flags", "quality", "mask")
PRIMARY_HINTS = (
    "rrs",
    "Rrs",
    "sst",
    "SST",
    "sss",
    "SSS",
    "salinity",
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a detailed report for one rrs/sss/sst file per category."
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=None,
        help="Base directory containing rrs/sss/sst subfolders (default: repo-root/experiments/initial_test/inputs).",
    )
    parser.add_argument(
        "--categories",
        nargs="*",
        default=None,
        help="Optional list of category subfolders to inspect (default: rrs sss sst).",
    )
    parser.add_argument(
        "--pick",
        choices=("first", "latest"),
        default="first",
        help="Which file to choose from each category (sorted order or latest mtime).",
    )
    parser.add_argument(
        "--max-elements",
        type=int,
        default=5_000_000,
        help="Maximum elements to use for stats before sampling.",
    )
    parser.add_argument(
        "--max-unique",
        type=int,
        default=25,
        help="Maximum unique values to print for flag-like variables.",
    )
    return parser.parse_args(argv)


def human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} PB"


def format_value(value: object, max_len: int = 160) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def pick_sample_file(files: Sequence[Path], strategy: str) -> Optional[Path]:
    if not files:
        return None
    if strategy == "latest":
        return max(files, key=lambda p: p.stat().st_mtime)
    return sorted(files)[0]


def find_coord(ds: xr.Dataset, names: Iterable[str]) -> Optional[xr.DataArray]:
    for name in names:
        if name in ds.coords:
            return ds.coords[name]
    for name in names:
        if name in ds:
            return ds[name]
    return None


def summarize_axis(coord: xr.DataArray) -> Optional[dict]:
    values = coord.values
    if values.size == 0:
        return None
    values = np.asarray(values)
    if values.ndim == 0:
        return None

    dtype = values.dtype
    is_numeric = np.issubdtype(dtype, np.number)
    if np.issubdtype(dtype, np.datetime64):
        valid = values[~np.isnat(values)]
        if valid.size == 0:
            return None
        min_val = valid.min()
        max_val = valid.max()
    elif np.issubdtype(dtype, np.timedelta64):
        min_val = values.min()
        max_val = values.max()
    elif is_numeric:
        min_val = float(np.nanmin(values))
        max_val = float(np.nanmax(values))
    else:
        min_val = values.min()
        max_val = values.max()
    spacing = None
    diffs = None
    can_diff = is_numeric or np.issubdtype(dtype, np.datetime64) or np.issubdtype(dtype, np.timedelta64)
    if can_diff:
        if values.ndim == 1:
            diffs = np.diff(values)
        elif values.ndim >= 2:
            diffs = np.diff(values, axis=-1).ravel()
            if np.issubdtype(diffs.dtype, np.timedelta64):
                any_finite = np.any(~np.isnat(diffs))
            else:
                any_finite = np.any(np.isfinite(diffs))
            if not any_finite:
                diffs = np.diff(values, axis=0).ravel()

    if diffs is not None:
        if np.issubdtype(diffs.dtype, np.timedelta64):
            diffs = diffs[~np.isnat(diffs)]
        else:
            diffs = diffs[np.isfinite(diffs)]
        diffs = diffs[np.abs(diffs) > 0]
        if diffs.size:
            if np.issubdtype(diffs.dtype, np.timedelta64):
                diffs_sec = diffs / np.timedelta64(1, "s")
                spacing = {
                    "median": float(np.median(diffs_sec)),
                    "min": float(np.min(diffs_sec)),
                    "max": float(np.max(diffs_sec)),
                    "units": "seconds",
                }
            else:
                spacing = {
                    "median": float(np.median(diffs)),
                    "min": float(np.min(diffs)),
                    "max": float(np.max(diffs)),
                }

    return {
        "min": min_val,
        "max": max_val,
        "count": int(values.size),
        "spacing": spacing,
        "dtype": str(dtype),
        "dims": coord.dims,
        "shape": coord.shape,
        "is_numeric": is_numeric,
    }


def infer_resolution_from_attrs(ds: xr.Dataset) -> List[str]:
    keys = (
        "spatial_resolution",
        "geospatial_lat_resolution",
        "geospatial_lon_resolution",
        "geospatial_lat_min",
        "geospatial_lat_max",
        "geospatial_lon_min",
        "geospatial_lon_max",
    )
    lines = []
    for key in keys:
        if key in ds.attrs:
            lines.append(f"  {key}: {format_value(ds.attrs[key])}")
    return lines


def describe_global_attrs(ds: xr.Dataset) -> List[str]:
    preferred = (
        "title",
        "summary",
        "institution",
        "source",
        "platform",
        "instrument",
        "processing_level",
        "cdm_data_type",
        "Conventions",
        "time_coverage_start",
        "time_coverage_end",
        "time_coverage_duration",
    )
    lines = []
    shown = 0
    for key in preferred:
        if key in ds.attrs:
            lines.append(f"  {key}: {format_value(ds.attrs[key])}")
            shown += 1
    total = len(ds.attrs)
    if total:
        lines.append(f"  attributes shown: {shown} of {total}")
    return lines


def looks_like_flag(name: str, da: xr.DataArray) -> bool:
    lower = name.lower()
    if any(hint in lower for hint in FLAG_HINTS):
        return True
    if "flag_values" in da.attrs or "flag_masks" in da.attrs:
        return True
    if np.issubdtype(da.dtype, np.integer):
        return True
    return False


def looks_like_primary(name: str) -> bool:
    return any(hint in name for hint in PRIMARY_HINTS)


def extract_fill_values(da: xr.DataArray) -> List[str]:
    lines = []
    for key in ("_FillValue", "missing_value", "fill_value"):
        if key in da.attrs:
            lines.append(f"{key}={format_value(da.attrs[key])}")
    return lines


def extract_valid_range(da: xr.DataArray) -> Tuple[Optional[float], Optional[float], List[str]]:
    notes = []
    valid_min = None
    valid_max = None
    if "valid_range" in da.attrs:
        value = da.attrs["valid_range"]
        try:
            valid_min = float(np.asarray(value)[0])
            valid_max = float(np.asarray(value)[1])
            notes.append(f"valid_range={format_value(value)}")
        except Exception:
            notes.append(f"valid_range={format_value(value)}")
    if "valid_min" in da.attrs:
        try:
            valid_min = float(da.attrs["valid_min"])
            notes.append(f"valid_min={valid_min}")
        except Exception:
            notes.append(f"valid_min={format_value(da.attrs['valid_min'])}")
    if "valid_max" in da.attrs:
        try:
            valid_max = float(da.attrs["valid_max"])
            notes.append(f"valid_max={valid_max}")
        except Exception:
            notes.append(f"valid_max={format_value(da.attrs['valid_max'])}")
    return valid_min, valid_max, notes


def sample_flattened(values: np.ndarray, max_elements: int) -> Tuple[np.ndarray, Optional[str]]:
    flat = values.ravel()
    if max_elements <= 0 or flat.size <= max_elements:
        return flat, None
    idx = np.linspace(0, flat.size - 1, num=max_elements, dtype=np.int64)
    return flat[idx], f"sampled {max_elements} of {flat.size} values (evenly spaced)"


def summarize_unique(values: np.ndarray, max_unique: int) -> str:
    unique, counts = np.unique(values, return_counts=True)
    order = np.argsort(counts)[::-1]
    unique = unique[order]
    counts = counts[order]
    if unique.size > max_unique:
        unique = unique[:max_unique]
        counts = counts[:max_unique]
        suffix = f" (top {max_unique} values)"
    else:
        suffix = ""
    pairs = ", ".join(f"{format_value(u)}:{int(c)}" for u, c in zip(unique, counts))
    return pairs + suffix


def analyze_variable(
    name: str, da: xr.DataArray, max_elements: int, max_unique: int
) -> List[str]:
    lines = [f"  Variable: {name}"]
    lines.append(
        f"    dims={da.dims} shape={da.shape} dtype={da.dtype}"
    )
    if "units" in da.attrs:
        lines.append(f"    units={format_value(da.attrs['units'])}")
    if "long_name" in da.attrs:
        lines.append(f"    long_name={format_value(da.attrs['long_name'])}")
    if "standard_name" in da.attrs:
        lines.append(f"    standard_name={format_value(da.attrs['standard_name'])}")

    fill_values = extract_fill_values(da)
    if fill_values:
        lines.append(f"    fill/missing: {', '.join(fill_values)}")

    valid_min, valid_max, valid_notes = extract_valid_range(da)
    if valid_notes:
        lines.append(f"    valid_range: {', '.join(valid_notes)}")

    data = da.values
    mask = None
    if isinstance(data, np.ma.MaskedArray):
        mask = np.ma.getmaskarray(data)
        data = np.ma.getdata(data)

    data = np.asarray(data)
    mask_sample = None
    data_sample, sample_note = sample_flattened(data, max_elements)
    if mask is not None:
        mask_sample, _ = sample_flattened(np.asarray(mask), max_elements)

    total = int(data_sample.size)
    if total == 0:
        lines.append("    stats: empty array")
        return lines

    if mask_sample is not None:
        data_for_stats = data_sample.astype(float, copy=False)
        data_for_stats = np.where(mask_sample, np.nan, data_for_stats)
        mask_count = int(np.sum(mask_sample))
    else:
        data_for_stats = data_sample.astype(float, copy=False) if np.issubdtype(data_sample.dtype, np.number) else data_sample
        mask_count = 0

    nan_count = int(np.isnan(data_for_stats).sum()) if np.issubdtype(data_for_stats.dtype, np.number) else 0
    inf_count = int(np.isinf(data_for_stats).sum()) if np.issubdtype(data_for_stats.dtype, np.number) else 0
    valid_count = total - (mask_count + nan_count + inf_count)

    stats_line = [
        f"    stats: total={total}",
        f"valid={valid_count}",
        f"missing={mask_count + nan_count + inf_count}",
        f"nan={nan_count}",
        f"inf={inf_count}",
    ]
    if sample_note:
        stats_line.append(sample_note)
    lines.append(" ".join(stats_line))

    if np.issubdtype(data_for_stats.dtype, np.number) and valid_count > 0:
        finite_data = data_for_stats[np.isfinite(data_for_stats)]
        if finite_data.size:
            percentile_vals = np.nanpercentile(finite_data, [5, 25, 50, 75, 95])
            lines.append(
                "    min={:.6g} max={:.6g} mean={:.6g} std={:.6g}".format(
                    np.nanmin(finite_data),
                    np.nanmax(finite_data),
                    np.nanmean(finite_data),
                    np.nanstd(finite_data),
                )
            )
            lines.append(
                "    p05={:.6g} p25={:.6g} median={:.6g} p75={:.6g} p95={:.6g}".format(
                    *percentile_vals
                )
            )

        if valid_min is not None or valid_max is not None:
            lower = -np.inf if valid_min is None else valid_min
            upper = np.inf if valid_max is None else valid_max
            out_of_range = int(np.sum((finite_data < lower) | (finite_data > upper)))
            lines.append(f"    out_of_range={out_of_range}")

    if looks_like_flag(name, da):
        if np.issubdtype(data_sample.dtype, np.number):
            flag_values = data_sample
            if mask_sample is not None:
                flag_values = flag_values[~mask_sample]
            if flag_values.size:
                unique_summary = summarize_unique(flag_values, max_unique)
                lines.append(f"    flag_values: {unique_summary}")
        if "flag_meanings" in da.attrs:
            lines.append(f"    flag_meanings={format_value(da.attrs['flag_meanings'])}")
        if "flag_masks" in da.attrs:
            lines.append(f"    flag_masks={format_value(da.attrs['flag_masks'])}")

    return lines


def format_dims(ds: xr.Dataset) -> str:
    return ", ".join(f"{name}={size}" for name, size in ds.dims.items())


def format_var_list(names: Iterable[str]) -> str:
    return ", ".join(names)


def report_dataset(
    category: str,
    path: Path,
    max_elements: int,
    max_unique: int,
) -> List[str]:
    lines = []
    stats = path.stat()
    lines.append(f"Dataset: {category}")
    lines.append(f"  file: {path}")
    lines.append(f"  size: {human_bytes(stats.st_size)}")
    lines.append(
        f"  modified: {dt.datetime.fromtimestamp(stats.st_mtime).isoformat()}"
    )

    with xr.open_dataset(path, decode_times=True, mask_and_scale=True) as ds:
        lines.append(f"  dimensions: {format_dims(ds)}")
        lines.append(f"  coordinates: {format_var_list(ds.coords.keys())}")
        lines.append(f"  data variables: {len(ds.data_vars)}")

        attr_lines = describe_global_attrs(ds)
        if attr_lines:
            lines.append("  global attributes:")
            lines.extend(attr_lines)

        res_lines = infer_resolution_from_attrs(ds)
        if res_lines:
            lines.append("  spatial metadata:")
            lines.extend(res_lines)

        lat = find_coord(ds, LAT_NAMES)
        lon = find_coord(ds, LON_NAMES)
        time = find_coord(ds, TIME_NAMES)
        if lat is not None:
            lat_summary = summarize_axis(lat)
            if lat_summary:
                lines.append(
                    "  latitude: min={:.6g} max={:.6g} count={} dtype={} dims={} shape={}".format(
                        lat_summary["min"],
                        lat_summary["max"],
                        lat_summary["count"],
                        lat_summary["dtype"],
                        lat_summary["dims"],
                        lat_summary["shape"],
                    )
                )
                if lat_summary["spacing"]:
                    spacing = lat_summary["spacing"]
                    lines.append(
                        "    lat spacing: median={:.6g} min={:.6g} max={:.6g}".format(
                            spacing["median"], spacing["min"], spacing["max"]
                        )
                    )
        if lon is not None:
            lon_summary = summarize_axis(lon)
            if lon_summary:
                lines.append(
                    "  longitude: min={:.6g} max={:.6g} count={} dtype={} dims={} shape={}".format(
                        lon_summary["min"],
                        lon_summary["max"],
                        lon_summary["count"],
                        lon_summary["dtype"],
                        lon_summary["dims"],
                        lon_summary["shape"],
                    )
                )
                if lon_summary["spacing"]:
                    spacing = lon_summary["spacing"]
                    lines.append(
                        "    lon spacing: median={:.6g} min={:.6g} max={:.6g}".format(
                            spacing["median"], spacing["min"], spacing["max"]
                        )
                    )
        if time is not None:
            time_summary = summarize_axis(time)
            if time_summary:
                lines.append(
                    "  time: min={} max={} count={} dtype={} dims={} shape={}".format(
                        format_value(time_summary["min"]),
                        format_value(time_summary["max"]),
                        time_summary["count"],
                        time_summary["dtype"],
                        time_summary["dims"],
                        time_summary["shape"],
                    )
                )

        primary_vars = [name for name in ds.data_vars if looks_like_primary(name)]
        if primary_vars:
            lines.append(f"  primary-variable candidates: {format_var_list(primary_vars)}")

        lines.append("  variable details:")
        for name in ds.data_vars:
            lines.extend(analyze_variable(name, ds[name], max_elements, max_unique))

    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.base is None:
        base = DEFAULT_BASE
    else:
        base = args.base.expanduser().resolve()
    categories = args.categories or DEFAULT_CATEGORIES

    print("Data Examination Report")
    print(f"Generated: {dt.datetime.now().isoformat()}")
    print(f"Base directory: {base}")
    print()

    for category in categories:
        category_path = base / category
        if not category_path.exists():
            print(f"Dataset: {category}")
            print(f"  missing: {category_path} does not exist")
            print()
            continue

        files = sorted([p for p in category_path.iterdir() if p.is_file() and p.suffix == ".nc"])
        if not files:
            print(f"Dataset: {category}")
            print("  missing: no .nc files found")
            print()
            continue

        sample_file = pick_sample_file(files, args.pick)
        if sample_file is None:
            print(f"Dataset: {category}")
            print("  missing: no file chosen")
            print()
            continue

        report_lines = report_dataset(category, sample_file, args.max_elements, args.max_unique)
        print("\n".join(report_lines))
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
