#!/usr/bin/env python3
"""
Inspect a local PACE OCI L2 netCDF file to understand its group/variable layout.

This helper is intentionally lightweight and is meant to answer:
- Where are `Rrs`, `l2_flags`, `latitude`, `longitude`, and scanline `time`?
- What are the dimension names and shapes?
- What wavelength coordinate exists for the Rrs spectral dimension?

Example:
  python scripts/data/inspect_pace_l2_rrs.py /path/to/PACE_OCI....L2.OC_AOP....nc
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr


def _print_var(ds: xr.Dataset, name: str) -> None:
    if name not in ds:
        print(f"  - {name}: (missing)")
        return
    da = ds[name]
    print(f"  - {name}: dims={da.dims} shape={da.shape} dtype={da.dtype}")
    for attr in ("long_name", "standard_name", "units", "_FillValue", "flag_meanings", "flag_masks"):
        if attr in da.attrs:
            val = da.attrs[attr]
            if isinstance(val, (list, tuple, np.ndarray)) and len(val) > 20:
                print(f"      {attr}: <{type(val).__name__} len={len(val)}>")
            else:
                print(f"      {attr}: {val}")


def _open_group(path: Path, group: str) -> Optional[xr.Dataset]:
    try:
        return xr.open_dataset(path, group=group, mask_and_scale=True, decode_cf=True, decode_timedelta=False)
    except Exception as exc:
        print(f"Could not open group '{group}': {exc}")
        return None


def inspect(path: Path) -> None:
    path = path.expanduser().resolve()
    print(f"File: {path}")

    try:
        groups = xr.open_groups(path, decode_timedelta=False)
    except Exception as exc:
        print(f"xarray.open_groups failed: {exc}")
        groups = {}

    if groups:
        print("Groups:")
        for key in sorted(groups.keys()):
            print(f"  - {key}")
        print("")

    geophys = _open_group(path, "geophysical_data")
    if geophys is not None:
        print("geophysical_data:")
        _print_var(geophys, "Rrs")
        _print_var(geophys, "l2_flags")
        # Try common wavelength coordinate names.
        for cand in ("wavelength_3d", "wavelength"):
            _print_var(geophys, cand)
        print("")
        geophys.close()

    nav = _open_group(path, "navigation_data")
    if nav is not None:
        print("navigation_data:")
        _print_var(nav, "latitude")
        _print_var(nav, "longitude")
        print("")
        nav.close()

    scan = _open_group(path, "scan_line_attributes")
    if scan is not None:
        print("scan_line_attributes:")
        _print_var(scan, "time")
        _print_var(scan, "year")
        _print_var(scan, "day")
        _print_var(scan, "msec")
        print("")
        scan.close()

    band = _open_group(path, "sensor_band_parameters")
    if band is not None:
        print("sensor_band_parameters:")
        for cand in ("wavelength_3d", "wavelength"):
            _print_var(band, cand)
        print("")
        band.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=Path, help="Path to a local PACE OCI L2 netCDF file")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inspect(args.path)


if __name__ == "__main__":
    main()
