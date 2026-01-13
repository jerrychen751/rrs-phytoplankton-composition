"""Inspect an experiment's SDP output file for spatial coverage.

This is a small debugging/QA utility that loads the configured NetCDF output
and reports the lat/lon locations that have at least one finite value.

Usage:
  python experiments/seabass_validation/analysis/seabass_validation/check_sdp_coverage.py
  python experiments/seabass_validation/analysis/seabass_validation/check_sdp_coverage.py experiments/seabass_validation/config.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config_loader import (
    get_output_dir,
    get_output_filename,
    load_config_from_file,
)


def main() -> None:
    default_config = Path(__file__).resolve().parents[2] / "config.yaml"
    config_input = sys.argv[1] if len(sys.argv) > 1 else str(default_config)

    config_path = Path(config_input)
    config_file = config_path if config_path.is_absolute() else (PROJECT_ROOT / config_path)
    config_file = config_file.resolve()

    config = load_config_from_file(config_file)
    result_path = get_output_dir(config, PROJECT_ROOT, config_dir=config_file.parent) / get_output_filename(config)

    if not result_path.exists():
        print(f"Error: File not found at {result_path}")
        print(f"Config used: {config_input}")
        return

    print(f"Loading SDP results from {result_path}")
    ds = xr.open_dataset(result_path)

    print("Checking for valid data locations...")

    if "t_chla" in ds:
        valid_da = np.isfinite(ds["t_chla"])
    else:
        first_var = list(ds.data_vars)[0]
        valid_da = np.isfinite(ds[first_var])

    if "time" in valid_da.dims:
        valid_da = valid_da.any(dim="time")

    lats = ds["lat"].values
    lons = ds["lon"].values
    mask_values = valid_da.values

    dims = valid_da.dims
    if dims == ("lat", "lon"):
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    elif dims == ("lon", "lat"):
        lon_grid, lat_grid = np.meshgrid(lons, lats, indexing="ij")
        lat_grid = lat_grid.T
        lon_grid = lon_grid.T
    else:
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    valid_lats = lat_grid[mask_values]
    valid_lons = lon_grid[mask_values]

    if len(valid_lats) == 0:
        print("No valid data found in the entire dataset.")
        return

    print(f"\nFound {len(valid_lats)} locations with valid data:")
    print(f"{'Latitude':>10} | {'Longitude':>10}")
    print("-" * 25)

    count = 0
    for lat, lon in zip(valid_lats, valid_lons):
        print(f"{lat:10.4f} | {lon:10.4f}")
        count += 1
        if count >= 50:
            print(f"... and {len(valid_lats) - 50} more locations.")
            break

    print("\nValid Data Bounding Box:")
    print(f"Lat: {valid_lats.min():.4f} to {valid_lats.max():.4f}")
    print(f"Lon: {valid_lons.min():.4f} to {valid_lons.max():.4f}")


if __name__ == "__main__":
    main()

