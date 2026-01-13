import xarray as xr
import numpy as np
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from utils.config_loader import (
    load_config_from_file,
    get_output_dir,
    get_output_filename,
)

def main():
    config_input = (
        sys.argv[1] if len(sys.argv) > 1 else "config/seabass_validation.yaml"
    )

    config = load_config_from_file(Path(config_input))
    result_path = get_output_dir(config, PROJECT_ROOT) / get_output_filename(config)

    if not result_path.exists():
        print(f"Error: File not found at {result_path}")
        print(f"Config used: {config_input}")
        return

    print(f"Loading SDP results from {result_path}")
    ds = xr.open_dataset(result_path)
    
    # Check for any finite value across all data variables
    # We convert to an array (stacking variables) and check for finiteness
    print("Checking for valid data locations...")
    
    # Use xarray's isfinite to preserve DataArray structure (coords/dims)
    if 't_chla' in ds:
        # This returns a DataArray
        valid_da = np.isfinite(ds['t_chla'])
    else:
        first_var = list(ds.data_vars)[0]
        valid_da = np.isfinite(ds[first_var])
        
    # If it has time dimension, aggregate
    if 'time' in valid_da.dims:
        valid_da = valid_da.any(dim='time')
        
    # Now valid_da is a boolean DataArray (lat, lon)
    
    # Get indices where mask is True
    # We can use where(drop=True) to get only valid points, but that might return a sparse dataset
    # Let's just extract the coordinates using numpy for printing
    
    lats = ds['lat'].values
    lons = ds['lon'].values
    
    # Assuming 2D shape (lat, lon) matching the DataArray
    # We need to be careful if dimensions are (lat, lon) or (lon, lat)
    # valid_da.values will be a numpy array
    mask_values = valid_da.values
    
    # Create meshgrid to match the mask shape
    # xarray usually uses (lat, lon) order for 2D plots/arrays if those are the dims
    # Let's check dims order
    dims = valid_da.dims
    if dims == ('lat', 'lon'):
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')
    elif dims == ('lon', 'lat'):
        # This is less common for map data but possible
        lon_grid, lat_grid = np.meshgrid(lons, lats, indexing='ij')
        lat_grid = lat_grid.T
        lon_grid = lon_grid.T
    else:
        # Fallback to standard meshgrid assuming lat is axis 0, lon is axis 1
        lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

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
