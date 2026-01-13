import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap, BoundaryNorm
from typing import Union, List, Tuple, Optional
from models.sdp_pigments.core.prediction import run_sdp

def create_dataset(
    rrs_paths: Union[str, List[str]], 
    sal_paths: Union[str, List[str], None], 
    temp_paths: Union[str, List[str], None], 
    bbox: Tuple[float, float, float, float]
) -> Tuple[xr.DataArray, Optional[xr.DataArray], Optional[xr.DataArray]]:
    """
    Create xarray dataset from mapped (gridded) PACE Rrs, salinity, and temperature files.
    
    If multiple files provided, averages over time dimension. Interpolates sal/temp
    to match Rrs grid using nearest neighbor. Rrs interpolated to 400-700 nm at 1 nm.

    Notes:
    - This function expects mapped/gridded Rrs inputs with a root-level `Rrs` variable
      on a (`lat`, `lon`, `wavelength`) grid (PACE OCI L3m-style).
    - PACE OCI L2 swath products (e.g., `L2.OC_AOP`) store Rrs under netCDF groups
      and are not compatible with this loader.
    
    bbox format: (west, south, east, north).
    Returns: (rrs, sal, temp) DataArrays.
    """

    n = bbox[3]
    s = bbox[1]
    e = bbox[2]
    w = bbox[0]
    
    # creates a dataset of rrs values of the given file
    if isinstance(rrs_paths, str):
        rrs_data = xr.open_dataset(rrs_paths)
        if "Rrs" not in rrs_data:
            raise KeyError(
                "Expected a mapped/gridded PACE Rrs input with variable `Rrs` at the root group. "
                "If you are working with PACE OCI L2 swath products (e.g., `L2.OC_AOP`), "
                "use `scripts/data/download_pace_rrs.py` to build per-station matchup datasets."
            )
        rrs = rrs_data["Rrs"].sel({"lat": slice(n, s), "lon": slice(w, e)})
    elif isinstance(rrs_paths, list):
        # if given a list of files, create a date averaged dataset of Rrs values 
        rrs_data = xr.open_mfdataset(
            rrs_paths,
            combine="nested",
            concat_dim="date",
        )
        if "Rrs" not in rrs_data:
            raise KeyError(
                "Expected mapped/gridded PACE Rrs inputs with variable `Rrs` at the root group. "
                "If you are working with PACE OCI L2 swath products (e.g., `L2.OC_AOP`), "
                "use `scripts/data/download_pace_rrs.py` to build per-station matchup datasets."
            )
        rrs = rrs_data["Rrs"].sel({"lat": slice(n, s), "lon": slice(w, e)}).mean('date')
        rrs = rrs.compute()
    else:
        raise ValueError('rrs_paths must be a string or a list containing at least one filepath')
    
    # creates a dataset of sal and temp values of the given file
    if sal_paths is None:
        sal = None
    elif isinstance(sal_paths, str):
        sal = xr.open_dataset(sal_paths)
        sal = sal["smap_sss"].interp(longitude=rrs.lon, latitude=rrs.lat, method='nearest')
    elif isinstance(sal_paths, list):
        # if given a list of files, create a date averaged dataset of salinity values 
        sal = xr.open_mfdataset(
            sal_paths,
            combine="nested",
            concat_dim="date",
        )
        sal = sal["smap_sss"].interp(longitude=rrs.lon, latitude=rrs.lat, method='nearest').mean('date')
        sal = sal.compute()
    else:
        raise TypeError('sal_paths must be a string, list, or None')
    
    # creates a dataset of sal and temp values of the given file
    if temp_paths is None:
        temp = None
    elif isinstance(temp_paths, str):
        temp_ds = xr.open_dataset(temp_paths)
        if 'sst' not in temp_ds:
            raise KeyError("Temperature dataset must contain variable 'sst'")
        temp = temp_ds['sst']
        if 'time' in temp.dims:
            temp = temp.squeeze('time')
        temp = temp.interp(lon=rrs.lon, lat=rrs.lat, method='nearest')
    elif isinstance(temp_paths, list):
        temp_arrays: List[xr.DataArray] = []
        for temp_file in temp_paths:
            temp_ds = xr.open_dataset(temp_file)
            if 'sst' not in temp_ds:
                raise KeyError(f"Temperature dataset '{temp_file}' must contain variable 'sst'")
            temp_arr = temp_ds['sst']
            if 'time' in temp_arr.dims:
                temp_arr = temp_arr.squeeze('time')
            temp_arrays.append(temp_arr)
        if not temp_arrays:
            raise ValueError('temp_paths list must contain at least one filepath')
        temp = xr.concat(temp_arrays, dim='stack')
        temp = temp.interp(lon=rrs.lon, lat=rrs.lat, method='nearest').mean('stack')
        temp = temp.compute()
    else:
        raise TypeError('temp_paths must be a string, list, or None')
    
    rrs = rrs.interp(wavelength=np.arange(400,701,1))

    return rrs, sal, temp

def plot_pigments(data: xr.DataArray, lower_bound: float, upper_bound: float, label: str) -> None:
    """Plot pigment data on map with specified color scale bounds."""

    data.attrs["long_name"] = label

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, cmap.N))
    colors = np.vstack((np.array([1, 1, 1, 1]), colors)) 
    custom_cmap = ListedColormap(colors)
    norm = BoundaryNorm(list(np.linspace(lower_bound, upper_bound, cmap.N)), ncolors=custom_cmap.N) 

    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines(draw_labels={"left": "y", "bottom": "x"})
    data.plot(cmap=custom_cmap, ax=ax, norm=norm)
    ax.add_feature(cfeature.LAND, facecolor='white', zorder=1)
    plt.show()

def sdp_from_pace(
    pace_file: Union[str, List[str]], 
    sss: Union[str, List[str], None], 
    sst: Union[str, List[str], None],
    bbox: Optional[Tuple[float, float, float, float]],
    pigments: Optional[List[str]]
) -> xr.Dataset:
    """
    Apply SDP model to PACE data to predict pigment concentrations.
    
    Creates dataset from PACE files, applies run_sdp, then reshapes results to lat/lon grid.
    If pigments=None, predicts all 13 trained pigments.
    
    Returns xarray Dataset with predicted pigments on lat/lon grid.
    """
    
    rrs_interp, sss_interp, sst_interp = create_dataset(pace_file, sss, sst, bbox)

    print(rrs_interp)
    wl = np.arange(400,701,1)

    rrs_np = rrs_interp.to_numpy().reshape(-1, rrs_interp.wavelength.size)
    rrs_df = pd.DataFrame(rrs_np, columns=wl)
    
    # Handle optional sss and sst
    if sss_interp is None:
        raise ValueError("SSS (salinity) data is required for SDP model")
    if sst_interp is None:
        raise ValueError("SST (temperature) data is required for SDP model")
    
    sss_np = sss_interp.to_numpy().flatten()
    sst_np = sst_interp.to_numpy().flatten()

    sdp = run_sdp(rrs_df, wl, sst_np, sss_np, pigments=pigments)

    # Dynamically create dataset from all predicted pigments
    data_vars = {}
    for col in sdp.columns:
        # Convert column name to valid variable name (replace spaces and special chars)
        var_name = col.replace(' ', '_').replace('+', '_').replace('-', '_').lower()
        pigment_values = sdp[col].values.reshape(rrs_interp.lat.size, rrs_interp.lon.size)
        data_vars[var_name] = (['lat', 'lon'], pigment_values)

    pigments_dataset = xr.Dataset(
        data_vars,
        coords={
            'lat': rrs_interp.lat.values,
            'lon': rrs_interp.lon.values,
        },
    )
    
    return pigments_dataset
