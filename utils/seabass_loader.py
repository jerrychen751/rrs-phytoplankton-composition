"""Utilities for loading SeaBASS format files."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

def parse_seabass_header(filepath: Path) -> Dict[str, str]:
    """
    Parse SeaBASS file header to extract metadata.
    
    Returns dictionary with header fields (without leading '/').
    """
    header = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Remove quotes if present (some SeaBASS files have quoted headers)
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            
            # Stop at end of header
            if line == '/end_header':
                break
            
            # Skip comment lines
            if line.startswith('!'):
                continue
            
            # Parse header fields
            if line.startswith('/'):
                # Remove leading '/' and split on '='
                if '=' in line:
                    key, value = line[1:].split('=', 1)
                    header[key] = value
    
    return header


def load_seabass_file(filepath: Path, missing_value: Optional[float] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Load a SeaBASS format file.
    
    Parameters:
    -----------
    filepath : Path
        Path to SeaBASS file
    missing_value : float, optional
        Value to use for missing data (will be replaced with NaN)
        If None, will try to read from header '/missing' field
    
    Returns:
    --------
    df : pd.DataFrame
        Data as pandas DataFrame
    header : dict
        Header metadata
    """
    # Parse header
    header = parse_seabass_header(filepath)
    
    # Get delimiter
    delimiter = header.get('delimiter', 'comma')
    if delimiter == 'comma':
        sep = ','
    elif delimiter == 'tab':
        sep = '\t'
    else:
        sep = delimiter
    
    # Get missing value
    if missing_value is None:
        missing_str = header.get('missing', '-9999')
        try:
            missing_value = float(missing_str)
        except ValueError:
            missing_value = -9999
    
    # Get field names
    fields_str = header.get('fields', '')
    if fields_str:
        field_names = [f.strip() for f in fields_str.split(',')]
    else:
        field_names = None
    
    # Read data (skip header lines)
    with open(filepath, 'r', encoding='utf-8') as f:
        # Find where data starts
        data_start = False
        skip_lines = 0
        for line in f:
            skip_lines += 1
            line_stripped = line.strip()
            # Handle both quoted and unquoted end_header
            if line_stripped == '/end_header' or line_stripped == '"/end_header"':
                data_start = True
                break
        
        if not data_start:
            raise ValueError("Could not find /end_header in file")
    
    # Read data
    # Convert missing values to strings for pandas
    na_values_list = [str(missing_value), '-9999', '-8888']
    df = pd.read_csv(
        filepath,
        sep=sep,
        skiprows=skip_lines,
        names=field_names,
        na_values=na_values_list,
        encoding='utf-8'
    )
    
    return df, header


def load_hplc_data(filepath: Path) -> pd.DataFrame:
    """
    Load HPLC pigment data from SeaBASS file.
    
    Returns DataFrame with pigment concentrations and metadata.
    """
    df, header = load_seabass_file(filepath)
    
    # Convert date/time to datetime if present
    if 'date' in df.columns and 'time' in df.columns:
        try:
            df['datetime'] = pd.to_datetime(
                df['date'].astype(str) + ' ' + df['time'].astype(str),
                format='%Y%m%d %H:%M:%S',
                errors='coerce'
            )
        except:
            pass
    
    return df


def extract_pigment_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract pigment concentration columns from HPLC DataFrame.
    
    Returns DataFrame with only pigment columns.
    """
    # Standard pigment names (matching model output)
    pigment_names = [
        'Tot_Chl_a', 'Tchla',  # Total chlorophyll a
        'Zea',                 # Zeaxanthin
        'DV_Chl_a', 'DVchla',  # Divinyl chlorophyll a
        'But-fuco', 'ButFuco', # 19'-butanoyloxyfucoxanthin
        'Hex-fuco', 'HexFuco', # 19'-hexanoyloxyfucoxanthin
        'Allo',                # Alloxanthin
        'MV_Chl_b', 'MVchlb',  # Monovinyl chlorophyll b
        'Neo',                 # Neoxanthin
        'Viola',               # Violaxanthin
        'Fuco',                # Fucoxanthin
        'Chl_c1c2', 'Chlc12',  # Chlorophyll c1+c2
        'Chl_c3', 'Chlc3',     # Chlorophyll c3
        'Perid',               # Peridinin
        'Tchl'                 # Total chlorophyll
    ]
    
    # Find matching columns
    pigment_cols = []
    for col in df.columns:
        if any(pig in col for pig in pigment_names):
            pigment_cols.append(col)
    
    return df[pigment_cols + ['station', 'date', 'time', 'lon', 'lat']].copy()


def match_stations_to_pace(
    hplc_df: pd.DataFrame,
    pace_lons: np.ndarray,
    pace_lats: np.ndarray,
    max_distance_km: float = 5.0
) -> Dict[int, Tuple[int, int]]:
    """
    Match HPLC stations to nearest PACE pixel.

    Notes:
    - This matcher assumes a mapped/gridded product with 1-D longitude and latitude
      coordinate vectors (PACE L3m-style).
    - It is not compatible with PACE OCI L2 swath products where lat/lon are 2-D.
      For L2 in-situ validation matchups, use `scripts/data/download_pace_rrs.py`.
    
    Parameters:
    -----------
    hplc_df : pd.DataFrame
        HPLC data with 'lon', 'lat', 'station' columns
    pace_lons : np.ndarray
        PACE longitude grid
    pace_lats : np.ndarray
        PACE latitude grid
    max_distance_km : float
        Maximum distance for matching (km)
    
    Returns:
    --------
    matches : dict
        Dictionary mapping station number to (lon_idx, lat_idx) in PACE grid
    """
    try:
        from geopy.distance import geodesic
    except ImportError:
        raise ImportError("geopy is required for match_stations_to_pace. Install with: pip install geopy")
    
    matches = {}
    
    for station in hplc_df['station'].unique():
        station_data = hplc_df[hplc_df['station'] == station].iloc[0]
        in_situ_lat = station_data['lat']
        in_situ_lon = station_data['lon']
        
        # Find nearest PACE pixel
        min_dist = np.inf
        best_idx = None
        
        for i in range(len(pace_lons)):
            for j in range(len(pace_lats)):
                pace_lat = pace_lats[j]
                pace_lon = pace_lons[i]
                
                dist = geodesic(
                    (in_situ_lat, in_situ_lon),
                    (pace_lat, pace_lon)
                ).kilometers
                
                if dist < min_dist:
                    min_dist = dist
                    best_idx = (i, j)
        
        if min_dist <= max_distance_km:
            matches[station] = best_idx
    
    return matches
