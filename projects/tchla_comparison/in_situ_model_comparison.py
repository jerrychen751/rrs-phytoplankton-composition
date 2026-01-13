import sys
from pathlib import Path
import glob
import itertools

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

# Paths
SDP_RESULT_PATH = PROJECT_ROOT / "data" / "outputs" / "seabass_validation" / "sdp_results.nc"
HPLC_PATH = PROJECT_ROOT / "data" / "outputs" / "seabass_validation" / "in_situ_hplc" / "PACE-PAX_SHEARWATER_HPLC_pigments.csv"
PACE_L2_DIR = PROJECT_ROOT / "data" / "input" / "tchla_comparison" / "pace_bgc"
OUTPUT_DIR = Path(__file__).parent

def get_bounds(hplc_df: pd.DataFrame, buffer: float = 0.2):
    """Get spatial bounds from HPLC data."""
    lat_min = hplc_df["lat"].min() - buffer
    lat_max = hplc_df["lat"].max() + buffer
    lon_min = hplc_df["lon"].min() - buffer
    lon_max = hplc_df["lon"].max() + buffer
    return (lat_min, lat_max), (lon_min, lon_max)

def load_insitu_tchla(filepath: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Load in-situ Total Chlorophyll a."""
    print(f"Loading HPLC data from {filepath}")
    df = pd.read_csv(filepath)
    # Clean column names
    df.columns = [col.strip() for col in df.columns]
    
    # 'Tchl' is usually Total Chlorophyll a in SeaBASS/HPLC files
    if 'Tchl' not in df.columns:
        raise ValueError(f"'Tchl' column not found in {filepath}. Available columns: {df.columns}")
        
    vals = pd.to_numeric(df['Tchl'], errors='coerce').dropna().values
    return vals, df

def load_sdp_tchla(filepath: Path, lat_range, lon_range) -> np.ndarray:
    """Load SDP Model Total Chlorophyll a (t_chla) within bounds."""
    print(f"Loading SDP Model data from {filepath}")
    ds = xr.open_dataset(filepath)
    
    if 't_chla' not in ds:
        raise ValueError(f"'t_chla' variable not found in {filepath}")
    
    # Handle slice direction for latitude
    lat_slice = slice(lat_range[1], lat_range[0]) if ds.lat[0] > ds.lat[-1] else slice(lat_range[0], lat_range[1])
    
    subset = ds['t_chla'].sel(
        lat=lat_slice,
        lon=slice(lon_range[0], lon_range[1])
    )
    
    vals = subset.values.flatten()
    vals = vals[np.isfinite(vals)]
    return vals

def load_pace_standard_tchla(directory: Path, lat_range, lon_range) -> np.ndarray:
    """Load PACE Standard Chlorophyll a (chlor_a) from L2 files within bounds."""
    print(f"Loading PACE L2 data from {directory}")
    files = sorted(list(directory.glob("*.nc")))
    if not files:
        print("No PACE L2 .nc files found.")
        return np.array([])
    
    all_vals = []
    
    for f in files:
        try:
            # L2 files typically have groups
            # chlor_a is in 'geophysical_data'
            # lat/lon is in 'navigation_data'
            with xr.open_dataset(f, group='geophysical_data') as ds_geo, \
                 xr.open_dataset(f, group='navigation_data') as ds_nav:
                
                if 'chlor_a' not in ds_geo:
                    continue
                
                # Load data into memory to mask
                chla = ds_geo['chlor_a'].values
                lat = ds_nav['latitude'].values
                lon = ds_nav['longitude'].values
                
                # Create mask for region
                mask = (
                    (lat >= lat_range[0]) & (lat <= lat_range[1]) &
                    (lon >= lon_range[0]) & (lon <= lon_range[1])
                )
                
                # Apply mask
                valid_pixels = chla[mask]
                
                # Filter NaNs/Fill values
                valid_pixels = valid_pixels[np.isfinite(valid_pixels)]
                # Filter outliers: nonnegative and <= 100 mg/m^3
                valid_pixels = valid_pixels[(valid_pixels >= 0) & (valid_pixels <= 100)]
                
                if len(valid_pixels) > 0:
                    all_vals.append(valid_pixels)
                    
        except Exception as e:
            print(f"Error reading {f.name}: {e}")
            continue
            
    if all_vals:
        return np.concatenate(all_vals)
    else:
        return np.array([])

def print_stats(data_map: dict):
    """Print summary statistics."""
    stats = []
    for name, vals in data_map.items():
        stats.append({
            "Source": name,
            "Count": len(vals),
            "Mean": np.mean(vals) if len(vals) > 0 else np.nan,
            "Median": np.median(vals) if len(vals) > 0 else np.nan,
            "Std": np.std(vals) if len(vals) > 0 else np.nan,
            "Min": np.min(vals) if len(vals) > 0 else np.nan,
            "Max": np.max(vals) if len(vals) > 0 else np.nan
        })
    
    df = pd.DataFrame(stats)
    print("\nTotal Chlorophyll-a Statistics:")
    print(df.round(3).to_string(index=False))
    
    # Save stats image
    fig, ax = plt.subplots(figsize=(10, 2 + len(df)*0.5))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=df.round(3).values.tolist(), colLabels=df.columns.tolist(), loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title("Total Chlorophyll-a Statistics", fontweight='bold')
    plt.savefig(OUTPUT_DIR / "tchla_statistics.png", bbox_inches='tight', dpi=300)
    plt.close()

def plot_distributions(data_map: dict):
    """Plot boxplots of distributions."""
    data = [vals for vals in data_map.values() if len(vals) > 0]
    labels = [name for name, vals in data_map.items() if len(vals) > 0]
    
    if not data:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_title("Total Chlorophyll-a Distributions")
    ax.set_ylabel("Concentration (mg/m^3)")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / "tchla_distributions.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved distribution plot to {save_path}")
    plt.close()

def get_quantiles(vals, n_points=100):
    if len(vals) == 0:
        return np.full(n_points, np.nan)
    return np.percentile(vals, np.linspace(0, 100, n_points))

def plot_qq(data_map: dict):
    """Plot Q-Q comparisons for all combinations of sources."""
    sources = list(data_map.keys())
    pairs = list(itertools.combinations(sources, 2))
    
    if not pairs:
        return

    fig, axes = plt.subplots(1, len(pairs), figsize=(5 * len(pairs), 5), squeeze=False)
    axes = axes.flatten()
    
    for i, (name1, name2) in enumerate(pairs):
        ax = axes[i]
        vals1 = data_map[name1]
        vals2 = data_map[name2]
        
        q1 = get_quantiles(vals1)
        q2 = get_quantiles(vals2)
        
        if np.isnan(q1).all() or np.isnan(q2).all():
            continue
            
        ax.scatter(q1, q2, alpha=0.7, edgecolors='none')
        
        # 1:1 line
        # Set limits to focus on the bulk of the data (e.g., up to 5 mg/m^3)
        # or dynamic based on the non-outlier dataset (usually In-situ or SDP)
        plot_max = 5.0
        
        ax.plot([0, plot_max], [0, plot_max], 'k--', label="1:1")
        
        ax.set_xlim(0, plot_max)
        ax.set_ylim(0, plot_max)
        
        ax.set_xlabel(f"{name1} (mg/m^3)")
        ax.set_ylabel(f"{name2} (mg/m^3)")
        ax.set_title(f"{name2} vs {name1}")
        ax.grid(True, linestyle=':', alpha=0.6)
        
    plt.tight_layout()
    save_path = OUTPUT_DIR / "tchla_qq_plots.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved Q-Q plots to {save_path}")
    plt.close()

def main():
    # 1. Load In-situ to get bounds
    try:
        hplc_vals, hplc_df = load_insitu_tchla(HPLC_PATH)
    except Exception as e:
        print(f"Failed to load HPLC data: {e}")
        return

    (lat_min, lat_max), (lon_min, lon_max) = get_bounds(hplc_df)
    print(f"Region: Lat [{lat_min:.2f}, {lat_max:.2f}], Lon [{lon_min:.2f}, {lon_max:.2f}]")

    # 2. Load SDP Model Data
    try:
        sdp_vals = load_sdp_tchla(SDP_RESULT_PATH, (lat_min, lat_max), (lon_min, lon_max))
    except Exception as e:
        print(f"Failed to load SDP data: {e}")
        sdp_vals = np.array([])

    # 3. Load PACE Standard Data
    try:
        pace_vals = load_pace_standard_tchla(PACE_L2_DIR, (lat_min, lat_max), (lon_min, lon_max))
    except Exception as e:
        print(f"Failed to load PACE L2 data: {e}")
        pace_vals = np.array([])

    # 4. Compare
    data_map = {
        "In-situ HPLC": hplc_vals,
        "SDP Model": sdp_vals,
        "PACE Standard (NASA)": pace_vals
    }
    
    # Filter out empty
    data_map = {k: v for k, v in data_map.items() if len(v) > 0}
    
    if not data_map:
        print("No data available for comparison.")
        return

    print_stats(data_map)
    plot_distributions(data_map)
    plot_qq(data_map)

if __name__ == "__main__":
    main()
