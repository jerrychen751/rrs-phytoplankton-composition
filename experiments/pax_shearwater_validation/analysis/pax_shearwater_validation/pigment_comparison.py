import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SDP_RESULT_PATH = PROJECT_ROOT / "experiments" / "pax_shearwater_validation" / "outputs" / "sdp_results.nc"
HPLC_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "pax_shearwater_validation"
    / "inputs"
    / "validation"
    / "hplc"
    / "PACE-PAX_SHEARWATER_HPLC_pigments.csv"
)
OUTPUT_DIR = Path(__file__).parent

BIOMARKER_INFO = {
    "fuco": "Diatoms",
    "perid": "Dinoflagellates",
    "hexfuco": "Haptophytes",
    "mv_chlb": "Green Algae",
    "zea": "Cyanobacteria"
}

def get_regional_data(model_ds: xr.Dataset, hplc_df: pd.DataFrame, pigment_mapping: dict) -> dict:
    """
    Extract unpaired regional data for each pigment.
    Returns a dictionary: {pigment_name: {'model': np.array, 'hplc': np.array}}
    """
    data_dict = {}
    
    # Define region based on HPLC data bounds (plus buffer)
    lat_min, lat_max = hplc_df["lat"].min() - 0.2, hplc_df["lat"].max() + 0.2
    lon_min, lon_max = hplc_df["lon"].min() - 0.2, hplc_df["lon"].max() + 0.2
    
    print(f"Filtering model data to region: Lat [{lat_min:.2f}, {lat_max:.2f}], Lon [{lon_min:.2f}, {lon_max:.2f}]")
    
    # Subset model dataset to the region
    # Handle slice direction (lat might be descending)
    if model_ds.lat[0] > model_ds.lat[-1]:
        model_subset = model_ds.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    else:
        model_subset = model_ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))

    for ds_name, col_name in pigment_mapping.items():
        if ds_name not in model_ds or col_name not in hplc_df.columns:
            continue
            
        # Extract all valid model values in the region
        model_vals = model_subset[ds_name].values.flatten()
        model_vals = model_vals[np.isfinite(model_vals)]
        
        # Extract all valid in-situ values
        hplc_vals = pd.to_numeric(hplc_df[col_name], errors='coerce').dropna().values
        
        if len(model_vals) > 0 or len(hplc_vals) > 0:
            data_dict[ds_name] = {
                'model': model_vals,
                'hplc': hplc_vals,
                'label': col_name
            }
            
    return data_dict

def print_stats(data_dict: dict):
    stats_records = []
    for pigment, data in data_dict.items():
        model_vals = data['model']
        hplc_vals = data['hplc']
        
        stats_records.append({
            "pigment": pigment,
            "source": "Model",
            "count": len(model_vals),
            "mean": np.mean(model_vals) if len(model_vals) > 0 else np.nan,
            "median": np.median(model_vals) if len(model_vals) > 0 else np.nan,
            "std": np.std(model_vals) if len(model_vals) > 0 else np.nan
        })
        stats_records.append({
            "pigment": pigment,
            "source": "HPLC",
            "count": len(hplc_vals),
            "mean": np.mean(hplc_vals) if len(hplc_vals) > 0 else np.nan,
            "median": np.median(hplc_vals) if len(hplc_vals) > 0 else np.nan,
            "std": np.std(hplc_vals) if len(hplc_vals) > 0 else np.nan
        })
        
    stats_df = pd.DataFrame(stats_records)
    if not stats_df.empty:
        pivot_df = stats_df.pivot(index="pigment", columns="source", values=["mean", "median", "std", "count"])
        print("\nRegional Statistics Summary:")
        print(pivot_df.round(3))

        # Save as image
        fig, ax = plt.subplots(figsize=(14, len(pivot_df) * 0.4 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        display_df = pivot_df.round(3)
        # Flatten column names for the table header
        col_labels = [f"{c[0]}\n{c[1]}" for c in display_df.columns]
        
        table = ax.table(cellText=display_df.values,
                         rowLabels=display_df.index,
                         colLabels=col_labels,
                         loc='center',
                         cellLoc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        ax.set_title("Regional Statistics Summary", fontweight='bold')
        
        save_path = OUTPUT_DIR / "statistics.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved statistics table to {save_path}")
        plt.close()

def plot_distributions(data_dict: dict):
    """Boxplots of distributions"""
    plot_data = []
    labels = []
    
    for pigment, data in data_dict.items():
        if len(data['model']) > 0:
            plot_data.append(data['model'])
            labels.append(f"{pigment}\n(Model)")
        if len(data['hplc']) > 0:
            plot_data.append(data['hplc'])
            labels.append(f"{pigment}\n(HPLC)")
            
    if plot_data:
        fig, ax = plt.subplots(figsize=(max(10, len(data_dict) * 1.5), 6))
        ax.boxplot(plot_data, tick_labels=labels, showfliers=False)
        ax.set_title("Regional Pigment Distributions (Model vs HPLC)")
        ax.set_ylabel("Concentration (mg/m^3)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        save_path = OUTPUT_DIR / "distributions_boxplot.png"
        plt.savefig(save_path, dpi=300)
        print(f"Saved figure to {save_path}")
        plt.close()

def get_quantiles(vals, n_points=100):
    if len(vals) == 0:
        return np.full(n_points, np.nan)
    return np.percentile(vals, np.linspace(0, 100, n_points))

def plot_qq_comparison(data_dict: dict):
    """
    Q-Q Plot: Compares the quantiles of Model vs HPLC.
    This serves as a 'Parity Plot' for unpaired data distributions.
    """
    pigments = sorted(data_dict.keys())
    ncols = 4
    nrows = int(np.ceil(len(pigments) / ncols)) or 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows), squeeze=False)
    
    for idx, pigment in enumerate(pigments):
        ax = axes[idx // ncols][idx % ncols]
        data = data_dict[pigment]
        
        model_q = get_quantiles(data['model'])
        hplc_q = get_quantiles(data['hplc'])
        
        if np.isnan(model_q).all() or np.isnan(hplc_q).all():
            ax.axis("off")
            continue
            
        ax.scatter(hplc_q, model_q, s=30, alpha=0.8, edgecolors="none")
        
        # 1:1 line
        min_val = min(np.nanmin(hplc_q), np.nanmin(model_q))
        max_val = max(np.nanmax(hplc_q), np.nanmax(model_q))
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)
        
        title_text = f"{pigment}"
        title_color = "black"
        if pigment in BIOMARKER_INFO:
            title_text += f"\n[{BIOMARKER_INFO[pigment]}]"
            title_color = "red"
            
        ax.set_title(title_text, color=title_color)
        ax.set_xlabel("HPLC Concentration (mg/m^3)")
        ax.set_ylabel("Model Concentration (mg/m^3)")

    for idx in range(len(pigments), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
        
    fig.suptitle("Regional Distribution Parity (Q-Q Plots)", fontsize=16, y=1.02)
    fig.tight_layout()
    
    save_path = OUTPUT_DIR / "qq_parity.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved figure to {save_path}")
    plt.close()

def plot_quantile_difference(data_dict: dict):
    """
    Quantile Difference Plot: Difference between Model and HPLC quantiles vs Mean of quantiles.
    This serves as a 'Bland-Altman' for unpaired data distributions.
    """
    pigments = sorted(data_dict.keys())
    ncols = 4
    nrows = int(np.ceil(len(pigments) / ncols)) or 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows), squeeze=False)
    
    for idx, pigment in enumerate(pigments):
        ax = axes[idx // ncols][idx % ncols]
        data = data_dict[pigment]
        
        model_q = get_quantiles(data['model'])
        hplc_q = get_quantiles(data['hplc'])
        
        if np.isnan(model_q).all() or np.isnan(hplc_q).all():
            ax.axis("off")
            continue
            
        mean_val = (model_q + hplc_q) / 2
        diff = model_q - hplc_q
        bias = np.mean(diff)
        std = np.std(diff)
        
        ax.scatter(mean_val, diff, alpha=0.8, s=30, edgecolors="none")
        ax.axhline(bias, color="red", linestyle="--", label="Bias")
        ax.axhline(bias + 1.96 * std, color="gray", linestyle=":", label="±1.96σ")
        ax.axhline(bias - 1.96 * std, color="gray", linestyle=":")
        
        title_text = f"{pigment}"
        title_color = "black"
        if pigment in BIOMARKER_INFO:
            title_text += f"\n[{BIOMARKER_INFO[pigment]}]"
            title_color = "red"
            
        ax.set_title(title_text, color=title_color)
        ax.set_xlabel("Mean Concentration (mg/m^3)")
        ax.set_ylabel("Model - HPLC Difference (mg/m^3)")

    for idx in range(len(pigments), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
        
    fig.suptitle("Bland-Altman Plots", fontsize=16, y=1.02)
    fig.tight_layout()
    
    save_path = OUTPUT_DIR / "bland_altman.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved figure to {save_path}")
    plt.close()

def main():
    print(f"Loading SDP results from {SDP_RESULT_PATH}")
    dataset = xr.open_dataset(SDP_RESULT_PATH)
    
    print(f"Loading HPLC data from {HPLC_PATH}")
    hplc_df = pd.read_csv(HPLC_PATH)
    hplc_df.columns = [col.strip() for col in hplc_df.columns]

    pigment_mapping = {
        "t_chla": "Tchl",
        "zea": "Zea",
        "dv_chla": "DV_Chl_a",
        "butfuco": "But-fuco",
        "hexfuco": "Hex-fuco",
        "allo": "Allo",
        "mv_chlb": "MV_Chl_b",
        "neo": "Neo",
        "viola": "Viola",
        "fuco": "Fuco",
        "chl_c1_c2": "Chl_c1c2",
        "chl_c3": "Chl_c3",
        "perid": "Perid",
    }

    # 1. Get Unpaired Regional Data
    data_dict = get_regional_data(dataset, hplc_df, pigment_mapping)
    
    if not data_dict:
        print("No valid data found for any pigment in the region.")
        return

    # 2. Print Statistics
    print_stats(data_dict)
    
    # 3. Plot Distributions (Boxplots)
    plot_distributions(data_dict)
    
    # 4. Plot Q-Q (Parity-like)
    plot_qq_comparison(data_dict)
    
    # 5. Plot Quantile Difference (Bland-Altman-like)
    plot_quantile_difference(data_dict)

if __name__ == "__main__":
    main()
