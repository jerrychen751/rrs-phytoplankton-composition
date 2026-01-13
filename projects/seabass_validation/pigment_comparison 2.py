import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

SDP_RESULT_PATH = PROJECT_ROOT / "data" / "outputs" / "seabass_validation" / "sdp_results.nc"
HPLC_PATH = PROJECT_ROOT / "data" / "outputs" / "seabass_validation" / "in_situ_hplc" / "PACE-PAX_SHEARWATER_HPLC_pigments.csv"

GRID_CELL_DEG = 0.1
GRID_SNAP_FRACTION = 0.75
LAT_TOL_DEG = GRID_CELL_DEG * GRID_SNAP_FRACTION
LON_TOL_DEG = GRID_CELL_DEG * GRID_SNAP_FRACTION

def _grid_spacing(values: np.ndarray) -> float:
    diffs = np.diff(np.sort(np.unique(values)))
    diffs = diffs[np.abs(diffs) > 0]
    return float(np.nanmean(np.abs(diffs))) if diffs.size else np.nan

def snap_hplc_to_grid(df: pd.DataFrame, lat_grid: np.ndarray, lon_grid: np.ndarray, pigment_mapping: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Snap each HPLC record to the nearest model grid cell and aggregate repeats."""

    records: list[dict] = []
    pigment_columns = [col for col in pigment_mapping.values() if col in df.columns]

    for _, row in df.iterrows():
        try:
            lat = float(row["lat"])
            lon = float(row["lon"])
        except (TypeError, ValueError, KeyError):
            continue

        lat_idx = int(np.abs(lat_grid - lat).argmin())
        lon_idx = int(np.abs(lon_grid - lon).argmin())
        grid_lat = float(lat_grid[lat_idx])
        grid_lon = float(lon_grid[lon_idx])

        record = {
            "station": row.get("station"),
            "timestamp": row.get("timestamp"),
            "lat": lat,
            "lon": lon,
            "grid_lat": grid_lat,
            "grid_lon": grid_lon,
            "grid_lat_delta": lat - grid_lat,
            "grid_lon_delta": lon - grid_lon,
        }
        for column in pigment_columns:
            record[column] = pd.to_numeric(row[column], errors="coerce")
        records.append(record)

    snapped_df = pd.DataFrame.from_records(records)
    if snapped_df.empty:
        return snapped_df, snapped_df

    grouped = snapped_df.groupby(["grid_lat", "grid_lon"])
    value_columns = [
        *pigment_columns,
        "lat",
        "lon",
        "grid_lat_delta",
        "grid_lon_delta",
    ]
    aggregated = (
        grouped[value_columns]
        .mean()
        .reset_index()
        .rename(
            columns={
                "lat": "mean_sample_lat",
                "lon": "mean_sample_lon",
                "grid_lat_delta": "mean_lat_offset",
                "grid_lon_delta": "mean_lon_offset",
            }
        )
    )
    aggregated["sample_count"] = grouped.size().values
    stations = (
        grouped["station"]
        .agg(lambda vals: sorted({str(v) for v in vals.dropna()}))
        .reset_index()
        .rename(columns={"station": "stations"})
    )
    time_bounds = (
        grouped["timestamp"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "timestamp_start", "max": "timestamp_end"})
    )
    aggregated = aggregated.merge(stations, on=["grid_lat", "grid_lon"], how="left")
    aggregated = aggregated.merge(time_bounds, on=["grid_lat", "grid_lon"], how="left")
    return snapped_df, aggregated

def collect_matches(ds: xr.Dataset, dataframe: pd.DataFrame, pigment_mapping: dict) -> pd.DataFrame:
    records: list[dict] = []
    lat_values = ds["lat"].values
    lon_values = ds["lon"].values

    for _, row in dataframe.iterrows():
        target_lat = row.get("grid_lat", row.get("lat"))
        target_lon = row.get("grid_lon", row.get("lon"))
        if target_lat is None or target_lon is None:
            continue
        try:
            target_lat = float(target_lat)
            target_lon = float(target_lon)
        except (TypeError, ValueError):
            continue

        lat_idx = int(np.abs(lat_values - target_lat).argmin())
        lon_idx = int(np.abs(lon_values - target_lon).argmin())
        nearest_lat = float(lat_values[lat_idx])
        nearest_lon = float(lon_values[lon_idx])

        lat_delta = float(
            row.get(
                "mean_lat_offset",
                row.get("grid_lat_delta", row.get("lat_delta", target_lat - nearest_lat)),
            )
        )
        lon_delta = float(
            row.get(
                "mean_lon_offset",
                row.get("grid_lon_delta", row.get("lon_delta", target_lon - nearest_lon)),
            )
        )

        for ds_name, column in pigment_mapping.items():
            if column not in dataframe.columns:
                continue
            in_situ_value = row[column]
            if pd.isna(in_situ_value):
                continue

            predicted_value = float(ds[ds_name].isel(lat=lat_idx, lon=lon_idx).values)
            if not np.isfinite(predicted_value):
                continue

            record = {
                "grid_lat": target_lat,
                "grid_lon": target_lon,
                "mean_sample_lat": float(row.get("mean_sample_lat", row.get("lat", target_lat))),
                "mean_sample_lon": float(row.get("mean_sample_lon", row.get("lon", target_lon))),
                "lat_delta": abs(lat_delta),
                "lon_delta": abs(lon_delta),
                "pigment": ds_name,
                "pigment_label": column,
                "in_situ": float(in_situ_value),
                "predicted": predicted_value,
                "sample_count": int(row.get("sample_count", 1)),
                "stations": row.get("stations", row.get("station")),
                "timestamp_start": row.get("timestamp_start", row.get("timestamp")),
                "timestamp_end": row.get("timestamp_end", row.get("timestamp")),
            }
            records.append(record)

    columns = [
        "grid_lat",
        "grid_lon",
        "mean_sample_lat",
        "mean_sample_lon",
        "lat_delta",
        "lon_delta",
        "pigment",
        "pigment_label",
        "in_situ",
        "predicted",
        "sample_count",
        "stations",
        "timestamp_start",
        "timestamp_end",
    ]
    return pd.DataFrame.from_records(records, columns=columns)

def summarize_pigment(group: pd.DataFrame) -> pd.Series:
    observed = group["in_situ"].values
    predicted = group["predicted"].values
    diff = predicted - observed

    mae = np.mean(np.abs(diff))
    rmse = np.sqrt(np.mean(diff**2))
    bias = np.mean(diff)
    corr = pearsonr(observed, predicted)[0] if len(group) > 1 else np.nan

    return pd.Series(
        {
            "n": len(group),
            "bias": bias,
            "mae": mae,
            "rmse": rmse,
            "pearson_r": corr,
        }
    )

def plot_parity(data: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    if data.empty:
        print("No validated pigment pairs available for parity plots.")
        return

    pigments = sorted(data["pigment_label"].unique())
    ncols = 4
    nrows = int(np.ceil(len(pigments) / ncols)) or 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows), squeeze=False)

    for idx, pigment in enumerate(pigments):
        ax = axes[idx // ncols][idx % ncols]
        subset = data[data["pigment_label"] == pigment]
        if subset.empty:
            ax.axis("off")
            continue

        ax.scatter(
            subset["in_situ"],
            subset["predicted"],
            s=30,
            alpha=0.8,
            edgecolors="none",
        )
        min_val = min(subset["in_situ"].min(), subset["predicted"].min())
        max_val = max(subset["in_situ"].max(), subset["predicted"].max())
        ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black", linewidth=1)

        stats_row = summary_df[summary_df["pigment_label"] == pigment]
        rmse = stats_row["rmse"].iloc[0] if not stats_row.empty else np.nan
        corr = stats_row["pearson_r"].iloc[0] if not stats_row.empty else np.nan
        ax.set_title(f"{pigment}\nRMSE={rmse:.3f}, r={corr:.2f}")
        ax.set_xlabel("In situ (µg/L)")
        ax.set_ylabel("SDP (µg/L)")

    for idx in range(len(pigments), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle("Parity plots by pigment", fontsize=16, y=1.02)
    fig.tight_layout()
    plt.show()

def plot_bland_altman(data: pd.DataFrame) -> None:
    if data.empty:
        print("No validated pigment pairs available for Bland-Altman plots.")
        return

    pigments = sorted(data["pigment_label"].unique())
    ncols = 4
    nrows = int(np.ceil(len(pigments) / ncols)) or 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4 * nrows), squeeze=False)

    for idx, pigment in enumerate(pigments):
        ax = axes[idx // ncols][idx % ncols]
        subset = data[data["pigment_label"] == pigment]
        if subset.empty:
            ax.axis("off")
            continue

        mean_val = (subset["predicted"] + subset["in_situ"]) / 2
        diff = subset["predicted"] - subset["in_situ"]
        bias = diff.mean()
        std = diff.std()

        ax.scatter(mean_val, diff, alpha=0.8, s=30, edgecolors="none")
        ax.axhline(bias, color="red", linestyle="--", label="Bias")
        ax.axhline(bias + 1.96 * std, color="gray", linestyle=":", label="±1.96σ")
        ax.axhline(bias - 1.96 * std, color="gray", linestyle=":")
        ax.set_title(pigment)
        ax.set_xlabel("Mean (µg/L)")
        ax.set_ylabel("SDP - In situ (µg/L)")

    for idx in range(len(pigments), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Bland-Altman plots", fontsize=16, y=1.02)
    fig.tight_layout()
    plt.show()

def compare_regional_distributions(ds: xr.Dataset, df: pd.DataFrame, pigment_mapping: dict) -> None:
    """
    Compare the overall distribution of pigments in the region (Model vs In-Situ),
    ignoring precise spatial matching.
    """
    print("\n" + "="*50)
    print("REGIONAL DISTRIBUTION COMPARISON (Unpaired)")
    print("="*50)
    
    stats_records = []
    
    # Prepare data for plotting
    plot_data = []
    labels = []
    
    for ds_name, col_name in pigment_mapping.items():
        if ds_name not in ds or col_name not in df.columns:
            continue
            
        # Extract all valid model values in the region
        model_vals = ds[ds_name].values.flatten()
        model_vals = model_vals[np.isfinite(model_vals)]
        
        # Extract all valid in-situ values
        obs_vals = pd.to_numeric(df[col_name], errors='coerce').dropna().values
        
        if len(model_vals) == 0 and len(obs_vals) == 0:
            continue
            
        # Calculate statistics
        stats_records.append({
            "pigment": ds_name,
            "source": "Model (SDP)",
            "count": len(model_vals),
            "mean": np.mean(model_vals) if len(model_vals) > 0 else np.nan,
            "median": np.median(model_vals) if len(model_vals) > 0 else np.nan,
            "std": np.std(model_vals) if len(model_vals) > 0 else np.nan,
            "min": np.min(model_vals) if len(model_vals) > 0 else np.nan,
            "max": np.max(model_vals) if len(model_vals) > 0 else np.nan
        })
        
        stats_records.append({
            "pigment": ds_name,
            "source": "In-Situ (HPLC)",
            "count": len(obs_vals),
            "mean": np.mean(obs_vals) if len(obs_vals) > 0 else np.nan,
            "median": np.median(obs_vals) if len(obs_vals) > 0 else np.nan,
            "std": np.std(obs_vals) if len(obs_vals) > 0 else np.nan,
            "min": np.min(obs_vals) if len(obs_vals) > 0 else np.nan,
            "max": np.max(obs_vals) if len(obs_vals) > 0 else np.nan
        })
        
        if len(model_vals) > 0:
            plot_data.append(model_vals)
            labels.append(f"{ds_name}\n(Model)")
        if len(obs_vals) > 0:
            plot_data.append(obs_vals)
            labels.append(f"{ds_name}\n(Obs)")

    # Create summary DataFrame
    stats_df = pd.DataFrame(stats_records)
    if not stats_df.empty:
        # Pivot for cleaner reading
        pivot_df = stats_df.pivot(index="pigment", columns="source", values=["mean", "median", "std", "count"])
        print("\nRegional Statistics Summary:")
        print(pivot_df.round(3))
        
        # Plotting
        if plot_data:
            num_pigments = len(pigment_mapping)
            # Adjust figure size based on number of pigments
            fig, ax = plt.subplots(figsize=(max(10, num_pigments * 1.5), 6))
            
            # Create boxplot
            ax.boxplot(plot_data, labels=labels, showfliers=False)
            ax.set_title("Regional Pigment Distributions (Model vs In-Situ)")
            ax.set_ylabel("Concentration (mg/m^3)")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
    else:
        print("No valid data found for regional comparison.")

def main():
    global LAT_TOL_DEG, LON_TOL_DEG
    
    print(f"Loading SDP results from {SDP_RESULT_PATH}")
    dataset = xr.open_dataset(SDP_RESULT_PATH)
    
    print(f"Loading HPLC data from {HPLC_PATH}")
    hplc_df = pd.read_csv(HPLC_PATH)

    hplc_df.columns = [col.strip() for col in hplc_df.columns]

    date_str = hplc_df["date"].astype(str).str.zfill(8)
    hplc_df["timestamp"] = pd.to_datetime(date_str + " " + hplc_df["time"].str.strip(), format="%Y%m%d %H:%M:%S")

    pigment_mapping = { # map to in_situ_hplc .csv column names
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

    available_pigments = [name for name in pigment_mapping if name in dataset.data_vars]
    print(f"Dataset pigments: {available_pigments}")
    print(f"HPLC rows: {len(hplc_df)}")

    lat_grid = dataset["lat"].values
    lon_grid = dataset["lon"].values

    lat_spacing = _grid_spacing(lat_grid)
    lon_spacing = _grid_spacing(lon_grid)
    if np.isfinite(lat_spacing):
        LAT_TOL_DEG = max(LAT_TOL_DEG, lat_spacing * GRID_SNAP_FRACTION)
    if np.isfinite(lon_spacing):
        LON_TOL_DEG = max(LON_TOL_DEG, lon_spacing * GRID_SNAP_FRACTION)
    print(
        "Grid spacing (deg):",
        f"lat={lat_spacing:.3f}",
        f"lon={lon_spacing:.3f}",
        "| tolerances set to",
        f"lat={LAT_TOL_DEG:.3f}",
        f"lon={LON_TOL_DEG:.3f}",
    )

    snapped_hplc_df, grid_hplc_df = snap_hplc_to_grid(hplc_df, lat_grid, lon_grid, pigment_mapping)
    print(
        f"Snapped {len(snapped_hplc_df)} HPLC rows onto {len(grid_hplc_df)} unique grid cells",
        f"(original rows: {len(hplc_df)})",
    )

    matches_df = collect_matches(dataset, grid_hplc_df, pigment_mapping)
    print(f"Total pigment pairs: {len(matches_df)}")
    matches_df["within_tol"] = (matches_df["lat_delta"] <= LAT_TOL_DEG) & (
        matches_df["lon_delta"] <= LON_TOL_DEG
    )
    print(f"Pairs within ±{LAT_TOL_DEG:.3f}° lat/lon: {matches_df['within_tol'].sum()}")

    validated_df = matches_df[matches_df["within_tol"]].copy()
    validated_df.reset_index(drop=True, inplace=True)
    print(f"Validated pigment matches: {len(validated_df)}")

    if validated_df.empty:
        summary_df = pd.DataFrame(
            columns=["pigment", "pigment_label", "n", "bias", "mae", "rmse", "pearson_r"]
        )
    else:
        summary_df = (
            validated_df.groupby(["pigment", "pigment_label"], sort=True, group_keys=False)
            .apply(summarize_pigment)
            .reset_index()
            .sort_values("pigment")
        )
    print(summary_df)

    plot_parity(validated_df, summary_df)
    plot_bland_altman(validated_df)
    compare_regional_distributions(dataset, hplc_df, pigment_mapping)

    if snapped_hplc_df.empty:
        print("No HPLC stations could be snapped to the model grid.")
    else:
        lat_deltas = np.abs(snapped_hplc_df["grid_lat_delta"].values)
        lon_deltas = np.abs(snapped_hplc_df["grid_lon_delta"].values)
        within_tol = (lat_deltas <= LAT_TOL_DEG) & (lon_deltas <= LON_TOL_DEG)
        print(
            "Nearest-grid offsets (deg):",
            f"lat median={np.median(lat_deltas):.3f}, max={np.max(lat_deltas):.3f};",
            f"lon median={np.median(lon_deltas):.3f}, max={np.max(lon_deltas):.3f}",
        )
        print(
            f"Stations (records) within ±{LAT_TOL_DEG:.3f}° of their snapped cell: {within_tol.sum()}"
        )

    if not grid_hplc_df.empty:
        print(
            f"Grid cells with stations: {len(grid_hplc_df)} | median samples/cell = {grid_hplc_df['sample_count'].median():.1f}"
        )
    else:
        print("No grid-aligned HPLC cells available for comparison.")

    pigment_availability = (
        pd.Series({label: hplc_df[label].apply(pd.to_numeric, errors="coerce").notna().sum()
                for label in pigment_mapping.values() if label in hplc_df})
        .sort_values(ascending=False)
    )
    print(pigment_availability)

    # Run regional comparison
    compare_regional_distributions(dataset, hplc_df, pigment_mapping)

if __name__ == "__main__":
    main()
