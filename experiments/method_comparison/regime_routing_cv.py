#!/usr/bin/env python3
"""
Regime-Routing Cross-Validation.

Splits the combined dataset into two ocean regimes based on HPLC Tchla:
  - oligotrophic (Tchla < threshold mg/m³): open ocean / gyre waters
  - productive   (Tchla >= threshold mg/m³): coastal / upwelling / shelf waters

Runs 5-fold CV independently within each regime to identify which method
performs best per regime per pigment. Outputs a routing table for deployment:
given a PACE Chl estimate (same threshold applied to PACE OCI L3 chlor_a),
select the champion model for that regime.

Deployment Chl URL (same OPeNDAP server as Rrs):
  PACE_OCI.{YYYYMMDD}.L3m.DAY.CHL.V3_1.chlor_a.0p1deg.nc

Usage:
    python experiments/method_comparison/regime_routing_cv.py

Outputs: experiments/method_comparison/outputs/regime_routing/
    - oligotrophic_gof.csv  — R²/RMSE/MAE per method × pigment for oligo regime
    - productive_gof.csv    — same for productive regime
    - routing_table.csv     — per-pigment champion method per regime
    - regime_r2_comparison.png
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.method_comparison.config import CONFIG, SOURCES
from experiments.method_comparison.multi_source_cv import load_all_data, run_all_cv, METHOD_COLORS

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "regime_routing"

REGIME_NAMES = {0: "oligotrophic", 1: "productive"}


# ---------------------------------------------------------------------------
# 1. Regime labeling
# ---------------------------------------------------------------------------

def label_regime(tchla: np.ndarray, threshold: float) -> np.ndarray:
    """
    Assign each sample to a regime based on HPLC Tchla.

    Returns an integer array where:
      0  = oligotrophic (Tchla < threshold mg/m³)
      1  = productive   (Tchla >= threshold mg/m³)
     -1  = NaN Tchla, excluded from regime CV

    The 0.5 mg/m³ default separates open-ocean/gyre waters (dominated by
    pico-phytoplankton and low-Chl conditions) from coastal/upwelling/shelf
    waters with elevated chlorophyll and diverse phytoplankton assemblages.
    """
    labels = np.full(len(tchla), -1, dtype=int)
    valid = np.isfinite(tchla)
    labels[valid] = (tchla[valid] >= threshold).astype(int)
    return labels


# ---------------------------------------------------------------------------
# 2. Per-regime CV
# ---------------------------------------------------------------------------

def run_regime_cv(
    X: np.ndarray,
    y_dict: dict[str, np.ndarray],
    regime_labels: np.ndarray,
    config: dict,
    threshold: float,
) -> dict[int, pd.DataFrame]:
    """
    Run full 5-fold CV (all 8 methods × all 13 pigments) within each regime.

    Each regime is treated as an independent dataset: only samples labeled
    for that regime enter its k-fold CV. The KFold split is re-drawn from
    scratch per regime using the same seed, so results are directly comparable
    to the pooled multi_source_cv run.

    Returns a dict: {regime_id: gof_table DataFrame}.
    """
    k = config["cv"]["kfold_k"]
    regime_gof: dict[int, pd.DataFrame] = {}

    for regime_id, regime_name in REGIME_NAMES.items():
        mask = regime_labels == regime_id
        n = int(mask.sum())

        print(f"\n{'='*60}")
        thr_str = f"< {threshold}" if regime_id == 0 else f">= {threshold}"
        print(f"Regime {regime_id}: {regime_name}  (Tchla {thr_str} mg/m³)  n={n}")
        print(f"{'='*60}")

        if n < k * 2:
            print(f"  Skipping: fewer than {k * 2} samples for {k}-fold CV.")
            continue

        X_r = X[mask]
        y_r = {pig: y[mask] for pig, y in y_dict.items()}
        tchla_r = y_r.get("Tchla")

        gof_table, _ = run_all_cv(X_r, y_r, config, tchla=tchla_r)
        regime_gof[regime_id] = gof_table

        out_path = OUTPUT_DIR / f"{regime_name}_gof.csv"
        gof_table.to_csv(out_path, index=False)
        print(f"  Saved {out_path}")

    return regime_gof


# ---------------------------------------------------------------------------
# 3. Routing table
# ---------------------------------------------------------------------------

def build_routing_table(
    regime_gof: dict[int, pd.DataFrame],
    methods: list[str],
) -> pd.DataFrame:
    """
    For each pigment, find the best method (highest R²) per regime.

    Also prints the overall regime champion — the method with the best mean R²
    across all pigments within each regime.

    Returns a DataFrame with columns:
        pigment | oligotrophic_best | oligotrophic_R2 | productive_best | productive_R2
    """
    # Collect union of pigments across both regimes
    pigments: list[str] = []
    seen: set[str] = set()
    for gof in regime_gof.values():
        for pig in gof["pigment"].tolist():
            if pig not in seen:
                pigments.append(pig)
                seen.add(pig)

    rows = []
    for pig in pigments:
        row: dict = {"pigment": pig}
        for regime_id, regime_name in REGIME_NAMES.items():
            if regime_id not in regime_gof:
                row[f"{regime_name}_best"] = "N/A"
                row[f"{regime_name}_R2"] = np.nan
                continue

            gof = regime_gof[regime_id]
            pig_rows = gof[gof["pigment"] == pig]
            if pig_rows.empty:
                row[f"{regime_name}_best"] = "N/A"
                row[f"{regime_name}_R2"] = np.nan
                continue

            pig_row = pig_rows.iloc[0]
            r2s = {m: pig_row[f"{m}_R2"] for m in methods if f"{m}_R2" in pig_row.index}
            if not r2s:
                row[f"{regime_name}_best"] = "N/A"
                row[f"{regime_name}_R2"] = np.nan
                continue

            best = max(r2s, key=r2s.get)
            row[f"{regime_name}_best"] = best
            row[f"{regime_name}_R2"] = r2s[best]
        rows.append(row)

    routing = pd.DataFrame(rows)

    # Print overall regime champions (mean R² across all pigments)
    print("\n=== Overall Regime Champions (mean R² across pigments) ===")
    for regime_id, regime_name in REGIME_NAMES.items():
        if regime_id not in regime_gof:
            continue
        gof = regime_gof[regime_id]
        mean_r2 = {m: gof[f"{m}_R2"].mean() for m in methods if f"{m}_R2" in gof.columns}
        ranked = sorted(mean_r2.items(), key=lambda kv: kv[1], reverse=True)
        print(f"\n  {regime_name} (n={gof['n_samples'].sum()}):")
        for m, r2 in ranked:
            marker = " <-- champion" if m == ranked[0][0] else ""
            print(f"    {m:>14}: {r2:.3f}{marker}")

    return routing


# ---------------------------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------------------------

def plot_regime_r2_comparison(
    regime_gof: dict[int, pd.DataFrame],
    regime_sizes: dict[int, int],
    methods: list[str],
    threshold: float,
) -> None:
    """
    Side-by-side grouped bar charts: R² by method × pigment for each regime.
    Both regimes appear in one figure (one subplot per regime).
    """
    valid_regimes = [r for r in REGIME_NAMES if r in regime_gof]
    if not valid_regimes:
        return

    fig, axes = plt.subplots(
        1, len(valid_regimes),
        figsize=(max(10, 7 * len(valid_regimes)), 5),
        squeeze=False,
    )

    for col, regime_id in enumerate(valid_regimes):
        ax = axes[0, col]
        regime_name = REGIME_NAMES[regime_id]
        gof = regime_gof[regime_id]
        pigments = gof["pigment"].tolist()
        x = np.arange(len(pigments))
        width = 0.8 / len(methods)

        for i, method in enumerate(methods):
            col_name = f"{method}_R2"
            if col_name not in gof.columns:
                continue
            vals = gof[col_name].values
            offset = (i - len(methods) / 2 + 0.5) * width
            color = METHOD_COLORS.get(method, "#333333")
            ax.bar(x + offset, vals, width, label=method, color=color,
                   alpha=0.8, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels(pigments, rotation=45, ha="right")
        ax.set_ylabel("R²")
        ax.set_ylim(0, 1)
        thr_str = f"< {threshold}" if regime_id == 0 else f"≥ {threshold}"
        ax.set_title(
            f"{regime_name.title()} regime\n"
            f"Tchla {thr_str} mg/m³  (n={regime_sizes.get(regime_id, '?')})"
        )
        ax.legend(fontsize=7)

    fig.suptitle("Regime-Stratified 5-Fold CV: R² by Method", fontsize=13)
    fig.tight_layout()
    out = OUTPUT_DIR / "regime_r2_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main() -> None:
    global OUTPUT_DIR  # allow dynamic override based on --rrs-level

    parser = argparse.ArgumentParser(description="Regime-Routing CV")
    parser.add_argument(
        "--rrs-level", choices=["L3", "L2"], default="L3",
        help="Satellite Rrs resolution: L3 (0.1-deg gridded) or L2 (~1 km swath)",
    )
    args = parser.parse_args()
    rrs_level = args.rrs_level

    # Set output directory based on Rrs level
    if rrs_level == "L2":
        OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "regime_routing_l2"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.time()

    ms_cfg = CONFIG["multi_source"]
    pigments = ms_cfg["pigments"]
    threshold = CONFIG["regime_routing"]["tchla_threshold"]
    methods = CONFIG["sklearn"]["methods"]

    print("\n" + "=" * 60)
    print(f"Regime-Routing Cross-Validation  (Rrs level: {rrs_level})")
    print(f"Tchla threshold: {threshold} mg/m³  (0=oligotrophic, 1=productive)")
    print("=" * 60)

    # Load full combined dataset (identical to multi_source_cv.py)
    X, y_dict, source_labels, sample_ids = load_all_data(
        sources=SOURCES,
        pigments=pigments,
        spectral_cfg=ms_cfg["spectral"],
        temporal_window_days=ms_cfg["temporal_window_days"],
        rrs_level=rrs_level,
    )

    # Label regimes from HPLC Tchla
    tchla = y_dict["Tchla"]
    regime_labels = label_regime(tchla, threshold)

    n_oligo = int((regime_labels == 0).sum())
    n_prod = int((regime_labels == 1).sum())
    n_unknown = int((regime_labels == -1).sum())

    print(f"\nRegime split at Tchla = {threshold} mg/m³:")
    print(f"  Oligotrophic (0): {n_oligo} samples")
    print(f"  Productive   (1): {n_prod} samples")
    if n_unknown:
        print(f"  NaN Tchla (excl): {n_unknown} samples")

    # Warn if a regime is too small; report fallback counts
    k = CONFIG["cv"]["kfold_k"]
    if n_oligo < k * 2 or n_prod < k * 2:
        print(f"\nWARNING: a regime has < {k * 2} samples. Reporting fallback threshold sizes:")
        for fb in [0.3, 1.0]:
            rl_fb = label_regime(tchla, fb)
            n0, n1 = int((rl_fb == 0).sum()), int((rl_fb == 1).sum())
            print(f"  threshold={fb}: oligotrophic={n0}, productive={n1}")

    # Run per-regime CV
    regime_gof = run_regime_cv(X, y_dict, regime_labels, CONFIG, threshold)

    if not regime_gof:
        print("No regimes with sufficient samples — exiting.")
        return

    # Build and save routing table
    routing_table = build_routing_table(regime_gof, methods)
    routing_path = OUTPUT_DIR / "routing_table.csv"
    routing_table.to_csv(routing_path, index=False)
    print(f"\nRouting table → {routing_path}")
    print(routing_table.to_string(index=False))

    # Plot
    print("\nGenerating plots...")
    regime_sizes = {0: n_oligo, 1: n_prod}
    plot_regime_r2_comparison(regime_gof, regime_sizes, methods, threshold)

    total_elapsed = time.time() - total_start
    print(f"\nDone. Total time: {total_elapsed / 60:.1f} min")
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
