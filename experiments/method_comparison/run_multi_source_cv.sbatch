#!/bin/bash
#SBATCH --job-name=method_cv
#SBATCH --output=experiments/method_comparison/outputs/multi_source_cv/slurm_%j.log
#SBATCH --error=experiments/method_comparison/outputs/multi_source_cv/slurm_%j.err
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# Multi-Source 5-Fold CV: 7 methods × 13 pigments across 7 HPLC datasets
# with Tchla conditioning (log10 Tchla appended for non-Tchla pigments).
#
# All methods are sklearn-based — no GPU required.
#
# Memory: 48G for xarray L3 file I/O across all source date ranges.
# Time: ~2-4h; SVR grid search (5×5 grid, 5-fold inner CV) is the bottleneck.
#
# Prerequisites (run on LOGIN NODE before submitting):
#   1. python experiments/method_comparison/download_l3_rrs.py
#   2. (optional, for accurate physics residuals):
#      python experiments/method_comparison/download_sst_sss.py
#      Without SST/SSS files, the script falls back to climatological
#      defaults (SST=20°C, SSS=35 PSU) which is a minor approximation.
# ============================================================================

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"

# --- Environment ---
source activate ai

# Fix libstdc++ ABI mismatch: conda's newer libstdc++ must precede the
# system GCC 12 libstdc++ on GT PACE ICE nodes.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Cap BLAS threads — we parallelise across pigments/folds conceptually, but
# each individual matrix operation is on data small enough (~120×299) that
# multi-threading overhead dominates. 4 threads is a reasonable sweet spot
# at this training-set size (vs 2 for the tiny PAX LOOCV).
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# --- Setup ---
cd ~/projects/rrs-SDP-pigments
mkdir -p experiments/method_comparison/outputs/multi_source_cv

echo "=== Starting 5-fold CV ==="
python -u experiments/method_comparison/multi_source_cv.py

echo "=== Job finished at $(date) ==="
