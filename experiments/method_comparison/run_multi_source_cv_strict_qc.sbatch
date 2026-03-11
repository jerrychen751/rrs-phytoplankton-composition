#!/bin/bash
#SBATCH --job-name=cv_strict_qc
#SBATCH --output=experiments/method_comparison/outputs/multi_source_cv_strict_qc/slurm_%j.log
#SBATCH --error=experiments/method_comparison/outputs/multi_source_cv_strict_qc/slurm_%j.err
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# Multi-Source 5-Fold CV with strict QC: same as run_multi_source_cv.sbatch
# but with --strict-qc flag for A/B comparison against the baseline.
#
# Strict QC applies:
#   - HPLC concentration bounds (negative / implausibly high)
#   - Negative Rrs rejection (atmospheric correction failures)
#   - Rrs magnitude cap (glint/foam/shallow water)
#   - Spectral shape check (Rrs(670) > Rrs(490) = non-Case-1)
#   - GSM MAPD threshold (poor bio-optical model fit)
#   - Tighter temporal window (±1 day instead of ±3 days)
#
# Expected sample loss: ~10–30 from n≈267, leaving ~237–257.
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
mkdir -p experiments/method_comparison/outputs/multi_source_cv_strict_qc

echo "=== Starting 5-fold CV with strict QC ==="
python -u experiments/method_comparison/multi_source_cv.py --strict-qc

echo "=== Job finished at $(date) ==="
