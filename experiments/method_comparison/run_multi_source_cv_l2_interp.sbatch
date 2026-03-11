#!/bin/bash
#SBATCH --job-name=method_cv_l2i
#SBATCH --output=experiments/method_comparison/outputs/multi_source_cv_l2_interp/slurm_%j.log
#SBATCH --error=experiments/method_comparison/outputs/multi_source_cv_l2_interp/slurm_%j.err
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# Multi-Source 5-Fold CV with L2-interp (neighborhood-averaged) Rrs.
#
# Same 7 methods × 13 pigments × 5 folds as the L3/L2 runs, but using
# neighborhood-averaged PACE OCI L2 swath-level Rrs (median of valid pixels
# within a 3 km radius) instead of single nearest pixel (L2) or gridded
# 0.1-deg (L3).
#
# Uses the same L2 granule files as the L2 run — no additional downloads.
#
# Memory: 64G because L2 granule files are ~400 MB each (vs ~3 MB L3 subsets).
# Time: 8h to allow for more I/O per matchup (bounding box extraction).
#
# Prerequisites (run on LOGIN NODE before submitting):
#   1. python experiments/method_comparison/download_l2_rrs.py
#   2. (optional) python experiments/method_comparison/download_sst_sss.py
# ============================================================================

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"

# --- Environment ---
source activate ai

# Fix libstdc++ ABI mismatch: conda's newer libstdc++ must precede the
# system GCC 12 libstdc++ on GT PACE ICE nodes.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Cap BLAS threads — individual matrix operations on ~120×299 data are small
# enough that multi-threading overhead dominates.
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

# --- Setup ---
cd ~/projects/rrs-SDP-pigments
mkdir -p experiments/method_comparison/outputs/multi_source_cv_l2_interp

echo "=== Starting 5-fold CV (L2-interp Rrs) ==="
python -u experiments/method_comparison/multi_source_cv.py --rrs-level L2-interp

echo "=== Job finished at $(date) ==="
