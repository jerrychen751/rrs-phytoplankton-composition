#!/bin/bash
#SBATCH --job-name=method_cv_l2
#SBATCH --output=experiments/method_comparison/outputs/multi_source_cv_l2/slurm_%j.log
#SBATCH --error=experiments/method_comparison/outputs/multi_source_cv_l2/slurm_%j.err
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# Multi-Source 5-Fold CV with L2 (~1 km swath) Rrs.
#
# Same 7 methods × 13 pigments × 5 folds as the L3 run, but using PACE OCI
# Level-2 swath-level Rrs instead of the 0.1-deg gridded L3 product.
#
# Memory: 64G because L2 granule files are ~400 MB each (vs ~3 MB L3 subsets).
# Time: 8h to allow for more I/O per matchup and potentially fewer matches
# (L2 QC is stricter; may need more granule searches).
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
mkdir -p experiments/method_comparison/outputs/multi_source_cv_l2

echo "=== Starting 5-fold CV (L2 Rrs) ==="
python -u experiments/method_comparison/multi_source_cv.py --rrs-level L2

echo "=== Job finished at $(date) ==="
