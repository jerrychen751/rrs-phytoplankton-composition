#!/bin/bash
#SBATCH --job-name=regime_cv_l2
#SBATCH --output=experiments/method_comparison/outputs/regime_routing_l2/slurm_%j.log
#SBATCH --error=experiments/method_comparison/outputs/regime_routing_l2/slurm_%j.err
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# Regime-Routing CV with L2 (~1 km swath) Rrs.
#
# Same regime split + per-regime 5-fold CV as the L3 run, but using PACE OCI
# Level-2 swath-level Rrs.
#
# Prerequisites: same L2 granules as multi_source_cv_l2 — already downloaded.
# ============================================================================

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"

source activate ai

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

cd ~/projects/rrs-SDP-pigments
mkdir -p experiments/method_comparison/outputs/regime_routing_l2

echo "=== Starting regime-routing CV (L2 Rrs) ==="
python -u experiments/method_comparison/regime_routing_cv.py --rrs-level L2

echo "=== Job finished at $(date) ==="
