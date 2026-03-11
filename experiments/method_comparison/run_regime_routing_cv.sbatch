#!/bin/bash
#SBATCH --job-name=regime_cv
#SBATCH --output=experiments/method_comparison/outputs/regime_routing/slurm_%j.log
#SBATCH --error=experiments/method_comparison/outputs/regime_routing/slurm_%j.err
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# Regime-Routing CV: per-regime 5-fold CV to build the routing table.
#
# Splits the 268-sample dataset into oligotrophic (Tchla < 0.5 mg/m³) and
# productive (Tchla >= 0.5 mg/m³) regimes, then runs all 7 methods × 13
# pigments × 5 folds within each regime. All methods are sklearn-based.
#
# Expected runtime: ~2–3h (two independent sub-experiments, each smaller
# than the 268-sample pooled run which took 2h55m).
#
# Prerequisites: same L3 Rrs files as multi_source_cv — already downloaded.
# ============================================================================

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"

source activate ai

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

cd ~/projects/rrs-SDP-pigments
mkdir -p experiments/method_comparison/outputs/regime_routing

echo "=== Starting regime-routing CV ==="
python -u experiments/method_comparison/regime_routing_cv.py

echo "=== Job finished at $(date) ==="
