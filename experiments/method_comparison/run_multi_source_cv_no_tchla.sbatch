#!/bin/bash
#SBATCH --job-name=cv_no_tchla
#SBATCH --output=experiments/method_comparison/outputs/multi_source_cv_no_tchla/slurm_%j.log
#SBATCH --error=experiments/method_comparison/outputs/multi_source_cv_no_tchla/slurm_%j.err
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# Multi-Source 5-Fold CV WITHOUT Tchla conditioning.
#
# Identical to run_multi_source_cv.sbatch except --no-tchla-conditioning is
# passed.  Pigments are predicted from spectral features only (299 features),
# without the log10(Tchla) auxiliary feature.
#
# Compare outputs against multi_source_cv/ (WITH Tchla) to quantify the
# benefit of Tchla conditioning.
# ============================================================================

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"

source activate ai

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

cd ~/projects/rrs-SDP-pigments
mkdir -p experiments/method_comparison/outputs/multi_source_cv_no_tchla

echo "=== Starting 5-fold CV (no Tchla conditioning) ==="
python -u experiments/method_comparison/multi_source_cv.py --no-tchla-conditioning

echo "=== Job finished at $(date) ==="
