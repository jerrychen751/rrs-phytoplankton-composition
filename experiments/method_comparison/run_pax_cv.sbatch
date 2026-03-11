#!/bin/bash
#SBATCH --job-name=pax_cv
#SBATCH --output=outputs/pax_coastal_cv/slurm_%j.log
#SBATCH --error=outputs/pax_coastal_cv/slurm_%j.err
#SBATCH --partition=ice-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# PAX Coastal LOOCV: 7 methods x 13 pigments comparison
#
# All methods are sklearn-based (no deep learning). 8 CPUs keeps BLAS
# threads reasonable for the small matrix operations (~30x299).
#
# Expected runtime: ~1-2 hours
# ============================================================================

echo "=== Job started at $(date) ==="
echo "Node: $(hostname)"

# --- Environment setup ---
source activate ai

# Fix libstdc++ ABI mismatch: conda's newer libstdc++ must come before
# PACE's system GCC 12 libstdc++ in LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Cap BLAS threads — our matrices are tiny (~30x299), so multi-threading
# overhead exceeds the parallelism benefit. 2 threads is a sweet spot.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

# --- Navigate to project ---
cd ~/projects/rrs-SDP-pigments

# Create output directory
mkdir -p experiments/method_comparison/outputs/pax_coastal_cv

# --- Run the experiment ---
echo "=== Starting LOOCV experiment ==="
python -u experiments/method_comparison/pax_coastal_cv.py

echo "=== Job finished at $(date) ==="
