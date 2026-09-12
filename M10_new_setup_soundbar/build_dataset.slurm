#!/bin/bash
#SBATCH --job-name=sb_dataset
#SBATCH --account=<your_allocation>
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --array=1-46
#SBATCH --output=outputs/dataset_%A_%a.out
#SBATCH --error=outputs/dataset_%A_%a.err

# One array task per chunk. The AM-sideband front-end is CPU-only (STFT + mel),
# so this runs on the cpu partition — no GPU needed for dataset building.
set -euo pipefail

module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/miniconda3/envs/tf_gpu"

cd "<REPO_ROOT>/setup_soundbar"

echo "=== chunk index $SLURM_ARRAY_TASK_ID ==="
python -u dataset.py --chunk-index "$SLURM_ARRAY_TASK_ID" "$@"
echo "=== done ==="
