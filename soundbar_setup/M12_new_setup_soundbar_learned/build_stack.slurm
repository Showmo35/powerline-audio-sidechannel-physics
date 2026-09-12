#!/bin/bash
#SBATCH --job-name=sbl_build
#SBATCH --account=<your_allocation>
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --array=1-46
#SBATCH --output=outputs/build_%A_%a.out
#SBATCH --error=outputs/build_%A_%a.err
set -euo pipefail
module purge
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$HOME/miniconda3/envs/tf_gpu"
cd "<REPO_ROOT>/setup_soundbar_learned"
python -u dataset_enh.py --chunk-index "$SLURM_ARRAY_TASK_ID" "$@"
