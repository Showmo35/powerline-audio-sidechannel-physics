#!/bin/bash
#SBATCH --job-name=eval_soundbar
#SBATCH --account=<your_allocation>
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-exp
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=<SCRATCH_ROOT>/slurm_soundbar_%j.out
#SBATCH --error=<SCRATCH_ROOT>/slurm_soundbar_%j.err

set -euo pipefail
module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "========================================"
echo "Eval: sound_bar/alice_chap04_real.bin"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "========================================"

cd "<REPO_ROOT>"
python3 -u eval_soundbar.py

echo ""
echo "========================================"
echo "Done: $(date)"
echo "========================================"
