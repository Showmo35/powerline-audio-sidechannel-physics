#!/bin/bash
#SBATCH --job-name=Exp1_PureCTC
#SBATCH --account=<your_allocation>
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=Exp1_%j.out
#SBATCH --error=Exp1_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "======================================================================"
echo "EXP 1: Pure CTC (no UNet)"
echo "Job ID: $SLURM_JOB_ID | $(date) | $SLURMD_NODENAME | GPU $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

PROJECT="<REPO_ROOT>"
cd "$PROJECT/Experiments"

python3 -u train_pure_ctc.py \
    --data "$PROJECT/E2E/data/e2e_data.npz" \
    --out "$PROJECT/Experiments/exp1_pure_ctc" \
    --epochs 200 \
    --batch 16 \
    --lr 3e-4 \
    --gru_hidden 256 \
    --gru_layers 2 \
    --dropout 0.2 \
    --patience 30 \
    --decode_batches 8 \
    --num_workers 4 \
    --device cuda

echo "======================================================================"
echo "End: $(date)"
echo "======================================================================"
