#!/bin/bash
#SBATCH --job-name=FSP_Eval
#SBATCH --account=<your_allocation>
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=24GB
#SBATCH --output=FSP_Eval_%j.out
#SBATCH --error=FSP_Eval_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

PIPE_DIR="<REPO_ROOT>/FullSubNetPlus"
cd "$PIPE_DIR"

mkdir -p "$PIPE_DIR/checkpoints"

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

python3 -u evaluate.py \
    --data "$PIPE_DIR/data/train_data_fullsubnetplus.npz" \
    --ckpt "$PIPE_DIR/checkpoints/best_model.pt" \
    --batch_size 8 \
    --out_json "$PIPE_DIR/checkpoints/val_metrics.json"

echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
