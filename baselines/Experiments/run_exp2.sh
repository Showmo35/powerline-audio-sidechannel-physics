#!/bin/bash
#SBATCH --job-name=Exp2_CTCWeight
#SBATCH --account=<your_allocation>
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=Exp2_%j.out
#SBATCH --error=Exp2_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "======================================================================"
echo "EXP 2: UNet + CTC with higher CTC weight (alpha=0.5, beta=2.0)"
echo "Job ID: $SLURM_JOB_ID | $(date) | $SLURMD_NODENAME | GPU $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

PROJECT="<REPO_ROOT>"
cd "$PROJECT/E2E"

python3 -u train_e2e.py \
    --data "$PROJECT/E2E/data/e2e_data.npz" \
    --out "$PROJECT/Experiments/exp2_ctc_weighted" \
    --mode ctc \
    --epochs 180 \
    --batch 16 \
    --lr 2e-4 \
    --alpha 0.5 \
    --beta 2.0 \
    --gamma 0.0 \
    --base_ch 64 \
    --gru_hidden 256 \
    --gru_layers 2 \
    --dropout 0.1 \
    --max_text_len 200 \
    --num_workers 4 \
    --patience 30 \
    --save_every 20 \
    --decode_batches 8 \
    --device cuda

echo "======================================================================"
echo "End: $(date)"
echo "======================================================================"
