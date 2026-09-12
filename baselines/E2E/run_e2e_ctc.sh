#!/bin/bash
#SBATCH --job-name=E2E_CTC
#SBATCH --account=<your_allocation>
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=E2E_CTC_%j.out
#SBATCH --error=E2E_CTC_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

E2E_DIR="<REPO_ROOT>/baselines/E2E"
cd "$E2E_DIR"
DATA_NPZ="$E2E_DIR/data/e2e_data.npz"

if [[ ! -f "$DATA_NPZ" ]]; then
  echo "Missing data file: $DATA_NPZ"
  exit 1
fi

# Objective 1: direct envelope -> text via CTC.
python3 -u train_e2e.py \
    --data "$DATA_NPZ" \
    --out "$E2E_DIR/checkpoints_ctc" \
    --mode ctc \
    --epochs 180 \
    --batch 16 \
    --lr 2e-4 \
    --alpha 1.0 \
    --beta 1.0 \
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

echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
