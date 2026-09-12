#!/bin/bash
#SBATCH --job-name=M7_FullSubNet
#SBATCH --account=<your_allocation>
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48GB
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

DIR="<REPO_ROOT>/baselines/Module7_FullSubNet"
cd "$DIR"

echo "========================================"
echo "Module 7: FullSubNet+ Enhancement"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "========================================"

# ── Step 1: Prepare data ────────────────────────────────────────────────────
echo ""
echo ">>> STEP 1: Prepare data (bandpass 50-4kHz, mel spectrograms)"
echo "----------------------------------------------------------------------"

python3 -u prepare_data.py \
    --folders May29_Alice \
    --win_sec 1.0 \
    --hop_sec 0.25 \
    --out_dir "$DIR/data" \
    --out_name train_data.npz

# ── Step 2: Train FullSubNet+ ───────────────────────────────────────────────
echo ""
echo ">>> STEP 2: Train FullSubNet+ (L1 + MSTFT)"
echo "----------------------------------------------------------------------"

python3 -u train.py \
    --data "$DIR/data/train_data.npz" \
    --out "$DIR/checkpoints" \
    --epochs 300 \
    --lr 3e-4 \
    --batch 32 \
    --fb_hidden 64 \
    --sb_hidden 32 \
    --fb_layers 2 \
    --sb_layers 2 \
    --n_neighbor 3 \
    --l1_w 1.0 \
    --mstft_w 1.0 \
    --patience 30

# ── Step 3: Inference ────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 3: Evaluate (MSE, MAE, correlation vs clean)"
echo "----------------------------------------------------------------------"

python3 -u inference.py \
    --data "$DIR/data/train_data.npz" \
    --ckpt "$DIR/checkpoints/best_model.pt" \
    --out "$DIR/inference_output" \
    --n_samples 10

echo ""
echo "========================================"
echo "Module 7 complete: $(date)"
echo "========================================"
