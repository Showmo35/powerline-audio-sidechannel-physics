#!/bin/bash
#SBATCH --job-name=M6_HybridResume
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

DIR="<REPO_ROOT>/baselines/Module6_HybridCTCAttn"
cd "$DIR"

echo "========================================"
echo "Module 6: Hybrid CTC/Attention (RESUME)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "========================================"

# ── Step 1: Resume training from checkpoint_ep20 ────────────────────────────
echo ""
echo ">>> STEP 1: Resume Hybrid CTC/Attention training from epoch 20"
echo "----------------------------------------------------------------------"

python3 -u train.py \
    --data "$DIR/data/transcribe_data.npz" \
    --out "$DIR/checkpoints" \
    --epochs 200 \
    --lr 1e-3 \
    --batch 16 \
    --d_model 256 \
    --enc_layers 6 \
    --dec_layers 3 \
    --num_heads 4 \
    --conv_kernel 31 \
    --ctc_weight 0.7 \
    --warmup_pct 0.1 \
    --patience 30 \
    --resume "$DIR/checkpoints/checkpoint_ep20.pt"

# ── Step 2: Inference ────────────────────────────────────────────────────────
echo ""
echo ">>> STEP 2: Evaluate (CTC greedy vs Attention greedy)"
echo "----------------------------------------------------------------------"

python3 -u inference.py \
    --data "$DIR/data/transcribe_data.npz" \
    --ckpt "$DIR/checkpoints/best_cer_model.pt" \
    --out "$DIR/inference_output"

echo ""
echo "========================================"
echo "Module 6 resume complete: $(date)"
echo "========================================"
