#!/bin/bash
#SBATCH --job-name=M3_BeamLM
#SBATCH --account=<your_allocation>
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

DIR="<REPO_ROOT>/baselines/Module3_BeamSearchLM"
cd "$DIR"

echo "========================================"
echo "Module 3: CTC Beam Search + LM"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "========================================"

# ── Step 1: Prepare CTC data ────────────────────────────────────────────────
echo ""
echo ">>> STEP 1: Prepare CTC training data (bandpass 50-4kHz)"
echo "----------------------------------------------------------------------"

python3 -u prepare_data.py \
    --folders May29_Alice \
    --win_sec 5.0 \
    --hop_sec 1.0 \
    --out_dir "$DIR/data" \
    --out_name transcribe_data.npz \
    --unet_ckpt "<REPO_ROOT>/baselines/UNet/checkpoints_bp/best_model.pt"

# ── Step 2: Train LM from prepared npz ──────────────────────────────────────
echo ""
echo ">>> STEP 2: Train character-level LM (from npz text labels)"
echo "----------------------------------------------------------------------"

python3 -u train_lm.py \
    --npz_file "$DIR/data/transcribe_data.npz" \
    --out "$DIR/data/lm.pkl" \
    --order 4 \
    --smoothing 0.01

# ── Step 3: Train CTC model ─────────────────────────────────────────────────
echo ""
echo ">>> STEP 3: Train CTC model (dual checkpoints: best_loss + best_cer)"
echo "----------------------------------------------------------------------"

python3 -u train.py \
    --data "$DIR/data/transcribe_data.npz" \
    --out "$DIR/checkpoints" \
    --epochs 200 \
    --lr 3e-4 \
    --batch 16 \
    --gru_hidden 256 \
    --gru_layers 2 \
    --patience 30

# ── Step 4: Inference (greedy + beam + beam+LM) ─────────────────────────────
echo ""
echo ">>> STEP 4: Evaluate (greedy vs beam vs beam+LM)"
echo "----------------------------------------------------------------------"

python3 -u inference.py \
    --data "$DIR/data/transcribe_data.npz" \
    --ckpt "$DIR/checkpoints/best_cer_model.pt" \
    --lm "$DIR/data/lm.pkl" \
    --out "$DIR/inference_output" \
    --beam_width 50 \
    --lm_weight 0.3

echo ""
echo "========================================"
echo "Module 3 complete: $(date)"
echo "========================================"
