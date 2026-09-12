#!/bin/bash
#SBATCH --job-name=M4_Alice
#SBATCH --account=<your_allocation>
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=slurm_alice_%j.out
#SBATCH --error=slurm_alice_%j.err

set -euo pipefail

echo "========================================"
echo "Module 4 Soundbar: Alice Pipeline"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "========================================"

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

DIR="<REPO_ROOT>/baselines/Module4_UNet_Soundbar"
DATA_DIR="${DIR}/data"
CKPT_DIR="${DIR}/checkpoints_alice"
INF_DIR="${DIR}/inference_output_alice"
DATA_FILE="${DATA_DIR}/train_data_alice.npz"

cd "${DIR}"
mkdir -p "${DATA_DIR}" "${CKPT_DIR}" "${INF_DIR}"

# ── Step 1: Data prep (Alice, temporal split) ─────────────────────────────────
echo ""
echo "========================================"
echo "Step 1: Preparing Alice data"
echo "========================================"

python prepare_data_alice.py \
    --out_dir "${DATA_DIR}" \
    --out_name train_data_alice.npz \
    --win_sec 1.0 \
    --hop_sec 0.25 \
    --split temporal

# ── Step 2: Training ──────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Step 2: Training UNet (300 epochs)"
echo "========================================"

python train.py \
    --data "${DATA_FILE}" \
    --out "${CKPT_DIR}" \
    --epochs 300 \
    --lr 3e-4 \
    --batch 32 \
    --base_ch 64 \
    --l1_w 1.0 \
    --mstft_w 1.0 \
    --patience 30 \
    --plot_every 25 \
    --save_every 50 \
    --workers 4

# ── Step 3: Inference ─────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Step 3: Inference and evaluation"
echo "========================================"

python inference.py \
    --data "${DATA_FILE}" \
    --ckpt "${CKPT_DIR}/best_model.pt" \
    --out "${INF_DIR}" \
    --batch 32 \
    --n_samples 20

echo ""
echo "========================================"
echo "Alice pipeline complete!"
echo "Checkpoints : ${CKPT_DIR}"
echo "Results     : ${INF_DIR}/results.json"
echo "========================================"
