#!/bin/bash
#SBATCH --job-name=M4_Soundbar
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

echo "========================================"
echo "Module 4 Soundbar: UNet + Perceptual Loss"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "========================================"

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

DIR="<REPO_ROOT>/baselines/Module4_PerceptualLoss_Soundbar"
DATA_DIR="${DIR}/data"
CKPT_DIR="${DIR}/checkpoints"
INF_DIR="${DIR}/inference_output"
DATA_FILE="${DATA_DIR}/train_data.npz"
CTC_CKPT="<REPO_ROOT>/baselines/UNet/UNet-CTC/checkpoints/best_model.pt"

if [ ! -f "${CTC_CKPT}" ]; then
    echo "ERROR: No CTC checkpoint found at ${CTC_CKPT}"
    exit 1
fi

cd "${DIR}"

mkdir -p "${DATA_DIR}" "${CKPT_DIR}" "${INF_DIR}"

echo ""
echo "========================================"
echo "Step 1: Preparing Soundbar leakage data"
echo "========================================"

if [ -f "${DATA_FILE}" ]; then
    echo "Data file already exists: ${DATA_FILE}"
    echo "Skipping data preparation. Delete the file to regenerate."
else
    python prepare_data.py
fi

echo ""
echo "========================================"
echo "Step 2: Training UNet with perceptual loss"
echo "========================================"

python train.py \
    --data "${DATA_FILE}" \
    --out "${CKPT_DIR}" \
    --ctc_ckpt "${CTC_CKPT}" \
    --epochs 300 \
    --lr 3e-4 \
    --batch 8 \
    --base_ch 32 \
    --l1_w 1.0 \
    --mstft_w 1.0 \
    --percep_w 0.1 \
    --patience 30 \
    --plot_every 10 \
    --save_every 50

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
echo "Module 4 Soundbar complete!"
echo "Checkpoints: ${CKPT_DIR}"
echo "Results:     ${INF_DIR}/results.json"
echo "========================================"
