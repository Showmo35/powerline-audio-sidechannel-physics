#!/bin/bash
#SBATCH --job-name=M4_UNet_Soundbar
#SBATCH --account=<your_allocation>
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail

echo "========================================"
echo "Module 4 Soundbar: UNet Reconstruction"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node:   $(hostname)"
echo "Date:   $(date)"
echo "========================================"

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

DIR="<REPO_ROOT>/Module4_UNet_Soundbar"
DATA_DIR="${DIR}/data"
CKPT_DIR="${DIR}/checkpoints"
INF_DIR="${DIR}/inference_output"
DATA_FILE="${DATA_DIR}/train_data.npz"

# Use local node storage for fast shard I/O (avoids slow Lustre reads during training)
LOCAL_SHARD_DIR="/tmp/m4_shards_${SLURM_JOB_ID:-local}"
LOCAL_DATA_FILE="${LOCAL_SHARD_DIR}/train_data.npz"

cd "${DIR}"
mkdir -p "${DATA_DIR}" "${CKPT_DIR}" "${INF_DIR}"

# ── Step 1: Data prep ─────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Step 1: Preparing Soundbar data"
echo "========================================"

python prepare_data.py \
    --out_dir "${DATA_DIR}" \
    --out_name train_data.npz \
    --pair_mode direct \
    --tail_sec 1800 \
    --segment_sec 300 \
    --win_sec 1.0 \
    --hop_sec 0.25 \
    --shard_windows 512

# ── Copy shards to local /tmp for fast I/O ────────────────────────────────────
echo ""
echo "========================================"
echo "Copying shards to local node storage"
echo "========================================"
mkdir -p "${LOCAL_SHARD_DIR}/shards"
cp "${DATA_FILE}" "${LOCAL_DATA_FILE}"
cp "${DATA_DIR}/shards/"*.npz "${LOCAL_SHARD_DIR}/shards/"
# Patch the shard_dir pointer in the descriptor to point to local copy
python3 - "${LOCAL_DATA_FILE}" "${LOCAL_SHARD_DIR}/shards" <<'PYEOF'
import numpy as np, sys
src, shard_dir = sys.argv[1], sys.argv[2]
d = np.load(src, allow_pickle=True)
fields = {k: d[k] for k in d.files}
fields["shard_dir"] = np.array(shard_dir)
np.savez(src, **fields)
print(f"  shard_dir patched -> {shard_dir}")
PYEOF
echo "Done. Shards available at ${LOCAL_SHARD_DIR}/shards"

# ── Step 2: Training ──────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Step 2: Training UNet"
echo "========================================"

python train.py \
    --data "${LOCAL_DATA_FILE}" \
    --out "${CKPT_DIR}" \
    --epochs 80 \
    --lr 3e-4 \
    --batch 32 \
    --base_ch 48 \
    --l1_w 1.0 \
    --mstft_w 1.0 \
    --patience 12 \
    --plot_every 20 \
    --save_every 20 \
    --workers 8

# ── Step 3: Inference ─────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "Step 3: Inference and evaluation"
echo "========================================"

python inference.py \
    --data "${LOCAL_DATA_FILE}" \
    --ckpt "${CKPT_DIR}/best_model.pt" \
    --out "${INF_DIR}" \
    --batch 32 \
    --n_samples 20

# ── Cleanup local tmp ─────────────────────────────────────────────────────────
rm -rf "${LOCAL_SHARD_DIR}"

echo ""
echo "========================================"
echo "Module 4 Soundbar complete!"
echo "Checkpoints: ${CKPT_DIR}"
echo "Results:     ${INF_DIR}/results.json"
echo "========================================"