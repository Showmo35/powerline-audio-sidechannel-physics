#!/bin/bash
#SBATCH --job-name=E2E_Infer
#SBATCH --account=<your_allocation>
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=24GB
#SBATCH --output=E2E_Infer_%j.out
#SBATCH --error=E2E_Infer_%j.err

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
DATA_NPZ="$E2E_DIR/data/e2e_data.npz"
cd "$E2E_DIR"

if [[ ! -f "$DATA_NPZ" ]]; then
  echo "Missing data file: $DATA_NPZ"
  exit 1
fi

python3 -u inference_e2e.py \
  --data "$DATA_NPZ" \
  --ckpt "$E2E_DIR/checkpoints_ctc/best_model.pt" \
  --out "$E2E_DIR/inference_output_ctc" \
  --mode ctc \
  --n_samples 5 \
  --sample_mode first \
  --batch 2 \
  --device cuda

python3 -u inference_e2e.py \
  --data "$DATA_NPZ" \
  --ckpt "$E2E_DIR/checkpoints_conditioned/best_model.pt" \
  --out "$E2E_DIR/inference_output_conditioned" \
  --mode conditioned \
  --n_samples 5 \
  --sample_mode first \
  --batch 2 \
  --device cuda

python3 -u inference_e2e.py \
  --data "$DATA_NPZ" \
  --ckpt "$E2E_DIR/checkpoints_joint/best_model.pt" \
  --out "$E2E_DIR/inference_output_joint" \
  --mode joint \
  --n_samples 5 \
  --sample_mode first \
  --batch 2 \
  --device cuda

echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
