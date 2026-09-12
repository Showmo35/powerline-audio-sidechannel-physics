#!/bin/bash
#SBATCH --job-name=FSP_Infer
#SBATCH --account=<your_allocation>
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=24GB
#SBATCH --output=FSP_Infer_%j.out
#SBATCH --error=FSP_Infer_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

PIPE_DIR="<REPO_ROOT>/baselines/FullSubNetPlus"
cd "$PIPE_DIR"

mkdir -p "$PIPE_DIR/inference_output"

PL_BIN="<REPO_ROOT>/May29_Alice/Chap_4_real.bin"


echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

python3 -u inference.py \
    --pl_bin "$PL_BIN" \
    --chapter_name chapter_04 \
    --ckpt "$PIPE_DIR/checkpoints/best_model.pt" \
    --out_dir "$PIPE_DIR/inference_output" \
    --window_sec 4.0 \
    --hop_sec 1.0 \
    --report_metrics

echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
