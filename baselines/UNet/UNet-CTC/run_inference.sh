#!/bin/bash
#SBATCH --job-name=UNet_Infer
#SBATCH --account=<your_allocation>
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16GB
#SBATCH --output=UNet_Infer_%j.out
#SBATCH --error=UNet_Infer_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "======================================================================"
echo "UNet Transcribe: Inference & Evaluation"
echo "Job ID: $SLURM_JOB_ID | $(date) | $SLURMD_NODENAME | GPU $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

PROJECT="<REPO_ROOT>"
cd "$PROJECT/UNet/transcribe"

python3 -u inference.py \
    --data "$PROJECT/UNet/transcribe/data/transcribe_data.npz" \
    --ckpt "$PROJECT/UNet/transcribe/checkpoints/best_model.pt" \
    --out "$PROJECT/UNet/transcribe/inference_output" \
    --batch 32 \
    --n_samples 10 \
    --device cuda

echo "======================================================================"
echo "End: $(date)"
echo "======================================================================"
