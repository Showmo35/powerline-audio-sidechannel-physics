#!/bin/bash
#SBATCH --job-name=UNet_Prep
#SBATCH --account=<your_allocation>
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=UNet_Prep_%j.out
#SBATCH --error=UNet_Prep_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "======================================================================"
echo "UNet Transcribe: Data Preparation"
echo "Job ID: $SLURM_JOB_ID | $(date) | $SLURMD_NODENAME | GPU $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

PROJECT="<REPO_ROOT>"
cd "$PROJECT/UNet/transcribe"

python3 -u prepare_data.py \
    --folders May29_Alice \
    --win_sec 5.0 \
    --hop_sec 1.0 \
    --max_text_len 200 \
    --out_dir "$PROJECT/UNet/transcribe/data" \
    --out_name transcribe_data.npz \
    --unet_ckpt "$PROJECT/UNet/checkpoints_bp/best_model.pt" \
    --unet_batch 32 \
    --whisper_model base.en

echo "======================================================================"
echo "End: $(date)"
echo "======================================================================"
