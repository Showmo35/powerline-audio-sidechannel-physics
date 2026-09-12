#!/bin/bash
#SBATCH --job-name=Default_Infer
#SBATCH --account=<your_allocation>
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16GB
#SBATCH --output=Default_Infer_%j.out
#SBATCH --error=Default_Infer_%j.err

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "======================================================================"

UNET_DIR="<REPO_ROOT>/baselines/UNet"
cd "$UNET_DIR"

# Val-only inference: spectrogram comparison + Griffin-Lim audio (no Whisper)
python3 -u inference.py \
    --mode val \
    --ckpt "$UNET_DIR/checkpoints_default/best_model.pt" \
    --out  "$UNET_DIR/inference_output_default" \
    --data "$UNET_DIR/data/train_data_random.npz" \
    --n_samples 10

echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
