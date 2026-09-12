#!/bin/bash
#SBATCH --job-name=Powerline_Infer
#SBATCH --account=<your_allocation>
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16GB
#SBATCH --output=Powerline_Infer_%j.out
#SBATCH --error=Powerline_Infer_%j.err

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

UNET_DIR="<REPO_ROOT>/UNet"
cd "$UNET_DIR"

# Run on a 30s continuous segment (overlap-add + Whisper transcription)
python3 -u inference.py \
    --mode segment \
    --pl  "<REPO_ROOT>/May29_Alice/Chap_4_real.bin" \
    --mp3 "<REPO_ROOT>/Alice_In_Wonderland_mp3/Alice_In_Wonderland_ch_04.mp3" \
    --ckpt "$UNET_DIR/checkpoints/best_model.pt" \
    --out  "$UNET_DIR/inference_output" \
    --seg_dur 30.0

echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
