#!/bin/bash
#SBATCH --job-name=UNetOld_Train
#SBATCH --account=<your_allocation>
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=UNetOld_Train_%j.out
#SBATCH --error=UNetOld_Train_%j.err

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

UNET_OLD_DIR="<REPO_ROOT>/baselines/UNet_Old"
DATA="<REPO_ROOT>/baselines/UNet/data/train_data_random.npz"
cd "$UNET_OLD_DIR"

python3 -u train.py \
    --data "$DATA" \
    --out  "$UNET_OLD_DIR/checkpoints" \
    --epochs 300 \
    --lr 3e-4 \
    --batch 32 \
    --base_ch 32

echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
