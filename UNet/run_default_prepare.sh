#!/bin/bash
#SBATCH --job-name=Default_Prep
#SBATCH --account=<your_allocation>
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=Default_Prep_%j.out
#SBATCH --error=Default_Prep_%j.err

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "======================================================================"

UNET_DIR="<REPO_ROOT>/UNet"
cd "$UNET_DIR"

# Prepare data with random 90:10 split
python3 -u prepare_data_hpc.py \
    --folders May29_Alice \
    --split random \
    --out_name train_data_random.npz

echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
