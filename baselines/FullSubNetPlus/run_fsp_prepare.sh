#!/bin/bash
#SBATCH --job-name=FSP_Prep
#SBATCH --account=<your_allocation>
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=FSP_Prep_%j.out
#SBATCH --error=FSP_Prep_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

PIPE_DIR="<REPO_ROOT>/baselines/FullSubNetPlus"
cd "$PIPE_DIR"

mkdir -p "$PIPE_DIR/data"

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "======================================================================"

python3 -u prepare_data.py \
    --folders May29_Alice \
    --preprocess_mode bandpass \
    --window_sec 4.0 \
    --hop_sec 1.0 \
    --train_ratio 0.9 \
    --guard_sec 4.0 \
    --out_dir "$PIPE_DIR/data" \
    --out_name train_data_fullsubnetplus.npz

echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
