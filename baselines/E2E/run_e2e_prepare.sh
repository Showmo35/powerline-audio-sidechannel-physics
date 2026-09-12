#!/bin/bash
#SBATCH --job-name=E2E_Prep
#SBATCH --account=<your_allocation>
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --output=E2E_Prep_%j.out
#SBATCH --error=E2E_Prep_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "======================================================================"

E2E_DIR="<REPO_ROOT>/baselines/E2E"
cd "$E2E_DIR"
mkdir -p "$E2E_DIR/data"

# Build envelope/MP3/text windows with Whisper word timestamps.
python3 -u prepare_data.py \
    --folders May29_Alice \
    --win_sec 5.0 \
    --hop_sec 1.0 \
    --max_text_len 200 \
    --out_dir "$E2E_DIR/data" \
    --out_name e2e_data.npz \
    --whisper_model base.en

echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
