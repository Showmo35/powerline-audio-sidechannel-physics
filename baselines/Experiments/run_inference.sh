#!/bin/bash
#SBATCH --job-name=Exp_Infer
#SBATCH --account=<your_allocation>
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=24GB
#SBATCH --output=Exp_Infer_%j.out
#SBATCH --error=Exp_Infer_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "======================================================================"
echo "INFERENCE: Comparing all experiments"
echo "Job ID: $SLURM_JOB_ID | $(date) | $SLURMD_NODENAME | GPU $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

PROJECT="<REPO_ROOT>"
cd "$PROJECT/Experiments"

python3 -u inference_all.py \
    --data "$PROJECT/E2E/data/e2e_data.npz" \
    --out "$PROJECT/Experiments/results" \
    --exp1_ckpt "$PROJECT/Experiments/exp1_pure_ctc/best_model.pt" \
    --exp2_ckpt "$PROJECT/Experiments/exp2_ctc_weighted/best_model.pt" \
    --exp3_ckpt "$PROJECT/Experiments/exp3_larger_ctc/best_model.pt" \
    --n_samples 10 \
    --device cuda

echo "======================================================================"
echo "End: $(date)"
echo "======================================================================"
