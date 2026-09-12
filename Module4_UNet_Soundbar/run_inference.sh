#!/bin/bash
#SBATCH --job-name=M4_UNet_Infer
#SBATCH --account=<your_allocation>
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=16GB
#SBATCH --output=slurm_infer_%j.out
#SBATCH --error=slurm_infer_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

DIR="<REPO_ROOT>/Module4_UNet_Soundbar"
cd "${DIR}"

python inference.py \
    --data "${DIR}/data/train_data.npz" \
    --ckpt "${DIR}/checkpoints/best_model.pt" \
    --out "${DIR}/inference_output" \
    --batch 32 \
    --n_samples 20

echo "Inference complete. Results: ${DIR}/inference_output/results.json"
