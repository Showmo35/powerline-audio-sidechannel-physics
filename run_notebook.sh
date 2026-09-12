#!/bin/bash
#SBATCH --job-name=usb_whisper_finetune
#SBATCH --account=<your_allocation>
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32GB
#SBATCH --output=usb_whisper_finetune%j.out
#SBATCH --error=usb_whisper_finetune%j.err

set -euo pipefail

PROJECT_DIR="<REPO_ROOT>"
TRAIN_SCRIPT="Training_Scripts/usb_whisper_finetune.py"
CONDA_ENV="tf_gpu"
CONDA_ENV_PATH="$HOME/miniconda3/envs/tf_gpu"

load_miniconda_module() {
    if module load miniconda3 >/dev/null 2>&1; then
        return 0
    fi

    local module_ver
    module_ver="$(module -t spider miniconda3 2>&1 | grep -Eo 'miniconda3/[[:alnum:]._-]+' | head -n 1 || true)"
    if [[ -n "${module_ver}" ]]; then
        module load "${module_ver}"
        return 0
    fi

    echo "ERROR: Could not load miniconda3 module." >&2
    echo "Try: module spider miniconda3" >&2
    return 1
}

activate_conda_env() {
    if conda activate "${CONDA_ENV}" >/dev/null 2>&1; then
        return 0
    fi

    if [[ -d "${CONDA_ENV_PATH}" ]]; then
        conda activate "${CONDA_ENV_PATH}"
        return 0
    fi

    echo "ERROR: Could not activate conda env '${CONDA_ENV}'." >&2
    echo "Tried name and path: ${CONDA_ENV_PATH}" >&2
    echo "Run 'conda info --envs' to pick an available env." >&2
    return 1
}

# Load modules and activate conda environment
module purge
load_miniconda_module

source "$(conda info --base)/etc/profile.d/conda.sh"
activate_conda_env

# Print environment info
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
echo "======================================================================"

# Navigate to project directory
cd "${PROJECT_DIR}"

# Run ID training with unbuffered output, forwarding any CLI args (e.g. --predict-only)
python -u "${TRAIN_SCRIPT}" "$@"

# Print completion info
echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
