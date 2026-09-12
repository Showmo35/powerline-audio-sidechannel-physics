#!/bin/bash
#SBATCH --job-name=m14_plf_nb
#SBATCH --account=<your_allocation>
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=48GB
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=nb_m14_%j.out
#SBATCH --error=nb_m14_%j.err

# M14 PowerLine-Flow training, launched with the run_notebook.sh style
# (module-loaded miniconda, no explicit --partition → default scheduling lane).
# Resumes outputs/full_gpu so it accumulates with the other queued M14 jobs.
set -euo pipefail

PROJECT_DIR="<REPO_ROOT>/soundbar_setup/M14_new_setup_soundbar_melgen"
CONDA_ENV="tf_gpu"
CONDA_ENV_PATH="$HOME/miniconda3/envs/tf_gpu"

load_miniconda_module() {
    if module load miniconda3 >/dev/null 2>&1; then return 0; fi
    local module_ver
    module_ver="$(module -t spider miniconda3 2>&1 | grep -Eo 'miniconda3/[[:alnum:]._-]+' | head -n 1 || true)"
    if [[ -n "${module_ver}" ]]; then module load "${module_ver}"; return 0; fi
    echo "ERROR: Could not load miniconda3 module." >&2; return 1
}

activate_conda_env() {
    if conda activate "${CONDA_ENV}" >/dev/null 2>&1; then return 0; fi
    if [[ -d "${CONDA_ENV_PATH}" ]]; then conda activate "${CONDA_ENV_PATH}"; return 0; fi
    echo "ERROR: Could not activate conda env '${CONDA_ENV}'." >&2; return 1
}

module purge
load_miniconda_module
source "$(conda info --base)/etc/profile.d/conda.sh"
activate_conda_env

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID   Start: $(date)   Node: $SLURMD_NODENAME"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-?}   Python: $(which python)"
echo "======================================================================"

cd "${PROJECT_DIR}"
python -u train.py --resume --epochs 80 --batch 16 --windows-per-chunk 256 \
    --out-dir outputs/full_gpu "$@"
python -u viz.py --ckpt outputs/full_gpu/best.pt --n 8 --out outputs/full_gpu/plf_samples.png || true

echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
