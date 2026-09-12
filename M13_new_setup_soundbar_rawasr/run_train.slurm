#!/bin/bash
#SBATCH --job-name=m13_rawasr
#SBATCH --partition=gpu-exp
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --account=<your_allocation>
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# usage:  sbatch run_train.slurm <arch> [extra train.py args...]
#   e.g.  sbatch run_train.slurm m5 --epochs 30 --batch 8

set -e
ARCH=${1:?need arch: m3|m4|m5|m6|m7}
shift || true

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu

cd "<REPO_ROOT>/M13_new_setup_soundbar_rawasr"
python -u train.py --arch "$ARCH" "$@"
