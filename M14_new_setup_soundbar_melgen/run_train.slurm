#!/bin/bash
#SBATCH --job-name=m14_plf
#SBATCH --partition=gpu-exp
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --account=<your_allocation>
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

# Long, resumable full run.  --resume picks up outputs/last.pt if it exists, so a
# requeue / restart continues instead of starting over.
# usage:  sbatch run_train.slurm [extra train.py args...]
set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu

cd "<REPO_ROOT>/M14_new_setup_soundbar_melgen"
python -u train.py --resume "$@"
# render samples from the best checkpoint when training finishes
python -u viz.py --ckpt outputs/best.pt --n 8 --out outputs/plf_samples.png || true
