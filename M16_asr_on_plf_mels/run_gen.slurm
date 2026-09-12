#!/bin/bash
#SBATCH --job-name=m16_genmels
#SBATCH --account=<your_allocation>
#SBATCH --partition=nextgen
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu

cd "<REPO_ROOT>/M16_asr_on_plf_mels"
# m14_best_snapshot.pt = frozen copy of M14's mature EMA (last.pt @ ~79k).
# NOT outputs/best.pt — that is best-by-mel-L1, which stalled at step 2000 (L1 favours blur).
python gen_mels.py --ckpt m14_best_snapshot.pt "$@"
