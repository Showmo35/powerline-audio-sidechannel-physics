#!/bin/bash
#SBATCH --job-name=m22_monitor
#SBATCH --account=<your_allocation>
#SBATCH --clusters=<cluster_name>
#SBATCH --partition=quad
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=m22mon_%j.out
#SBATCH --error=m22mon_%j.err

# M22 retrieval generator trained on the MONITOR capture (monitor_bin_captures, .lag
# written by Capture_Analysis/check_monitor_align.py --apply). Separate output dir so
# the soundbar model (outputs/) is untouched. --resume continues on requeue.
set -e
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu

export PLF_BIN_DIR="<REPO_ROOT>/Powerline_Data_Captures/monitor_bin_captures"
export PLF_OUT_DIR="<REPO_ROOT>/M22_retrieval_generator/outputs_monitor"
mkdir -p "$PLF_OUT_DIR"

cd "<REPO_ROOT>/M22_retrieval_generator"
echo "[m22-monitor] BIN_DIR=$PLF_BIN_DIR"
echo "[m22-monitor] OUT_DIR=$PLF_OUT_DIR"
python -u train_retrieval.py --resume "$@"
