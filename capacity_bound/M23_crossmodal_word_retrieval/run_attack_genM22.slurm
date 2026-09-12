#!/bin/bash
#SBATCH --job-name=m23_atk_genM22
#SBATCH --account=<your_allocation>
#SBATCH --partition=quad
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=atk_genM22_%j.out
#SBATCH --error=atk_genM22_%j.err

# THE M23 sentence-reconstruction attack, re-run with the M22 fresh open-vocab
# retrieval-optimized generator (M22/outputs/best.pt) plugged in as the query
# front-end instead of M14. Same M23 pipeline (attack_build.py + attack_llm.py);
# only the generator changes (M23_GEN=m22). Outputs -> data_attack_genM22/ and
# results_attack_genM22/ so the M14-generator baseline stays intact for comparison.
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate $HOME/miniconda3/envs/tf_gpu
cd "<REPO_ROOT>/capacity_bound/M23_crossmodal_word_retrieval"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export M23_GEN=m22
set -e
echo "[m23-attack/genM22] stage 1: build gallery + sentence lattices (M22 generator, GPU)"
python -u attack_build.py
echo "[m23-attack/genM22] stage 2: LLM reconstruction + lift (GPU)"
python -u attack_llm.py
echo "[m23-attack/genM22] done"
