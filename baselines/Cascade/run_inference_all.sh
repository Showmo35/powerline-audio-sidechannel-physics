#!/bin/bash
#SBATCH --job-name=Infer_All
#SBATCH --account=<your_allocation>
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=24GB
#SBATCH --output=Infer_All_%j.out
#SBATCH --error=Infer_All_%j.err

set -euo pipefail

module load miniconda3/24.1.2-py310
source /users/<allocation>/user/miniconda3/etc/profile.d/conda.sh
conda activate tf_gpu

echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "======================================================================"

PROJECT="<REPO_ROOT>"
E2E_DATA="$PROJECT/E2E/data/e2e_data.npz"

if [[ ! -f "$E2E_DATA" ]]; then
  echo "Missing data file: $E2E_DATA"
  exit 1
fi

# ── 1. E2E CTC baseline inference ──────────────────────────────────────────
echo ""
echo ">>> E2E CTC Baseline Inference"
echo "----------------------------------------------------------------------"

cd "$PROJECT/E2E"
python3 -u inference_e2e.py \
    --data "$E2E_DATA" \
    --ckpt "$PROJECT/E2E/checkpoints_ctc/best_model.pt" \
    --out "$PROJECT/E2E/inference_output_ctc" \
    --mode ctc \
    --n_samples 5 \
    --sample_mode first \
    --batch 2 \
    --device cuda

# ── 2. Cascade inference ───────────────────────────────────────────────────
echo ""
echo ">>> Cascade Inference"
echo "----------------------------------------------------------------------"

cd "$PROJECT/Cascade"
python3 -u inference_cascade.py \
    --data "$E2E_DATA" \
    --ckpt "$PROJECT/Cascade/checkpoints/best_model.pt" \
    --out "$PROJECT/Cascade/inference_output" \
    --n_samples 5 \
    --sample_mode first \
    --batch 2 \
    --device cuda

# ── 3. Print comparison ───────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  RESULTS COMPARISON"
echo "======================================================================"
echo ""
echo "--- E2E CTC Baseline ---"
cat "$PROJECT/E2E/inference_output_ctc/summary.json" 2>/dev/null || echo "(no summary)"
echo ""
echo "--- Cascade ---"
cat "$PROJECT/Cascade/inference_output/summary.json" 2>/dev/null || echo "(no summary)"
echo ""
echo "======================================================================"
echo "End Time: $(date)"
echo "======================================================================"
