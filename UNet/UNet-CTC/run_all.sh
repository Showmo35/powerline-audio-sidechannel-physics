#!/bin/bash
set -euo pipefail

DIR="<REPO_ROOT>/UNet/transcribe"
cd "$DIR"

echo "═══════════════════════════════════════════════════════════"
echo "  UNet Transcribe Pipeline: prep → train → inference"
echo "═══════════════════════════════════════════════════════════"

# Step 1: Data preparation (runs UNet + Whisper)
prep=$(sbatch run_prep.sh | awk '{print $4}')

# Step 2: CTC training (after prep)
train=$(sbatch --dependency=afterok:${prep} run_train.sh | awk '{print $4}')

# Step 3: Inference (after training)
infer=$(sbatch --dependency=afterok:${train} run_inference.sh | awk '{print $4}')

echo ""
echo "  prep      : ${prep}"
echo "  train     : ${train}  (afterok:${prep})"
echo "  inference : ${infer}  (afterok:${train})"
echo ""
echo "Monitor: squeue -u \$USER"
echo "═══════════════════════════════════════════════════════════"
