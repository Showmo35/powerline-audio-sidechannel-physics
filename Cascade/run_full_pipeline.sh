#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Full Retrain Pipeline with Consistent Temporal Splits
# ═══════════════════════════════════════════════════════════════════════════
#
# Dependency graph:
#
#   prep_unet ──────→ train_unet ──┐
#                                  ├──→ train_cascade ──→ inference_all
#   prep_e2e  ──────→ train_e2e  ──┘
#
# Both prep scripts use --split temporal so that the last 10% of each
# chapter is always validation, regardless of window size (1s vs 5s).
# This eliminates data leakage between UNet and E2E/Cascade.
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

CASCADE_DIR="<REPO_ROOT>/Cascade"
cd "$CASCADE_DIR"

echo "═══════════════════════════════════════════════════════════════"
echo "  Full Retrain Pipeline (temporal splits)"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Step 1: Data preparation (parallel)
prep_unet=$(sbatch run_prep_unet.sh | awk '{print $4}')
prep_e2e=$(sbatch run_prep_e2e.sh | awk '{print $4}')

# Step 2: Training (depends on respective prep jobs)
train_unet=$(sbatch --dependency=afterok:${prep_unet} run_train_unet.sh | awk '{print $4}')
train_e2e=$(sbatch --dependency=afterok:${prep_e2e} run_train_e2e_baseline.sh | awk '{print $4}')

# Step 3: Cascade training (depends on both UNet + E2E data being ready)
train_cascade=$(sbatch --dependency=afterok:${train_unet}:${prep_e2e} run_cascade_train.sh | awk '{print $4}')

# Step 4: Inference comparison (depends on E2E baseline + cascade)
infer_all=$(sbatch --dependency=afterok:${train_e2e}:${train_cascade} run_inference_all.sh | awk '{print $4}')

echo "Submitted jobs:"
echo ""
echo "  Step 1 - Data Prep (parallel):"
echo "    prep_unet      : ${prep_unet}"
echo "    prep_e2e       : ${prep_e2e}"
echo ""
echo "  Step 2 - Training (parallel, after respective prep):"
echo "    train_unet     : ${train_unet}  (afterok:${prep_unet})"
echo "    train_e2e_ctc  : ${train_e2e}   (afterok:${prep_e2e})"
echo ""
echo "  Step 3 - Cascade Training (after UNet trained + E2E data ready):"
echo "    train_cascade  : ${train_cascade}  (afterok:${train_unet}:${prep_e2e})"
echo ""
echo "  Step 4 - Inference Comparison (after both trainings done):"
echo "    infer_all      : ${infer_all}  (afterok:${train_e2e}:${train_cascade})"
echo ""
echo "Monitor with:  squeue -u \$USER"
echo "═══════════════════════════════════════════════════════════════"
