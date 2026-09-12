#!/bin/bash
set -euo pipefail

EXP_DIR="<REPO_ROOT>/Experiments"
cd "$EXP_DIR"

echo "═══════════════════════════════════════════════════════════"
echo "  Submitting 3 experiments (parallel) + inference"
echo "═══════════════════════════════════════════════════════════"

# All 3 experiments run in parallel
exp1=$(sbatch run_exp1.sh | awk '{print $4}')
exp2=$(sbatch run_exp2.sh | awk '{print $4}')
exp3=$(sbatch run_exp3.sh | awk '{print $4}')

# Inference after all 3 finish
infer=$(sbatch --dependency=afterok:${exp1}:${exp2}:${exp3} run_inference.sh | awk '{print $4}')

echo ""
echo "  exp1 (pure CTC)       : ${exp1}"
echo "  exp2 (CTC-weighted)   : ${exp2}"
echo "  exp3 (larger CTC)     : ${exp3}"
echo "  inference (comparison) : ${infer}  (afterok:${exp1}:${exp2}:${exp3})"
echo ""
echo "Monitor: squeue -u \$USER"
echo "═══════════════════════════════════════════════════════════"
