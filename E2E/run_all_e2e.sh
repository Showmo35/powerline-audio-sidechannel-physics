#!/bin/bash
set -euo pipefail

E2E_DIR="<REPO_ROOT>/E2E"
cd "$E2E_DIR"

echo "Submitting E2E HPC pipeline from: $E2E_DIR"

prep_job=$(sbatch run_e2e_prepare.sh | awk '{print $4}')
ctc_job=$(sbatch --dependency=afterok:${prep_job} run_e2e_ctc.sh | awk '{print $4}')
cond_job=$(sbatch --dependency=afterok:${prep_job} run_e2e_conditioned.sh | awk '{print $4}')
joint_job=$(sbatch --dependency=afterok:${prep_job} run_e2e_joint.sh | awk '{print $4}')

echo ""
echo "Submitted jobs:"
echo "  prepare      : ${prep_job}"
echo "  ctc          : ${ctc_job} (afterok:${prep_job})"
echo "  conditioned  : ${cond_job} (afterok:${prep_job})"
echo "  joint        : ${joint_job} (afterok:${prep_job})"
echo ""
echo "Check status with:"
echo "  squeue -u \$USER"
