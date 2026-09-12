#!/bin/bash
# Submit the full M13 sweep: all five architectures, one GPU job each.
# Usage:  bash run_all.sh [extra train.py args...]
#   e.g.  bash run_all.sh --epochs 30 --batch 8
set -e
cd "$(dirname "$0")"
for ARCH in m3 m4 m5 m6 m7; do
    jid=$(sbatch --parsable run_train.slurm "$ARCH" "$@")
    echo "submitted $ARCH -> job $jid"
done
echo "all submitted. collect with:  python collect_results.py"
