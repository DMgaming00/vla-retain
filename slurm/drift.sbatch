#!/bin/bash
# A single GPU job for drift, for probe building, and for merge selection. Usage: drift.sbatch <python args...>
# Requeueing is safe: drift rows are appended atomically per checkpoint, and finished checkpoints are skipped.
#SBATCH --requeue

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
source slurm/common.sh
vr_activate

"$PYTHON" "$@"
