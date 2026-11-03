#!/bin/bash
# One shard of an eval array. Usage: eval_array.sbatch <script> <args...>, with {TASK} replaced by the array index.
# Requeueing is safe: results are appended per task, and finished tasks are skipped on the next attempt.
#SBATCH --requeue

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
source slurm/common.sh
vr_activate

args=()
for arg in "$@"; do
    args+=("${arg//\{TASK\}/${SLURM_ARRAY_TASK_ID:-0}}")
done
"$PYTHON" "${args[@]}"
