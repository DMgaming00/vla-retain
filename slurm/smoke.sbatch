#!/bin/bash
# The smoke test: EGL, CUDA, and rollout timing at 1, 4, and 8 parallel environments.
#SBATCH --job-name=vr-smoke
#SBATCH --time=02:00:00
#SBATCH --no-requeue

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
source slurm/common.sh
vr_activate

nvidia-smi --query-gpu=name,memory.total --format=csv
"$PYTHON" scripts/smoke.py "$@"
