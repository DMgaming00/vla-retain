#!/bin/bash
# Train S1, freeze the probe set, and queue S1's evaluation.
# Usage: slurm/launch_stage1.sh [--dry-run] [--after JOBID]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/common.sh
vr_parse_common "$@"
vr_require_configured

args=(--config configs/base.yaml --config configs/stage1.yaml)
s1=$(vr_run_id "${args[@]}")
echo "S1 run id: $s1" >&2

if [ ! -f probes/probe_v1.pt ]; then
    vr_sbatch vr-probe cpu "$VR_AFTER" --time=04:00:00 -- slurm/drift.sbatch -m vlaretain.data.probes >/dev/null
fi
train=$(vr_chain "vr-s1-$s1" "$VR_AFTER" "${args[@]}")
vr_eval "vr-s1-eval" "$train" "$s1" 40000 retention,target,robustness,language,validation --video-dir "$VR_VIDEO_DIR"
