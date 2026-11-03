#!/bin/bash
# The pilots for Gate 1: M1 and M3 with the 2% pool, one seed each, then their evals.
# These configs are tuning trials, so launch_tuning.sh will reuse these runs rather than repeat them.
# Usage: slurm/launch_pilots.sh [--dry-run] [--after JOBID]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/common.sh
vr_parse_common "$@"
vr_require_configured

s1=$(vr_run_id --config configs/base.yaml --config configs/stage1.yaml)

launch_pilot() {
    local config="$1" trial="$2" label="$3"
    local args=(--config configs/base.yaml --config "configs/methods/$config.yaml" --trial "$trial" --set "s1_run=$s1")
    local run_id job
    run_id=$(vr_run_id "${args[@]}")
    echo "$label run id: $run_id" >&2
    job=$(vr_chain "vr-$label-$run_id" "$VR_AFTER" "${args[@]}")
    vr_eval "vr-$label-eval" "$job" "$run_id" 20000 retention,target
    vr_eval "vr-$label-val" "$job" "$run_id" 5000,20000 validation
    vr_drift "vr-$label-drift" "$job" "$run_id" 5000,20000
}

launch_pilot sft_full 1 m1
launch_pilot replay 0 m3
