#!/bin/bash
# All tuning trials with validation evals and drift, then, once those finish, merge sweeps on the selected M1 run.
# Usage: slurm/launch_tuning.sh [--dry-run] [--after JOBID] [--phase all|trials|merge]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/common.sh
vr_parse_common "$@"
vr_require_configured

phase=all
if [ "${#VR_REST[@]}" -ge 2 ] && [ "${VR_REST[0]}" = "--phase" ]; then
    phase="${VR_REST[1]}"
fi
s1=$(vr_run_id --config configs/base.yaml --config configs/stage1.yaml)
METHODS="sft_expert sft_full lora replay l2sp"

launch_trials() {
    local waits=() method trial n args run_id train
    for method in $METHODS; do
        args=(--config configs/base.yaml --config "configs/methods/$method.yaml" --set "s1_run=$s1")
        n=$("$PYTHON" scripts/train.py "${args[@]}" --print-n-trials | tail -1)
        for ((trial = 0; trial < n; trial++)); do
            run_id=$(vr_run_id "${args[@]}" --trial "$trial")
            train=$(vr_chain "vr-$method-t$trial" "$VR_AFTER" "${args[@]}" --trial "$trial")
            waits+=("$(vr_eval "vr-$method-t$trial-val" "$train" "$run_id" 5000,20000 validation)")
            waits+=("$(vr_drift "vr-$method-t$trial-drift" "$train" "$run_id" 5000,20000)")
        done
    done
    vr_join "${waits[@]}"
}

launch_merges() {
    local after="$1" best grid n sweep
    best=$("$PYTHON" scripts/plan_finals.py --best sft_full 2>/dev/null || echo BEST_M1_RUN)
    echo "merging the M1 run: $best" >&2
    for grid in global module; do
        n=$("$PYTHON" scripts/merge_sweep.py --run-id "$best" --grid "$grid" --list | tail -1)
        sweep=$(vr_sbatch "vr-merge-$grid" gpu "$after" "--time=$VR_EVAL_TIME" "--array=0-$((n - 1))" -- \
            slurm/eval_array.sbatch scripts/merge_sweep.py --run-id "$best" --grid "$grid" --alpha-index '{TASK}' \
            --run-root "$VR_RUN_ROOT" --n-envs "$VR_N_ENVS" --n-action-steps "$VR_N_ACTION_STEPS")
        vr_sbatch "vr-merge-$grid-select" gpu "$sweep" "--time=$VR_EVAL_TIME" -- slurm/drift.sbatch \
            scripts/merge_sweep.py --run-id "$best" --grid "$grid" --select --with-drift --run-root "$VR_RUN_ROOT" >/dev/null
    done
}

case "$phase" in
    trials) launch_trials >/dev/null ;;
    merge) launch_merges "$VR_AFTER" ;;
    all)
        waits=$(launch_trials)
        # The best M1 trial is known only after its validation evals, so this step runs as its own small job.
        vr_sbatch vr-merge-launcher cpu "$waits" --time=00:15:00 -- slurm/drift.sbatch -c \
            "import subprocess, sys; sys.exit(subprocess.call(['bash', 'slurm/launch_tuning.sh', '--phase', 'merge']))" >/dev/null
        ;;
    *) echo "Unknown phase: $phase" >&2; exit 2 ;;
esac
