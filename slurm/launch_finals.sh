#!/bin/bash
# Pick each method's best trial with scripts/plan_finals.py, then launch 3 seeds, full evals, drift, and merges.
# Usage: slurm/launch_finals.sh [--dry-run] [--after JOBID] [--plan FILE]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/common.sh
vr_parse_common "$@"
vr_require_configured

plan=plans/finals.json
if [ "${#VR_REST[@]}" -ge 2 ] && [ "${VR_REST[0]}" = "--plan" ]; then
    plan="${VR_REST[1]}"
elif [ "$VR_DRY_RUN" = "1" ]; then
    "$PYTHON" scripts/plan_finals.py --dry-run --out "$plan" >&2 || true
else
    "$PYTHON" scripts/plan_finals.py --out "$plan" >&2 || {
        echo "Some results are missing; plan_finals reported them above. Nothing was submitted." >&2
        exit 2
    }
fi
if [ ! -f "$plan" ]; then
    echo "No plan at $plan yet. Finish tuning, then run this again (or pass --plan FILE to preview one)." >&2
    exit 0
fi

s1=$(vr_run_id --config configs/base.yaml --config configs/stage1.yaml)
STEPS=2000,5000,10000,20000

"$PYTHON" scripts/plan_finals.py --list --out "$plan" | while read -r kind a b seed extra; do
    if [ "$kind" = "train" ]; then
        args=(--config configs/base.yaml --config "configs/$a" --trial "$b" --set "s1_run=$s1" --set "seed=$seed")
        run_id=$(vr_run_id "${args[@]}")
        train=$(vr_chain "vr-final-$run_id" "$VR_AFTER" "${args[@]}")
        vr_eval "vr-final-$run_id-eval" "$train" "$run_id" "$STEPS" retention,target,robustness,language \
            --video-dir "$VR_VIDEO_DIR" >/dev/null
        vr_drift "vr-final-$run_id-drift" "$train" "$run_id" "$STEPS" >/dev/null
        if [ "$a" = "methods/sft_full.yaml" ]; then
            echo "$train" >"$VR_DRY_COUNTER.m1_$seed"
        fi
    else
        args=(--config configs/base.yaml --config configs/methods/sft_full.yaml --trial "$extra" --set "s1_run=$s1" --set "seed=$seed")
        run_id=$(vr_run_id "${args[@]}")
        after=$(cat "$VR_DRY_COUNTER.m1_$seed" 2>/dev/null || true)
        vr_sbatch "vr-final-merge-$a" gpu "$after" "--time=$VR_EVAL_TIME" -- slurm/drift.sbatch \
            scripts/merge_sweep.py --run-id "$run_id" --grid "$a" --alpha "$b" --with-drift --run-root "$VR_RUN_ROOT" \
            --n-envs "$VR_N_ENVS" --n-action-steps "$VR_N_ACTION_STEPS" >/dev/null
    fi
done
