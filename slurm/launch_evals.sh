#!/bin/bash
# Queue evals for a run by hand. Safe to repeat: finished work is skipped, so this also fills gaps after failures.
# Usage: slurm/launch_evals.sh [--dry-run] [--after JOBID] --run-id ID --steps 5000,20000 [--sets retention,target] [--alpha SPEC]
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/common.sh
vr_parse_common "$@"
vr_require_configured

run_id="" steps="" sets="retention,target,robustness,language" alpha=""
set -- ${VR_REST[@]+"${VR_REST[@]}"}
while [ "$#" -gt 0 ]; do
    case "$1" in
        --run-id) run_id="$2"; shift ;;
        --steps) steps="$2"; shift ;;
        --sets) sets="$2"; shift ;;
        --alpha) alpha="$2"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done
if [ -z "$run_id" ] || [ -z "$steps" ]; then
    echo "Usage: slurm/launch_evals.sh [--dry-run] [--after JOBID] --run-id ID --steps 5000,20000 [--sets a,b] [--alpha SPEC]" >&2
    exit 2
fi
extra=()
if [ -n "$alpha" ]; then
    extra=(--alpha "$alpha")
fi
vr_eval "vr-eval-$run_id" "$VR_AFTER" "$run_id" "$steps" "$sets" ${extra[@]+"${extra[@]}"}
