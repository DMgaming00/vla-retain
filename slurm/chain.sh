#!/bin/bash
# Queue a resumable training chain by hand. Each link depends on the previous one with afterany, so a preempted or
# timed-out link is picked up by the next. Links exit immediately once the run's DONE marker exists.
# If a run ever exhausts its links without finishing, run the same command again.
#
# Usage: slurm/chain.sh [--dry-run] [--after JOBID] <train.py args...>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source slurm/common.sh
vr_parse_common "$@"
vr_require_configured

if [ "${#VR_REST[@]}" -eq 0 ]; then
    echo "Usage: slurm/chain.sh [--dry-run] [--after JOBID] --config FILE [--config FILE] [--set KEY=VALUE]" >&2
    exit 2
fi
run_id=$(vr_run_id "${VR_REST[@]}")
vr_chain "vr-$run_id" "$VR_AFTER" "${VR_REST[@]}"
