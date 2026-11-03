#!/bin/bash
# One link of a training chain. Arguments go to scripts/train.py.
# SLURM signals this shell 300 s before the time limit. We forward it, and Python saves and exits cleanly.
#SBATCH --signal=B:USR1@300
#SBATCH --no-requeue

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
source slurm/common.sh
vr_activate

if [ "$("$PYTHON" scripts/train.py "$@" --print-done | tail -1)" = "1" ]; then
    echo "DONE marker found; this link has nothing to do."
    exit 0
fi

"$PYTHON" scripts/train.py "$@" &
child=$!
trap 'echo "Forwarding signal to $child"; kill -USR1 "$child" 2>/dev/null' USR1 TERM

# `wait` returns early when a trap fires, so we keep waiting until the process has really gone,
# then ask once more for its own exit status rather than the interrupted wait's.
while kill -0 "$child" 2>/dev/null; do
    wait "$child" || true
done
wait "$child"
exit $?
