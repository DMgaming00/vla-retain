#!/bin/bash
# All cluster settings live here. Replace every TODO value before submitting; --dry-run works without them.
# Launch scripts source this file. Any value can also be set from the environment.

# ----- TODO: fill these in -----------------------------------------------------------------------------------
VR_PARTITION="${VR_PARTITION:-TODO_PARTITION}"      # scontrol show partition
VR_ACCOUNT="${VR_ACCOUNT:-TODO_ACCOUNT}"            # sacctmgr show assoc user=$USER format=account
VR_GPU_TYPE="${VR_GPU_TYPE:-TODO_GPU_TYPE}"         # for example h100 or a100; see: sinfo -o "%P %G"
VR_SCRATCH="${VR_SCRATCH:-TODO_SCRATCH_PATH}"       # checkpoints, videos, and the HF cache go here
VR_CONDA_ENV="${VR_CONDA_ENV:-TODO_CONDA_ENV}"      # the environment from docs/RUNBOOK.md
# ---------------------------------------------------------------------------------------------------------------

# Defaults that usually need no change.
VR_TIME="${VR_TIME:-48:00:00}"
VR_EVAL_TIME="${VR_EVAL_TIME:-24:00:00}"
VR_CPUS="${VR_CPUS:-16}"
VR_MEM="${VR_MEM:-64G}"
VR_CHAIN_LINKS="${VR_CHAIN_LINKS:-6}"               # queued links per training run
VR_EVAL_SHARDS="${VR_EVAL_SHARDS:-4}"               # array tasks per eval
VR_N_ENVS="${VR_N_ENVS:-8}"                         # set from the smoke test's timing table
VR_N_ACTION_STEPS="${VR_N_ACTION_STEPS:-10}"
VR_QOS="${VR_QOS:-}"                                # optional

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false

VR_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VR_BASE="$VR_SCRATCH/vla-retain"
VR_RUN_ROOT="$VR_BASE/runs"
VR_LOG_DIR="$VR_BASE/logs"
VR_VIDEO_DIR="$VR_BASE/videos"
export HF_HOME="$VR_SCRATCH/huggingface"
export HF_LEROBOT_HOME="$VR_SCRATCH/huggingface/lerobot"
PYTHON="${PYTHON:-python}"
VR_DRY_RUN="${VR_DRY_RUN:-0}"

vr_is_configured() {
    case "$VR_PARTITION $VR_ACCOUNT $VR_GPU_TYPE $VR_SCRATCH $VR_CONDA_ENV" in
        *TODO_*) return 1 ;;
        *) return 0 ;;
    esac
}

vr_require_configured() {
    if [ "$VR_DRY_RUN" = "0" ] && ! vr_is_configured; then
        echo "slurm/common.sh still has TODO values. Fill them in, or pass --dry-run." >&2
        exit 2
    fi
    if [ "$VR_DRY_RUN" = "0" ]; then
        mkdir -p "$VR_RUN_ROOT" "$VR_LOG_DIR" "$VR_VIDEO_DIR"
    fi
}

vr_activate() {
    # Called inside jobs. Clusters differ in how conda is initialized, so we try the usual routes.
    if ! command -v conda >/dev/null 2>&1; then
        module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true
    fi
    eval "$(conda shell.bash hook)"
    conda activate "$VR_CONDA_ENV"
}

# Dry-run job ids come from a counter file, because vr_sbatch runs inside $(...).
VR_DRY_COUNTER="${VR_DRY_COUNTER:-$(mktemp "${TMPDIR:-/tmp}/vr_dry.XXXXXX")}"
export VR_DRY_COUNTER

vr_quote() {
    local out="" arg
    for arg in "$@"; do
        case "$arg" in
            *[!A-Za-z0-9_./:=,@%+-]*|"") out="$out '$arg'" ;;
            *) out="$out $arg" ;;
        esac
    done
    printf '%s' "${out# }"
}

# Usage: vr_sbatch <job name> <gpu|cpu> <dependency job ids, comma separated or empty> [sbatch options...] -- script args...
# Prints the job id on stdout, and the command on stderr.
vr_sbatch() {
    local name="$1" kind="$2" after="$3"
    shift 3
    local -a cmd=(sbatch --parsable "--job-name=$name" "--partition=$VR_PARTITION" "--account=$VR_ACCOUNT"
        "--cpus-per-task=$VR_CPUS" "--mem=$VR_MEM" "--output=$VR_LOG_DIR/%x-%A_%a.out")
    if [ "$kind" = "gpu" ]; then
        cmd+=("--gres=gpu:$VR_GPU_TYPE:1")
    fi
    if [ -n "$VR_QOS" ]; then
        cmd+=("--qos=$VR_QOS")
    fi
    if [ -n "$after" ]; then
        cmd+=("--dependency=afterany:${after//,/:}")
    fi
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do
        cmd+=("$1")
        shift
    done
    [ "${1:-}" = "--" ] && shift
    cmd+=("$@")
    echo "$(vr_quote "${cmd[@]}")" >&2
    if [ "$VR_DRY_RUN" = "1" ]; then
        local n
        n=$(($(wc -l <"$VR_DRY_COUNTER") + 1))
        echo x >>"$VR_DRY_COUNTER"
        echo "DRY$n"
    else
        "${cmd[@]}" | cut -d';' -f1
    fi
}

# Usage: vr_chain <name> <dependency or empty> <train.py args...>
# Queues links joined by afterany. Each link resumes from the latest checkpoint, and exits at once if DONE exists.
# Prints the last link's id, or nothing if the run has already finished.
vr_chain() {
    local name="$1" after="$2"
    shift 2
    local -a args=("$@" --set "paths.run_root=$VR_RUN_ROOT")
    if [ "$("$PYTHON" scripts/train.py "${args[@]}" --print-done 2>/dev/null | tail -1)" = "1" ]; then
        echo "$name is already done" >&2
        return 0
    fi
    local link job="$after"
    for ((link = 1; link <= VR_CHAIN_LINKS; link++)); do
        job=$(vr_sbatch "$name" gpu "$job" "--time=$VR_TIME" -- slurm/train.sbatch "${args[@]}")
    done
    echo "$job"
}

# Usage: vr_eval <name> <dependency or empty> <run id> <steps> <comma-separated eval sets> [extra eval.py args...]
vr_eval() {
    local name="$1" after="$2" run_id="$3" steps="$4" sets="$5"
    shift 5
    local -a set_args=()
    local item
    for item in ${sets//,/ }; do
        set_args+=(--eval-set "$item")
    done
    vr_sbatch "$name" gpu "$after" "--time=$VR_EVAL_TIME" "--array=0-$((VR_EVAL_SHARDS - 1))" -- \
        slurm/eval_array.sbatch scripts/eval.py --run-id "$run_id" --steps "$steps" "${set_args[@]}" \
        --run-root "$VR_RUN_ROOT" --shard '{TASK}' --num-shards "$VR_EVAL_SHARDS" \
        --n-envs "$VR_N_ENVS" --n-action-steps "$VR_N_ACTION_STEPS" "$@"
}

# Usage: vr_drift <name> <dependency or empty> <run id> <steps>
vr_drift() {
    vr_sbatch "$1" gpu "$2" "--time=$VR_EVAL_TIME" -- slurm/drift.sbatch scripts/drift.py \
        --run-id "$3" --steps "$4" --run-root "$VR_RUN_ROOT"
}

# Usage: vr_run_id <train.py args...>
vr_run_id() {
    "$PYTHON" scripts/train.py "$@" --print-run-id | tail -1
}

vr_join() {
    local IFS=,
    echo "$*"
}

# Shared argument handling. Sets VR_DRY_RUN and VR_AFTER, and leaves the rest in VR_REST.
vr_parse_common() {
    VR_AFTER=""
    VR_REST=()
    while [ "$#" -gt 0 ]; do
        case "$1" in
            --dry-run) VR_DRY_RUN=1 ;;
            --after) VR_AFTER="$2"; shift ;;
            *) VR_REST+=("$1") ;;
        esac
        shift
    done
}
