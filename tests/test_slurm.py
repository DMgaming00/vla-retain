"""The launch layer: dry-runs on any machine, and a real submission path against a fake sbatch."""

import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ["launch_stage1.sh", "launch_pilots.sh", "launch_tuning.sh", "launch_finals.sh"]
TODO_VARS = ["VR_PARTITION", "VR_ACCOUNT", "VR_GPU_TYPE", "VR_SCRATCH", "VR_CONDA_ENV"]


def _env(**extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("VR_")}
    bin_dir = str(Path(sys.executable).parent)
    env.update(PYTHON=sys.executable, PATH=f"{bin_dir}:{env['PATH']}", **extra)
    return env


def _run(script, *args, env=None):
    return subprocess.run(
        ["bash", f"slurm/{script}", *args], cwd=ROOT, env=env or _env(), capture_output=True, text=True, timeout=300
    )


def _sbatch_lines(result):
    return [line for line in result.stderr.splitlines() if line.startswith("sbatch ")]


@pytest.fixture
def fake_cluster(tmp_path):
    """A fake sbatch that logs each call and returns increasing job ids, plus filled-in settings."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "sbatch.log"
    sbatch = bin_dir / "sbatch"
    sbatch.write_text(
        f'#!/bin/bash\necho "$@" >> {log}\nwc -l < {log} | tr -d " " | sed "s/^/1000/"\n'
    )
    sbatch.chmod(sbatch.stat().st_mode | stat.S_IEXEC)
    env = _env(
        VR_PARTITION="gpu", VR_ACCOUNT="acct", VR_GPU_TYPE="h100", VR_SCRATCH=str(tmp_path / "scratch"),
        VR_CONDA_ENV="vla", VR_CHAIN_LINKS="3",
    )  # fmt: skip
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    return env, log


@pytest.mark.parametrize("path", sorted(ROOT.glob("slurm/*")), ids=lambda p: p.name)
def test_shell_syntax(path):
    assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0


def test_every_todo_is_at_the_top_of_common():
    text = (ROOT / "slurm/common.sh").read_text()
    for name in TODO_VARS:
        assert re.search(rf'^{name}="\$\{{{name}:-TODO_', text, re.M), name
    assert "MUJOCO_GL=egl" in text


@pytest.mark.parametrize("script", SCRIPTS + ["chain.sh", "launch_evals.sh"])
def test_real_submission_is_refused_while_todos_remain(script):
    result = _run(script)
    assert result.returncode == 2 and "TODO" in result.stderr


def test_stage1_dry_run_prints_commands_without_submitting():
    result = _run("launch_stage1.sh", "--dry-run")
    assert result.returncode == 0, result.stderr
    lines = _sbatch_lines(result)
    trains = [line for line in lines if "slurm/train.sbatch" in line]
    assert len(trains) == 6
    assert all("configs/stage1.yaml" in line for line in trains)
    assert all("--dependency=afterany:DRY" in line for line in trains[1:])
    evals = [line for line in lines if "eval_array.sbatch" in line]
    assert len(evals) == 1 and "--steps 40000" in evals[0] and "--eval-set validation" in evals[0]
    assert "--array=0-3" in evals[0]
    assert "--dependency=afterany:" in evals[0]


def test_pilots_dry_run_uses_the_tuning_run_ids():
    result = _run("launch_pilots.sh", "--dry-run")
    assert result.returncode == 0, result.stderr
    text = result.stderr
    assert "configs/methods/sft_full.yaml --trial 1" in text and "configs/methods/replay.yaml --trial 0" in text
    assert text.count("slurm/train.sbatch") == 12


def test_tuning_dry_run_covers_every_trial_and_merge_sweeps():
    result = _run("launch_tuning.sh", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("slurm/train.sbatch") == 15 * 6
    assert result.stderr.count("--eval-set validation") == 15
    assert result.stderr.count("scripts/drift.py") == 15
    assert "vr-merge-launcher" in result.stderr
    merge = _run("launch_tuning.sh", "--dry-run", "--phase", "merge")
    assert merge.returncode == 0, merge.stderr
    assert "--array=0-10" in merge.stderr and "--array=0-15" in merge.stderr
    assert merge.stderr.count("--select --with-drift") == 2


def test_finals_dry_run_with_a_plan(tmp_path):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "s1_run": "x", "seeds": [1, 2, 3], "unresolved": [],
        "trained": [{"method": "sft_full", "config": "methods/sft_full.yaml", "trial": 1, "tuning_run": "t", "score": 0}],
        "merges": [{"grid": "module", "alpha_spec": "expert=1.00,lm=0.50", "score": 0}],
    }))  # fmt: skip
    result = _run("launch_finals.sh", "--dry-run", "--plan", str(plan))
    assert result.returncode == 0, result.stderr
    assert result.stderr.count("slurm/train.sbatch") == 3 * 6
    for seed in (1, 2, 3):
        assert f"seed={seed}" in result.stderr
    merges = [line for line in _sbatch_lines(result) if "--alpha" in line]
    assert len(merges) == 3
    assert all("expert=1.00,lm=0.50" in m and "--dependency=afterany:" in m for m in merges)


def test_finals_dry_run_without_results_explains_itself():
    result = _run("launch_finals.sh", "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "Not resolved" in result.stderr or "No plan" in result.stderr


def test_launch_evals_dry_run_and_usage():
    result = _run("launch_evals.sh", "--dry-run", "--run-id", "abc", "--steps", "5000", "--sets", "target")
    assert result.returncode == 0, result.stderr
    assert "--run-id abc" in result.stderr and "--eval-set target" in result.stderr
    assert _run("launch_evals.sh", "--dry-run").returncode == 2


def test_chain_dry_run():
    result = _run("chain.sh", "--dry-run", "--config", "configs/base.yaml", "--config", "configs/methods/lora.yaml",
                  "--set", "s1_run=abc", "--after", "77")  # fmt: skip
    assert result.returncode == 0, result.stderr
    lines = _sbatch_lines(result)
    assert len(lines) == 6 and "--dependency=afterany:77" in lines[0]


def test_real_path_threads_job_ids_through_dependencies(fake_cluster):
    env, log = fake_cluster
    result = _run("launch_stage1.sh", env=env)
    assert result.returncode == 0, result.stderr
    calls = log.read_text().splitlines()
    trains = [c for c in calls if "slurm/train.sbatch" in c]
    assert len(trains) == 3 and "--dependency" not in trains[0]
    first = calls.index(trains[0]) + 1
    assert f"--dependency=afterany:1000{first}" in trains[1]
    eval_call = next(c for c in calls if "eval_array.sbatch" in c)
    assert f"--dependency=afterany:1000{calls.index(trains[-1]) + 1}" in eval_call
    assert "--gres=gpu:h100:1" in trains[0] and "--partition=gpu" in trains[0]
    assert "paths.run_root=" in trains[0]


def test_finished_run_is_not_resubmitted(fake_cluster, tmp_path):
    env, log = fake_cluster
    args = ["--config", "configs/base.yaml", "--config", "configs/stage1.yaml"]
    s1 = subprocess.run([sys.executable, "scripts/train.py", *args, "--print-run-id"], cwd=ROOT, env=env,
                        capture_output=True, text=True).stdout.strip().splitlines()[-1]  # fmt: skip
    run_dir = Path(env["VR_SCRATCH"]) / "vla-retain" / "runs" / s1
    run_dir.mkdir(parents=True)
    (run_dir / "DONE").write_text("{}")
    result = _run("launch_stage1.sh", env=env)
    assert result.returncode == 0, result.stderr
    assert "already done" in result.stderr
    calls = log.read_text().splitlines()
    assert not any("slurm/train.sbatch" in c for c in calls)
    assert any("eval_array.sbatch" in c for c in calls)


def test_train_sbatch_handles_preemption_as_specified():
    text = (ROOT / "slurm/train.sbatch").read_text()
    assert "#SBATCH --signal=B:USR1@300" in text and "trap" in text and "kill -USR1" in text


def test_train_sbatch_forwards_the_signal_to_a_running_child(tmp_path):
    """Run the wrapper's signal logic for real against a child that records what it received."""
    child = tmp_path / "child.py"
    marker = tmp_path / "got"
    child.write_text(
        "import signal, time, pathlib\n"
        f"p = pathlib.Path({str(marker)!r})\n"
        "signal.signal(signal.SIGUSR1, lambda *a: (p.write_text('usr1'), exit(0)))\n"
        "pathlib.Path(str(p) + '.ready').touch()\n"
        "time.sleep(30)\n"
    )
    text = (ROOT / "slurm/train.sbatch").read_text()
    body = text[text.index('"$PYTHON" scripts/train.py "$@" &') :].replace(
        '"$PYTHON" scripts/train.py "$@" &', f'"{sys.executable}" "{child}" &'
    )  # fmt: skip
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text("#!/bin/bash\n" + body)
    proc = subprocess.Popen(["bash", str(wrapper)])
    for _ in range(100):
        if Path(f"{marker}.ready").exists():
            break
        time.sleep(0.1)
    proc.send_signal(signal.SIGUSR1)
    assert proc.wait(timeout=20) == 0
    assert marker.read_text() == "usr1"
