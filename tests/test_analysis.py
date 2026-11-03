import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vlaretain.analysis.common import checkpoint_table, drift_table, forgetting, load_tables, s1_task_rates
from vlaretain.analysis.correlate import (
    join_tables,
    leave_one_method_out_r2,
    partial_spearman,
    primary_partial,
    spearman_ci,
)
from vlaretain.analysis.figures import table_one, to_latex, to_markdown
from vlaretain.analysis.pareto import at_budget, matched_budget, pareto_front
from vlaretain.analysis.synthetic import generate_rows, write_synthetic
from vlaretain.methods.merge import select_alpha
from vlaretain.utils.io import append_jsonl, make_eval_row

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import aggregate as aggregate_script  # noqa: E402
import check_gate1  # noqa: E402
import check_gate2  # noqa: E402
import plan_finals  # noqa: E402


def _row(run, method, step, eval_set, task, n, k, hours=1.0, alpha="none", seed=0):
    return make_eval_row(
        run_id=run, method=method, seed=seed, step=step, alpha_spec=alpha, eval_set=eval_set, task_id=task,
        n_episodes=n, n_success=k, gpu_hours_train=hours, gpu_hours_eval=0.1, git_sha="x", lerobot_sha="y",
    )  # fmt: skip


# ---------- tables ----------
@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    path = tmp_path_factory.mktemp("syn")
    write_synthetic(path)
    return path


def test_bundled_example_matches_the_generator(tmp_path):
    bundled = (ROOT / "examples/synthetic/results.jsonl").read_text().splitlines()
    fresh = generate_rows(0)
    assert len(bundled) == len(fresh)
    assert all(r["run_id"].startswith("synthetic-") for r in fresh)
    assert all(json.loads(line)["run_id"].startswith("synthetic-") for line in bundled)


def test_checkpoint_table_values_and_forgetting_sign(synthetic):
    eval_df, _ = load_tables(synthetic)
    table = checkpoint_table(eval_df)
    assert len(table) == 5 * 3 * 4 + 2 * 3  # trained methods at 4 steps, merges at 1
    assert table["target_sr"].between(0, 1).all() and table["retention_sr"].between(0, 1).all()
    assert "stage1" not in set(table["method"])
    s1 = pd.Series({"a": 0.9, "b": 0.5})
    run = pd.DataFrame({"task_id": ["a", "b"], "sr": [0.6, 0.5]})
    assert forgetting(run, s1) == pytest.approx(0.15)


def test_partial_evals_are_left_out_not_filled_in():
    rows = [_row("s1", "stage1", 40000, "retention", f"t{i}", 10, 9) for i in range(30)]
    rows += [_row("r", "sft_full", 20000, "retention", f"t{i}", 10, 5) for i in range(29)]
    rows += [_row("r", "sft_full", 20000, "target", f"u{i}", 20, 5) for i in range(10)]
    assert checkpoint_table(pd.DataFrame(rows)).empty
    rows += [_row("r", "sft_full", 20000, "retention", "t29", 10, 5)]
    table = checkpoint_table(pd.DataFrame(rows))
    assert len(table) == 1 and table.iloc[0]["forgetting"] == pytest.approx(0.4)
    assert s1_task_rates(pd.DataFrame(rows)).loc["t0"] == 0.9


def test_merge_compute_includes_its_validation_sweep():
    rows = [_row("s1", "stage1", 40000, "retention", f"t{i}", 10, 9) for i in range(30)]
    alpha = "global=0.50"
    rows += [_row("r", "merge_global", 20000, "retention", f"t{i}", 10, 5, 2.0, alpha) for i in range(30)]
    rows += [_row("r", "merge_global", 20000, "target", f"u{i}", 20, 5, 2.0, alpha) for i in range(10)]
    rows += [_row("r", "merge_global", 20000, "validation_target", f"u{i}", 5, 1, 2.0, alpha) for i in range(10)]
    table = checkpoint_table(pd.DataFrame(rows))
    assert table.iloc[0]["compute_hours"] == pytest.approx(2.0 + 10 * 0.1)


def test_empty_results_give_empty_tables(tmp_path):
    eval_df, drift_df = load_tables(tmp_path)
    assert eval_df.empty and drift_df.empty and checkpoint_table(eval_df).empty and drift_table(drift_df).empty


def test_pareto_front():
    points = pd.DataFrame({"target_sr": [0.1, 0.5, 0.4, 0.9], "retention_sr": [0.9, 0.7, 0.6, 0.2]})
    assert pareto_front(points)["target_sr"].tolist() == [0.1, 0.5, 0.9]


def test_matched_budget_and_selection():
    table = pd.DataFrame({
        "method": ["a", "a", "b", "b"], "run_id": ["x", "x", "y", "y"],
        "compute_hours": [1.0, 3.0, 2.0, 5.0], "step": [1, 2, 1, 2], "target_sr": [0.1, 0.2, 0.3, 0.4],
    })  # fmt: skip
    assert matched_budget(table) == 3.0
    picked = at_budget(table, 3.0)
    assert sorted(picked["compute_hours"].tolist()) == [2.0, 3.0]


# ---------- correlations ----------
def _frame(n_methods=4, per=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(n_methods):
        for r in range(per):
            x = rng.uniform(0, 1)
            rows.append({"method": f"m{m}", "run_id": f"m{m}-{r}", "div": x, "forgetting": 2 * x + rng.normal(0, 0.05)})
    return pd.DataFrame(rows)


def test_spearman_ci_brackets_a_strong_relation():
    result = spearman_ci(_frame(), "div", n_boot=300)
    assert result["rho"] > 0.9 and result["ci_lo"] > 0.8 and result["ci_lo"] <= result["rho"] <= result["ci_hi"]
    assert result["n"] == 40 and result["n_runs"] == 40


def test_spearman_ci_on_noise_includes_zero():
    frame = _frame()
    frame["forgetting"] = np.random.default_rng(1).normal(size=len(frame))
    result = spearman_ci(frame, "div", n_boot=300)
    assert result["ci_lo"] < 0 < result["ci_hi"]


def test_bootstrap_resamples_runs_not_checkpoints():
    frame = _frame(per=1)
    many = pd.concat([frame] * 5)  # each run repeated: more rows, the same number of runs
    assert spearman_ci(many, "div", n_boot=50)["n_runs"] == 4
    assert np.isnan(spearman_ci(frame.head(2), "div", n_boot=10)["ci_lo"])


def test_leave_one_method_out_r2():
    assert leave_one_method_out_r2(_frame(), "div") > 0.9
    frame = _frame()
    frame["forgetting"] = np.random.default_rng(2).normal(size=len(frame))
    assert leave_one_method_out_r2(frame, "div") < 0.2
    assert np.isnan(leave_one_method_out_r2(_frame(n_methods=2), "div"))


def test_partial_correlation_removes_a_confound():
    rng = np.random.default_rng(3)
    z = rng.normal(size=300)
    frame = pd.DataFrame({"x": z + rng.normal(0, 0.1, 300), "y": z + rng.normal(0, 0.1, 300), "z": z})
    assert frame["x"].corr(frame["y"], method="spearman") > 0.9
    assert abs(partial_spearman(frame, "x", "y", ["z"])["rho"]) < 0.15
    direct = pd.DataFrame({"x": z, "y": z + rng.normal(0, 0.1, 300), "z": rng.normal(size=300)})
    assert partial_spearman(direct, "x", "y", ["z"])["rho"] > 0.9
    assert np.isnan(partial_spearman(frame.head(3), "x", "y", ["z"])["rho"])


# ---------- table ----------
def test_table_shows_missing_rather_than_a_number():
    rows = [_row("r", "lora", 20000, "target", f"u{i}", 20, 5) for i in range(3)]
    table = table_one(pd.DataFrame(rows), checkpoint_table(pd.DataFrame(rows)))
    assert table.iloc[0]["Retention SR"] == "missing" and table.iloc[0]["Robust camera"] == "missing"
    assert "missing" in to_markdown(table) and r"\begin{tabular}" in to_latex(table)


def test_merge_selection_picks_the_best_and_reports_gaps():
    def rows(alpha, t, r):
        return [
            _row("x", "merge_global", 20000, "validation_target", "u", 10, t, alpha=alpha),
            _row("x", "merge_global", 20000, "validation_retention", "v", 10, r, alpha=alpha),
        ]

    data = rows("a", 5, 5) + rows("b", 9, 8) + rows("c", 1, 1)
    result = select_alpha(data, "x", 20000, ["a", "b", "c", "d"])
    assert result["best"]["alpha_spec"] == "b" and result["best"]["score"] == pytest.approx(0.85)
    assert result["missing"] == ["d"]
    assert select_alpha([], "x", 20000, ["a"])["best"] is None


# ---------- end to end ----------
def test_aggregate_runs_end_to_end_on_synthetic_jsonl(tmp_path, synthetic):
    code = aggregate_script.main(["--results", str(synthetic), "--out", str(tmp_path / "assets")])
    assert code == 0
    for name in ("fig1_pareto.png", "fig2_forgetting_vs_divergence.png", "fig3_drift.png", "table1.md", "table1.tex",
                 "table1.csv", "correlations.csv", "partial_correlation.txt"):  # fmt: skip
        assert (tmp_path / "assets" / name).stat().st_size > 0, name
    table = pd.read_csv(tmp_path / "assets" / "table1.csv")
    assert len(table) == 7 and not table.isna().any().any()
    assert "missing" not in set(table["Target SR"])


def test_aggregate_on_empty_results_writes_missing_assets(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert aggregate_script.main(["--results", str(empty), "--out", str(tmp_path / "out")]) == 0
    assert (tmp_path / "out/fig1_pareto.png").exists()
    assert not (tmp_path / "out/correlations.csv").exists()


def test_demo_figures_are_labeled_synthetic(tmp_path):
    assert aggregate_script.main(["--demo", "--out", str(tmp_path)]) == 0
    assert "synthetic data" in (tmp_path / "table1.md").read_text()
    with pytest.raises(SystemExit, match="only uses synthetic data"):
        aggregate_script.main(["--demo", "--update-readme", "--out", str(tmp_path)])


def _readme(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{aggregate_script.START}\n{aggregate_script.PLACEHOLDER}\n{aggregate_script.END}\nouttro\n")
    return readme


def test_update_readme_replaces_only_the_block(tmp_path, synthetic):
    readme = _readme(tmp_path)
    code = aggregate_script.main([
        "--results", str(synthetic), "--out", str(tmp_path / "assets"), "--update-readme", "--readme", str(readme),
    ])  # fmt: skip
    assert code == 0
    text = readme.read_text()
    assert text.startswith("intro\n") and text.endswith("outtro\n")
    assert aggregate_script.PLACEHOLDER not in text
    assert "M1 Full SFT" in text and "![Target SR vs retention SR](assets/fig1_pareto.png)" in text
    assert (tmp_path / "assets/fig1_pareto.png").exists()
    # A second run must leave exactly one block.
    aggregate_script.main(["--results", str(synthetic), "--out", str(tmp_path / "assets"), "--update-readme", "--readme", str(readme)])
    assert readme.read_text().count(aggregate_script.START) == 1


def test_update_readme_without_results_keeps_the_placeholder(tmp_path):
    readme = _readme(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    aggregate_script.main(["--results", str(empty), "--out", str(tmp_path / "a"), "--update-readme", "--readme", str(readme)])
    assert aggregate_script.PLACEHOLDER in readme.read_text()


def test_synthetic_rows_are_refused_for_the_project_readme(tmp_path, synthetic):
    with pytest.raises(SystemExit, match="Refusing"):
        aggregate_script.main(["--results", str(synthetic), "--out", str(tmp_path), "--update-readme"])


def test_missing_markers_raise():
    with pytest.raises(ValueError):
        aggregate_script.replace_block("no markers", "x")


# ---------- gates ----------
def _pilot_rows(s1=0.9, m1=0.7, m3=0.8):
    rows = []
    for suite in check_gate1.STAGE1_SUITES:
        for i in range(10):
            task = f"{suite}/{i:02d}"
            rows.append(_row("S", "stage1", 40000, "retention", task, 10, round(s1 * 10)))
            rows.append(_row("A", "sft_full", 20000, "retention", task, 10, round(m1 * 10)))
            rows.append(_row("B", "replay", 20000, "retention", task, 10, round(m3 * 10)))
    return rows


def test_gate1_pass_fail_and_missing():
    status = lambda rows: [c["status"] for c in check_gate1.evaluate(rows, "S", "A", "B", 20000)]  # noqa: E731
    assert status(_pilot_rows()) == ["PASS", "PASS", "PASS"]
    assert status(_pilot_rows(s1=0.7, m1=0.5, m3=0.6))[0] == "FAIL"
    assert status(_pilot_rows(m1=0.85, m3=0.8))[1] == "FAIL"
    assert status(_pilot_rows(m1=0.7, m3=0.6))[2] == "FAIL"
    assert status([]) == ["MISSING", "MISSING", "MISSING"]


def test_gate1_main_prints_fix_and_exit_codes(tmp_path, capsys):
    results = tmp_path / "r"
    append_jsonl(results / "x.jsonl", _pilot_rows(m1=0.85, m3=0.9))
    args = ["--results", str(results), "--s1-run", "S", "--m1-run", "A", "--m3-run", "B", "--configs", str(ROOT / "configs")]
    assert check_gate1.main(args) == 1
    out = capsys.readouterr().out
    assert "GATE 1: FAIL" in out and "Suggested fix" in out
    empty = tmp_path / "e"
    empty.mkdir()
    assert check_gate1.main(["--results", str(empty), "--configs", str(ROOT / "configs")]) == 2


def test_gate2_missing_then_reports(tmp_path, capsys, synthetic):
    empty = tmp_path / "e"
    empty.mkdir()
    assert check_gate2.main(["--results", str(empty)]) == 2
    assert "MISSING" in capsys.readouterr().out
    # The synthetic set has no validation rows, so it should also be missing rather than crash.
    assert check_gate2.main(["--results", str(synthetic)]) == 2


def test_gate2_reports_rho_and_ci_on_validation_rows(tmp_path, capsys):
    rows, _ = [], None
    syn = generate_rows(0)
    for r in syn:
        if r["eval_set"] == "retention":
            r = {**r, "eval_set": "validation_retention"}
        elif r["eval_set"] == "target":
            r = {**r, "eval_set": "validation_target"}
        rows.append(r)
    append_jsonl(tmp_path / "x.jsonl", rows)
    code = check_gate2.main(["--results", str(tmp_path), "--n-boot", "50"])
    assert code in (0, 1)
    assert "New-state divergence vs forgetting: rho=" in capsys.readouterr().out


# ---------- finals plan ----------
def test_plan_finals_picks_best_trial_and_uses_fresh_seeds(tmp_path, capsys):
    root = ROOT / "configs"
    s1 = plan_finals.s1_run_id(root)
    rows = []
    best_run = None
    for trial, k in enumerate([3, 8, 5]):
        from vlaretain.config import run_id_of, trial_config

        run = run_id_of(trial_config([root / "base.yaml", root / "methods/lora.yaml"], trial, [f"s1_run={s1}", "seed=0"]))
        if trial == 1:
            best_run = run
        for eval_set in ("validation_target", "validation_retention"):
            rows.append(_row(run, "lora", 20000, eval_set, "x/00", 10, k))
    append_jsonl(tmp_path / "r" / "a.jsonl", rows)
    out = tmp_path / "plan.json"
    args = ["--results", str(tmp_path / "r"), "--configs", str(root), "--out", str(out), "--methods", "lora"]
    assert plan_finals.main(args) == 0
    plan = json.loads(out.read_text())
    assert plan["trained"][0]["trial"] == 1 and plan["trained"][0]["tuning_run"] == best_run
    assert plan["seeds"] == [1, 2, 3] and plan_finals.TUNING_SEED not in plan["seeds"]
    capsys.readouterr()
    assert plan_finals.main([*args, "--list"]) == 0
    assert capsys.readouterr().out.splitlines() == [f"train methods/lora.yaml 1 {s}" for s in (1, 2, 3)]
    assert plan_finals.main([*args, "--best", "lora"]) == 0
    assert capsys.readouterr().out.strip() == best_run


def test_plan_finals_reports_missing_results(tmp_path, capsys):
    empty = tmp_path / "e"
    empty.mkdir()
    out = tmp_path / "plan.json"
    assert plan_finals.main(["--results", str(empty), "--configs", str(ROOT / "configs"), "--out", str(out), "--dry-run"]) == 2
    assert "missing" in capsys.readouterr().out and not out.exists()


def test_primary_partial_runs_on_joined_synthetic_tables(synthetic):
    eval_df, drift_df = load_tables(synthetic)
    joined = join_tables(checkpoint_table(eval_df), drift_table(drift_df))
    assert len(joined) > 0 and -1 <= primary_partial(joined)["rho"] <= 1


def test_make_figures_demo_target_writes_into_a_temp_copy(tmp_path):
    shutil.copy(ROOT / "Makefile", tmp_path / "Makefile")
    out = subprocess.run(["make", "-n", "figures-demo"], cwd=tmp_path, capture_output=True, text=True)
    assert "--demo --out paper_assets/demo" in out.stdout
