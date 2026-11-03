"""The README must describe the repository truthfully: real commands, real paths, and no results."""

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
BLOCK = re.compile(r"^```(\w*)\n(.*?)^```", re.S | re.M)
SYSTEM_TOOLS = {"source", "export", "mkdir", "tail", "squeue", "scontrol", "sbatch", "conda", "pip", "make", "bash", "python", "python3.12"}
STATUSES = {"Tested (CPU)", "Tested (real model, CPU)", "Written, needs GPU/LIBERO"}


def blocks():
    return [(m.group(1), m.group(2)) for m in BLOCK.finditer(README)]


def bash_commands():
    commands = []
    for language, body in blocks():
        if language == "bash":
            commands += [line.strip() for line in body.splitlines() if line.strip() and not line.startswith("#")]
    return commands


def prose():
    text = re.sub(r"<!--.*?-->", "", BLOCK.sub("", README))
    return re.sub(r"!?\[[^\]]*\]\([^)]*\)", "", text)


# ---------- commands ----------
def test_there_are_commands_to_check():
    assert len(bash_commands()) > 15


@pytest.mark.parametrize("command", bash_commands())
def test_every_command_uses_a_known_tool_and_existing_paths(command):
    tokens = shlex.split(command)
    assert tokens[0] in SYSTEM_TOOLS, command
    for token in tokens[1:]:
        value = token.split("=", 1)[-1] if token.startswith("--") else token
        if re.match(r"^(scripts|slurm|configs|tests|docs|src|examples)/", value):
            assert (ROOT / value).exists(), f"{command}: {value} does not exist"


def test_make_targets_exist():
    makefile = (ROOT / "Makefile").read_text()
    for command in bash_commands():
        if command.startswith("make "):
            assert re.search(rf"^{command.split()[1]}:", makefile, re.M), command


def test_flags_exist_in_each_python_script():
    """Each flag a README command passes must appear in that script's --help."""
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    for command in bash_commands():
        tokens = shlex.split(command)
        if tokens[0] != "python" or not tokens[1].startswith("scripts/"):
            continue
        help_text = subprocess.run([sys.executable, tokens[1], "--help"], cwd=ROOT, env=env, capture_output=True, text=True).stdout
        for token in tokens[2:]:
            if token.startswith("--"):
                assert token.split("=")[0] in help_text, f"{tokens[1]} has no flag {token}"


SCRIPT_HELP = sorted(p.name for p in (ROOT / "scripts").glob("*.py"))


@pytest.mark.parametrize("script", SCRIPT_HELP)
def test_every_script_answers_help(script):
    result = subprocess.run([sys.executable, f"scripts/{script}", "--help"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def _dry_runs():
    out = []
    for command in bash_commands():
        if "--dry-run" in command:
            out.append(command)
    return out


def test_every_launch_script_appears_with_dry_run():
    commands = _dry_runs()
    for path in sorted((ROOT / "slurm").glob("launch_*.sh")) + [ROOT / "slurm" / "chain.sh"]:
        assert any(f"slurm/{path.name}" in c for c in commands), path.name


@pytest.mark.parametrize("command", _dry_runs())
def test_dry_run_commands_really_run(command):
    tokens = shlex.split(command)
    if tokens[0] == "python":
        tokens[0] = sys.executable
    env = {k: v for k, v in os.environ.items() if not k.startswith("VR_")}
    env.update(PYTHON=sys.executable, PATH=f"{Path(sys.executable).parent}:{env['PATH']}")
    result = subprocess.run(tokens, cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    # plan_finals exits 2 to report that results are missing, which is correct before any experiment.
    allowed = {0, 2} if "plan_finals" in command else {0}
    assert result.returncode in allowed, result.stderr


def test_the_figures_demo_target_runs(tmp_path):
    env = {**os.environ, "PYTHON": sys.executable}
    result = subprocess.run(["make", "figures-demo", f"PYTHON={sys.executable}"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert (ROOT / "paper_assets/demo/fig1_pareto.png").exists()


def test_aggregate_update_readme_flag_on_a_temp_copy(tmp_path):
    copy = tmp_path / "README.md"
    copy.write_text(README)
    command = [
        sys.executable, "scripts/aggregate.py", "--results", "examples/synthetic", "--out", str(tmp_path / "assets"),
        "--update-readme", "--readme", str(copy),
    ]  # fmt: skip
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    updated = copy.read_text()
    assert "Experiments are in progress" not in updated and "M1 Full SFT" in updated
    assert updated.split("<!-- RESULTS:START -->")[0] == README.split("<!-- RESULTS:START -->")[0]
    assert updated.split("<!-- RESULTS:END -->")[1] == README.split("<!-- RESULTS:END -->")[1]
    assert README == (ROOT / "README.md").read_text()


# ---------- links and contents ----------
def _slug(heading):
    text = heading.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return text.replace(" ", "-")


def _headings(text):
    return [m.group(2) for m in re.finditer(r"^(#{1,6}) (.+)$", BLOCK.sub("", text), re.M)]


def test_config_example_matches_the_shipped_file():
    example = next(body for language, body in blocks() if language == "yaml")
    shipped = (ROOT / "configs/methods/l2sp.yaml").read_text()
    for line in example.splitlines()[1:]:
        assert line in shipped, line


def test_table_of_contents_links_match_headings():
    contents = README.split("## Contents")[1].split("## Overview")[0]
    anchors = re.findall(r"\]\(#([^)]+)\)", contents)
    slugs = {_slug(h) for h in _headings(README)}
    assert len(anchors) >= 15
    for anchor in anchors:
        assert anchor in slugs, anchor
    for anchor in re.findall(r"\]\(#([^)]+)\)", README):
        assert anchor in slugs, anchor


def test_relative_links_point_to_existing_files():
    links = re.findall(r"\]\(([^)]+)\)", BLOCK.sub("", README))
    relative = [link for link in links if not link.startswith(("http", "#"))]
    assert relative
    for link in relative:
        path, _, fragment = link.partition("#")
        assert (ROOT / path).exists(), link
        if fragment:
            assert fragment in {_slug(h) for h in _headings((ROOT / path).read_text())}, link


def test_required_sections_are_in_order():
    wanted = [
        "TL;DR", "Contents", "Overview", "What's implemented", "Methods", "Experimental protocol",
        "Drift and divergence metrics", "Repository layout", "Installation", "Quickstart",
        "Running on the HPC cluster", "Outputs and result schema", "Reproducibility",
        "Design decisions and trade-offs", "Results", "Limitations and scope", "Roadmap", "Citation",
        "Acknowledgments", "References", "License",
    ]  # fmt: skip
    found = [m.group(1) for m in re.finditer(r"^## (.+)$", BLOCK.sub("", README), re.M)]
    assert found == wanted


def test_status_table_comes_from_progress():
    progress = (ROOT / "docs/progress.md").read_text()
    section = README.split("## What's implemented")[1].split("## Methods")[0]
    rows = [r for r in section.splitlines() if r.startswith("| ") and "---" not in r and "Component" not in r]
    assert len(rows) >= 20
    for row in rows:
        component, _, status = [c.strip() for c in row.strip("|").split(" | ")]
        assert status in STATUSES, row
        assert f"| {component} | {status} |" in progress, component
    # Every row of progress's table should also reach the README.
    table = progress.split("## Status by component")[1].split("`Tested (CPU)` runs")[0]
    progress_rows = [r for r in table.splitlines() if r.startswith("| ") and "---" not in r and "Component" not in r]
    assert len(progress_rows) == len(rows)


def test_unverified_list_matches_the_code():
    listed = (ROOT / "docs/progress.md").read_text().split("## UNVERIFIED functions")[1].split("## Known risks")[0]
    found = []
    for base in ("src", "scripts"):
        for path in (ROOT / base).rglob("*.py"):
            for match in re.finditer(r"# UNVERIFIED: needs GPU/LIBERO\ndef (\w+)\(", path.read_text()):
                found.append((path.relative_to(ROOT).as_posix(), match.group(1)))
    assert len(found) > 10
    for rel, name in found:
        line = next(li for li in listed.splitlines() if rel in li)
        assert f"`{name}`" in line, (rel, name)
    assert len(re.findall(r"`\w+`", listed)) == len(found)


def test_rollout_counts_match_the_configs():
    from vlaretain.eval.suites import FULL_EVAL, expand_eval_set
    from vlaretain.methods.merge import global_alpha_grid, module_alpha_grid

    root = ROOT / "configs/eval"
    full = sum(j.n_episodes for name in FULL_EVAL for j in expand_eval_set(name, root))
    validation = sum(j.n_episodes for j in expand_eval_set("validation", root))
    sweep = (len(global_alpha_grid()) + len(module_alpha_grid())) * validation
    cell = lambda n: f"{n:,}"  # noqa: E731
    assert f"| S1 eval | {cell(full + validation)} |" in README
    assert f"validation at 2 steps | {cell(15 * 2 * validation)} |" in README
    assert f"on validation | {cell(sweep)} |" in README
    assert f"4 checkpoints | {cell(15 * 4 * full)} |" in README
    assert f"3 seeds | {cell(2 * 3 * full)} |" in README


def test_test_count_is_current():
    match = re.search(r"`make test` collects (\d+) tests\. The (\d+) in", README)
    assert match
    out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q"], cwd=ROOT, capture_output=True, text=True, timeout=300).stdout
    assert int(match.group(1)) == len(re.findall(r"^tests/\S+::", out, re.M))
    assert int(match.group(2)) == len(re.findall(r"^tests/test_real_model\.py::", out, re.M))


def test_length_and_required_elements():
    assert 500 <= len(README.splitlines()) <= 700
    assert "```mermaid" in README and "$$" in README
    assert "**Authors:** <Student 1>, <Student 2>, <Student 3>, <Student 4> · Advisor: <Advisor> · San José State University" in README
    assert "<!-- RESULTS:START -->\nExperiments are in progress. This section is generated automatically from results/ by scripts/aggregate.py --update-readme.\n<!-- RESULTS:END -->" in README
    assert README.count("- [ ]") == 6 and "- [x]" not in README


def test_code_blocks_are_tagged():
    fences = re.findall(r"^```(\w*)$", README, re.M)
    opening = fences[::2]
    assert all(opening), "every code block needs a language tag"
    assert {"bash", "yaml", "json", "mermaid"} <= set(opening)


def test_only_true_badges():
    badges = re.findall(r"img\.shields\.io/badge/([\w-]+)", README)
    assert sorted(b.split("-")[0] for b in badges) == ["license", "python"]
    assert not re.search(r"(?i)tests?.passing|build.passing|coverage", README)
    assert "python-3.12" in README and "requires-python = \">=3.12,<3.13\"" in (ROOT / "pyproject.toml").read_text()
    assert (ROOT / "LICENSE").read_text().startswith("MIT License")


def test_voice_has_no_hype_or_emoji_or_exclamations():
    text = prose()
    assert "!" not in text
    for word in ("revolutionary", "state-of-the-art", "state of the art", "blazing", "groundbreaking", "cutting-edge"):
        assert word not in README.lower()
    assert not re.search(r"[\U0001F300-\U0001FAFF☀-➿]", README)
    assert not re.search(r"(?i)\bwe (show|prove|demonstrate|found) that\b", README)


def test_readme_paragraphs_are_short():
    for block in BLOCK.sub("", README).split("\n\n"):
        block = block.strip()
        if not block or block.startswith(("|", "#", "-", "$", "[", "<", "**Authors")):
            continue
        sentences = re.findall(r"[.?](?:\s|$)", re.sub(r"`[^`]*`|\([^)]*\)|\$[^$]*\$", "", block))
        limit = 4 if block.startswith("Fine-tuning a pretrained") else 3  # the TL;DR is specified at 3-4 sentences
        assert len(sentences) <= limit, block


def test_no_result_numbers_in_the_readme():
    """Percentages in prose would be claims. The only ones allowed describe the protocol."""
    allowed = {"2%", "10%", "20%", "80%", "95%"}
    for number in re.findall(r"\d+(?:\.\d+)?%", prose()):
        assert number in allowed, number


def test_references_have_arxiv_links():
    section = README.split("## References")[1].split("## License")[0]
    for arxiv in ("2509.04259", "2505.23705", "2512.08333", "2510.13626", "2510.03827", "2506.01844", "2306.03310", "2603.03818"):
        assert f"https://arxiv.org/abs/{arxiv}" in section
