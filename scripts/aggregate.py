#!/usr/bin/env python
"""Rebuild every figure and table from JSONL results. Missing results show as missing and are never filled in."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
PLACEHOLDER = (
    "Experiments are in progress. This section is generated automatically from results/ "
    "by scripts/aggregate.py --update-readme."
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results", help="directory of *.jsonl files")
    parser.add_argument("--out", default="paper_assets", help="output directory")
    parser.add_argument("--demo", action="store_true", help="use the bundled synthetic example")
    parser.add_argument("--update-readme", action="store_true", help="rewrite the README's results block")
    parser.add_argument("--readme", default=str(REPO / "README.md"))
    parser.add_argument("--final-step", type=int, default=20_000)
    parser.add_argument("--dry-run", action="store_true", help="print what would be read and written, then exit")
    return parser.parse_args(argv)


def replace_block(text: str, block: str) -> str:
    if START not in text or END not in text or text.index(START) > text.index(END):
        raise ValueError("The README has no results markers")
    head, _, rest = text.partition(START)
    _, _, tail = rest.partition(END)
    return f"{head}{START}\n{block.strip()}\n{END}{tail}"


def results_block(table_md: str, assets: Path, readme: Path) -> str:
    relative = Path(os.path.relpath(assets, readme.resolve().parent)).as_posix()
    return "\n\n".join([
        "Built by `scripts/aggregate.py --update-readme`. Rates are percentages, as mean ± SD over seeds.",
        table_md,
        f"![Target SR vs retention SR]({relative}/fig1_pareto.png)",
        f"![Forgetting vs new-state divergence]({relative}/fig2_forgetting_vs_divergence.png)",
        f"![Per-module and per-layer drift]({relative}/fig3_drift.png)",
    ])  # fmt: skip


def main(argv=None) -> int:
    args = parse_args(argv)
    results_dir = REPO / "examples" / "synthetic" if args.demo else Path(args.results)
    out = Path(args.out)
    suffix = " (synthetic data)" if args.demo else ""
    if args.demo and args.update_readme:
        raise SystemExit("--demo cannot update the README: it only uses synthetic data")
    if args.dry_run:
        print(f"read {results_dir}/*.jsonl, write figures and tables to {out}")
        if args.update_readme:
            print(f"update the results block of {args.readme}")
        return 0

    import matplotlib

    matplotlib.use("Agg")

    from vlaretain.analysis.common import checkpoint_table, drift_table, load_tables
    from vlaretain.analysis.correlate import correlation_report, join_tables, primary_partial
    from vlaretain.analysis.figures import (
        completeness_note,
        plot_forgetting_vs_divergence,
        plot_module_drift,
        table_one,
        to_latex,
        to_markdown,
    )
    from vlaretain.analysis.pareto import plot_pareto
    from vlaretain.analysis.synthetic import SYNTHETIC_PREFIX

    out.mkdir(parents=True, exist_ok=True)
    eval_df, drift_df = load_tables(results_dir)
    checkpoints = checkpoint_table(eval_df)
    drift = drift_table(drift_df)
    joined = join_tables(checkpoints, drift)

    plot_pareto(checkpoints, out / "fig1_pareto.png", suffix)
    plot_forgetting_vs_divergence(joined, out / "fig2_forgetting_vs_divergence.png", suffix)
    plot_module_drift(drift_df, out / "fig3_drift.png", suffix)
    table = table_one(eval_df, checkpoints, args.final_step)
    heading = "# Table 1" + (" (synthetic data)\n\n" if args.demo else "\n\n")
    (out / "table1.md").write_text(heading + to_markdown(table) + "\n")
    (out / "table1.tex").write_text(to_latex(table))
    table.to_csv(out / "table1.csv", index=False)

    report = correlation_report(joined) if not joined.empty else None
    if report is not None:
        report.to_csv(out / "correlations.csv", index=False)
        partial = primary_partial(joined)
        (out / "partial_correlation.txt").write_text(
            f"{suffix.strip()}\nPartial Spearman of new-state velocity distance with forgetting, controlling for "
            f"weight drift and target SR: rho={partial['rho']:.3f} p={partial['p']:.3g} n={partial['n']}\n"
        )
    print(completeness_note(eval_df))
    print(f"{len(checkpoints)} complete checkpoints, {len(joined)} with drift. Wrote assets to {out}")

    if args.update_readme:
        readme = Path(args.readme)
        if readme.resolve() == (REPO / "README.md").resolve() and not eval_df.empty:
            if eval_df["run_id"].astype(str).str.startswith(SYNTHETIC_PREFIX).any():
                raise SystemExit("Refusing to put synthetic results in the project README")
        block = results_block(to_markdown(table), out, readme) if len(table) else PLACEHOLDER
        readme.write_text(replace_block(readme.read_text(), block))
        print(f"Updated {readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
