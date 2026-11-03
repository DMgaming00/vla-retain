#!/usr/bin/env python
"""Train Stage-1 or one Stage-2 method from YAML configs. Resumes automatically and exits cleanly on SIGUSR1."""

from __future__ import annotations

import argparse
import json
import logging
import sys

from vlaretain.config import load_config, resolved_dict, run_id_of, trial_config


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", action="append", required=True, help="YAML file; repeat to overlay")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--trial", type=int, default=None, help="index into the config's tuning list")
    parser.add_argument("--dry-run", action="store_true", help="print the resolved config and exit")
    parser.add_argument("--print-run-id", action="store_true", help="print only the run id and exit")
    parser.add_argument("--print-done", action="store_true", help="print 1 if the run has a DONE marker, else 0")
    parser.add_argument("--print-n-trials", action="store_true", help="print the number of tuning trials and exit")
    parser.add_argument("--allow-disabled", action="store_true", help="run a method disabled by default")
    return parser.parse_args(argv)


def resolve(args: argparse.Namespace):
    cfg = load_config(args.config, args.overrides)
    if args.trial is not None and not 0 <= args.trial < len(cfg.tuning):
        raise SystemExit(f"--trial {args.trial} is outside the {len(cfg.tuning)} trials in this config")
    return trial_config(args.config, args.trial, args.overrides)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = resolve(args)
    if args.print_n_trials:
        print(len(cfg.tuning))
        return 0
    if args.print_run_id:
        print(run_id_of(cfg))
        return 0
    if args.print_done:
        print(int((cfg.run_dir / "DONE").exists()))
        return 0
    if args.dry_run:
        print(f"run_id: {run_id_of(cfg)}")
        print(f"run_dir: {cfg.run_dir}")
        print(f"s1_ckpt: {cfg.s1_ckpt}")
        print(json.dumps(resolved_dict(cfg), indent=2, sort_keys=True, default=str))
        return 0
    if not cfg.method.enabled and not args.allow_disabled:
        raise SystemExit(f"Method {cfg.method.name} is disabled by default; pass --allow-disabled to run it")

    # UNVERIFIED: needs GPU/LIBERO
    from vlaretain.methods import get_method
    from vlaretain.train.loop import build_train_data, make_processors, train

    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s")
    method = get_method(cfg)
    policy = method.build_model(cfg.s1_ckpt, cfg)
    processors = make_processors(cfg, policy)
    data = build_train_data(cfg, policy.config, processors)
    del policy  # the loop builds its own, so that resume logic has one code path
    result = train(cfg, method, data)
    print(f"status={result.status} step={result.step} gpu_hours={result.gpu_hours:.3f} run_id={run_id_of(cfg)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
