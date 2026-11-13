"""Command line entry point: pcb <command> [options]."""

from __future__ import annotations

import argparse
import json
import random
import sys

from . import CONFIG, DATA_BUILT


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="pcb")
    sp = ap.add_subparsers(dest="cmd", required=True)

    sp.add_parser("build-b4", help="build Experiment B facet design (30 facets x high/low/off) -> exp_b4")
    sp.add_parser("build-b5", help="build the complement with 0/1/2 content items -> exp_b5")
    sp.add_parser("build-c", help="build the timing baseline (same instruction on both sides) -> exp_c")
    p = sp.add_parser("rate", help="independent IPIP-NEO observer rating of Experiment B replies (judge logprobs)")
    p.add_argument("--exp", choices=["b4", "b5", "c"], required=True)
    p.add_argument("--judges", required=True, help="comma-separated judge names")
    p.add_argument("--models", default=None, help="comma-separated subset of generation models")
    p.add_argument("--k", default=None, help="comma-separated k values, e.g. 0,3")
    p.add_argument("--personas", default=None, help="comma-separated personas, e.g. warm,cold,off")
    p.add_argument("--max-sets", type=int, default=None)
    p.add_argument("--seeds", default=None, help="comma-separated sampling seeds to rate, e.g. 1")
    p.add_argument("--cells", default=None, help="comma-separated substrings; only trials whose cell matches are rated (exp c: pers_)")
    p.add_argument("--concurrency", type=int, default=48)

    p = sp.add_parser("show", help="print rendered prompts for a few trials")
    p.add_argument("--exp", choices=["b4", "b5", "c"], required=True)
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--cell", default=None, help="substring filter on cell")
    p.add_argument("--seed", type=int, default=0)

    p = sp.add_parser("generate", help="run trials against models")
    p.add_argument("--exp", choices=["b4", "b5", "c"], required=True)
    p.add_argument("--models", required=True, help="comma-separated names from config/models.yaml")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--cell", default=None, help="comma-separated substrings; only matching cells are run")

    p = sp.add_parser("evaluate", help="score every reply (word count, content items, length compliance) into scores.csv")
    p.add_argument("--exp", choices=["b4", "b5", "c"], required=True)

    args = ap.parse_args(argv)

    if args.cmd == "rate":
        from .rate import rate
        rate(args.exp, args.judges.split(","), models=args.models.split(",") if args.models else None,
             ks=[int(x) for x in args.k.split(",")] if args.k else None,
             personas=args.personas.split(",") if args.personas else None, max_sets=args.max_sets,
             seeds=[int(x) for x in args.seeds.split(",")] if args.seeds else None, concurrency=args.concurrency, cells=args.cells.split(",") if args.cells else None)
    elif args.cmd == "build-b5":
        from .build_b import build_b5
        print(build_b5())
    elif args.cmd == "build-c":
        from .build_c import build_c
        print(build_c())
    elif args.cmd == "build-b4":
        from .build_b import build_b4
        print(build_b4())
    elif args.cmd == "show":
        path = DATA_BUILT / f"exp_{args.exp}" / "trials.jsonl"
        trials = [json.loads(l) for l in path.read_text().splitlines()]
        if args.cell:
            trials = [t for t in trials if args.cell in t["cell"]]
        random.Random(args.seed).shuffle(trials)
        for t in trials[: args.n]:
            print("=" * 80, t["trial_id"], "cell=", t["cell"])
            print("--- system\n" + t["system"])
            print("--- user\n" + t["user"])
    elif args.cmd == "generate":
        from .generate import run
        for m in args.models.split(","):
            run(args.exp, m.strip(), limit=args.limit, cell=args.cell)
    elif args.cmd == "evaluate":
        from .evaluate import evaluate
        evaluate(args.exp)


if __name__ == "__main__":
    main(sys.argv[1:])
