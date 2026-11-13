"""Join trials and generations into one scores table per experiment (word count, length compliance, content items).

Experiment A final label precedence: judge label (if the reply was routed and judged) >
NLI class (persona / instruction) > None.  Keyword class is kept as a diagnostic column only.
Experiment B: programmatic length/content metrics per trial; pairwise judge results are
aggregated separately in analyze.py.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from . import OUTPUTS, metrics_b
from .content_items import floor_length, from_dict
from .generate import gen_files, load_trials, rec_seed


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _latest_records(path: Path) -> dict[str, dict]:
    recs: dict[str, dict] = {}
    for r in _read_jsonl(path):
        if r.get("error") is None or r["trial_id"] not in recs:
            recs[r["trial_id"]] = r
    return recs


def evaluate(exp: str) -> Path:
    trials = {t["trial_id"]: t for t in load_trials(exp)}
    exp_dir = OUTPUTS / f"exp_{exp}"
    score_dir = exp_dir / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for gen_file in gen_files(exp):
        model = gen_file.stem
        per_model = []
        for tid, r in _latest_records(gen_file).items():
            t = trials.get(tid)
            if t is None:
                continue
            text = r.get("text", "")
            base = {"model": model, "trial_id": tid, "seed": rec_seed(r), "error": r.get("error"),
                    "finish_reason": r.get("finish_reason"),
                    "truncated": r.get("finish_reason") in ("length", "max_tokens"), "text": text}
            if True:
                items = [from_dict(d) for d in t["meta"]["items"]]
                # floor recomputed with regex-consistent minimal renderings (the stored `floor`
                # over-counted room/bring by one word each); k = 0 borrows the k_ref set's floor
                ref_items = items
                if not items:
                    k_ref = max(x["k"] for x in trials.values())
                    ref = trials.get(t.get("ref_id", "")) or \
                          trials.get(f"B-{t['persona']}-k{k_ref}-s{t['set_index']:02d}-r0") or \
                          trials.get(f"B2-{t['persona']}-k{k_ref}-s{t['set_index']:02d}-b0") or \
                          trials.get(f"B3-{t['persona']}-k{k_ref}-s{t['set_index']:02d}-b0")
                    ref_items = [from_dict(d) for d in ref["meta"]["items"]] if ref else []
                floor_true = floor_length(ref_items, t["unit"]) if ref_items else None
                base.update({"cell": t["cell"], "persona": t["persona"], "k": t["k"], "ratio": t["ratio"],
                             "budget": t["budget"], "floor_built": t["floor"], "floor": floor_true,
                             "ratio_true": (round(t["budget"] / floor_true, 2) if floor_true and t["budget"] else 0.0),
                             "unit": t["unit"], "set_index": t["set_index"]})
                base["lo"], base["hi"] = t.get("lo"), t.get("hi")
                for key in ("kind", "system_side", "user_side", "user_pole", "tight", "loose"):
                    if key in t:
                        base[key] = t[key]
                base.update(metrics_b.judge(text, items, t["budget"], t["unit"], t.get("lo"), t.get("hi")))
            per_model.append(base)
        with (score_dir / f"{model}.jsonl").open("w") as f:
            for row in per_model:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows.extend(per_model)
    df = pd.DataFrame(rows)
    out = exp_dir / "scores.csv"
    df.to_csv(out, index=False)
    print(f"evaluated {len(df)} rows for {df['model'].nunique() if len(df) else 0} models -> {out}")
    return out
