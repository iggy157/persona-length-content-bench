"""Independent persona rating of Experiment B replies (observer-report IPIP-NEO).

Each reply is shown alone to the judge together with the 30 IPIP-NEO items of the facets the personas were
built from (A3 altruism, E1 friendliness, A6 sympathy; 14 +keyed = warm persona lines, 16 -keyed = cold
persona lines). The judge answers each item on the questionnaire's own 5-point accuracy scale. Per item the
score is the probability-weighted expectation over the answer tokens 1..5 (G-Eval style), read from the
judge's logprobs; `warm` = mean of the 14 +keyed items, `cold` = mean of the 16 -keyed items.

Output: outputs/exp_<exp>/rate/<judge>.jsonl, one record per (model, seed, trial_id). Resumable.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from . import OUTPUTS
from .evaluate import _read_jsonl
from .generate import gen_files, load_model_specs, load_trials, rec_seed
from .ipip import load_facets
from .llm import ChatClient

FACETS = ["A3", "E1", "A6"]
SCALE = ("1 = very inaccurate, 2 = moderately inaccurate, 3 = neither accurate nor inaccurate, "
         "4 = moderately accurate, 5 = very accurate")

RATE_PROMPT = """\
Below is a text message someone received from a friend, and the reply they wrote.

Friend's message: "{friend_message}"
Reply: "{reply}"

The statements below are items from a personality questionnaire, written in the first person. Judging ONLY \
from the reply, rate how accurately each statement describes the person who wrote the reply, on the \
questionnaire's scale: {scale}.
If the reply gives no evidence either way for a statement, answer 3.

{items}

Answer with JSON only, in exactly this form: {{"1": n, "2": n, ..., "{n_items}": n}} where each n is 1, 2, 3, 4 or 5.
"""


def items() -> tuple[list[str], list[str]]:
    """(plus-keyed items, minus-keyed items) in fixed order A3, E1, A6."""
    f = load_facets()
    plus = [it for c in FACETS for it in f[c].plus]
    minus = [it for c in FACETS for it in f[c].minus]
    return plus, minus


def facet_items(code: str) -> tuple[list[str], list[str]]:
    f = load_facets()[code]
    return list(f.plus), list(f.minus)


def build_prompt(friend_message: str, reply: str, facet: str | None = None) -> tuple[str, int]:
    """facet given -> that facet's 10 items (exp_b4); else the 30 items of A3+E1+A6 (exp_b warm/cold)."""
    plus, minus = facet_items(facet) if facet else items()
    allitems = plus + minus
    lines = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(allitems))
    return RATE_PROMPT.format(friend_message=friend_message, reply=reply.replace('"', "'"), scale=SCALE,
                              items=lines, n_items=len(allitems)), len(plus)


_VAL = re.compile(r'"(\d+)"\s*:\s*([1-5])')


def expected_ratings(logprobs: list | None, text: str, n_items: int) -> tuple[list[float | None], list[int | None], str]:
    """Per item: expectation over tokens '1'..'5' at the value position; argmax (parsed) value; and a note.
    Positions are located in the concatenation of the generated tokens (which may include reasoning tokens),
    using the LAST JSON object found there."""
    exp: list[float | None] = [None] * n_items
    arg: list[int | None] = [None] * n_items
    if not logprobs:
        # no logprobs: fall back to the parsed text
        try:
            d = json.loads(text[text.rfind("{"):])
            for i in range(n_items):
                v = int(d[str(i + 1)]); arg[i] = v; exp[i] = float(v)
            return exp, arg, "no_logprobs"
        except (ValueError, KeyError, TypeError, json.JSONDecodeError):
            return exp, arg, "unparsable"
    toks = [lp["token"] for lp in logprobs]
    cat = "".join(toks)
    start = cat.rfind("{")
    if start < 0:
        return exp, arg, "no_json"
    ends = [0]
    for t in toks:
        ends.append(ends[-1] + len(t))
    for m in _VAL.finditer(cat, start):
        idx = int(m.group(1)) - 1
        if not 0 <= idx < n_items:
            continue
        pos = m.start(2)
        ti = bisect.bisect_right(ends, pos) - 1
        top = logprobs[ti].get("top") or {}
        probs = {}
        for k, v in top.items():
            ks = k.strip().strip('"')
            if ks in ("1", "2", "3", "4", "5"):
                probs[int(ks)] = max(probs.get(int(ks), 0.0), math.exp(v))
        v = int(m.group(2))
        if toks[ti].strip().strip('"') != str(v):
            probs = {v: 1.0}          # token boundary did not isolate the digit: fall back to the written value
        if v not in probs:
            probs[v] = max(probs.values(), default=1.0)
        z = sum(probs.values())
        exp[idx] = sum(k * p for k, p in probs.items()) / z
        arg[idx] = v
    note = "ok" if all(e is not None for e in exp) else f"missing:{sum(e is None for e in exp)}"
    return exp, arg, note


def _worker(t: dict, r: dict, client: ChatClient, model: str, seed: int) -> dict:
    facet = t["meta"].get("facet")
    prompt, n_plus = build_prompt(t["meta"]["scenario"], r["text"], facet)
    c = client.complete(None, prompt, temperature=0.0, logprobs=True, top_logprobs=10)
    n_items = sum(len(x) for x in (facet_items(facet) if facet else items()))
    if c.error:
        return {"model": model, "seed": seed, "trial_id": t["trial_id"], "error": c.error}
    exp, arg, note = expected_ratings(c.logprobs, c.text, n_items)
    plus = [e for e in exp[:n_plus] if e is not None]
    minus = [e for e in exp[n_plus:] if e is not None]
    return {"model": model, "seed": seed, "trial_id": t["trial_id"], "persona": t["persona"], "k": t["k"],
            "facet": facet, "n_plus": n_plus,
            "budget": t["budget"], "set_index": t["set_index"], "length": len(r["text"].split()),
            "warm": sum(plus) / len(plus) if plus else None, "cold": sum(minus) / len(minus) if minus else None,
            "ratings": exp, "argmax": arg, "note": note, "raw": c.text[-400:],
            "error": None if note.startswith("ok") or note == "no_logprobs" else note}


def rate(exp: str, judge_names: list[str], *, models: list[str] | None = None, ks: list[int] | None = None,
         personas: list[str] | None = None, max_sets: int | None = None, seeds: list[int] | None = None,
         concurrency: int = 48, cells: list[str] | None = None) -> None:
    specs = load_model_specs()
    trials = {t["trial_id"]: t for t in load_trials(exp)}
    out_dir = OUTPUTS / f"exp_{exp}" / "rate"
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for gen_file in gen_files(exp):
        model = gen_file.stem
        if models and model not in models:
            continue
        recs = [r for r in _read_jsonl(gen_file) if r.get("error") is None and r.get("finish_reason") not in ("length", "max_tokens")]
        for r in recs:
            t = trials.get(r["trial_id"])
            if t is None:
                continue
            if ks is not None and t["k"] not in ks:
                continue
            if personas is not None and str(t["persona"]) not in personas:
                continue
            if max_sets is not None and t["set_index"] >= max_sets:
                continue
            if cells is not None and not any(c in t["cell"] for c in cells):
                continue
            if seeds is not None and rec_seed(r) not in seeds:
                continue
            jobs.append((t, r, model, rec_seed(r)))
    for jname in judge_names:
        client = ChatClient(specs[jname])
        out = out_dir / f"{jname}.jsonl"
        done = set()
        if out.exists():
            for rr in _read_jsonl(out):
                if rr.get("error") is None:
                    done.add((rr["model"], rr["seed"], rr["trial_id"]))
        todo = [j for j in jobs if (j[2], j[3], j[0]["trial_id"]) not in done]
        print(f"[rate {jname}] {len(todo)} replies to rate ({len(done)} done)")
        lock = threading.Lock()
        n_err = 0
        with out.open("a") as f, ThreadPoolExecutor(max_workers=concurrency) as ex:
            futs = [ex.submit(_worker, t, r, client, m, s) for t, r, m, s in todo]
            for fu in tqdm(as_completed(futs), total=len(futs), desc=f"rate:{jname}"):
                rec = fu.result()
                n_err += rec.get("error") is not None
                with lock:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        print(f"[rate {jname}] wrote {len(todo)} ({n_err} errors) -> {out}")


def _env_list(name: str) -> list[str] | None:
    v = os.environ.get(name)
    return [x for x in v.split(",") if x] if v else None
