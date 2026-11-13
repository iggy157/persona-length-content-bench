"""Run every trial of an experiment against a model and store raw outputs.

Output: outputs/<exp>/gen/<model>.jsonl, one record per trial, resumable (records with
``error == null`` are not re-run)."""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

from . import CONFIG, DATA_BUILT, OUTPUTS
from .llm import ChatClient, ModelSpec


def load_model_specs(path: Path = CONFIG / "models.yaml") -> dict[str, ModelSpec]:
    cfg = yaml.safe_load(path.read_text())
    defaults = cfg.get("defaults", {})
    specs = {}
    override = os.environ.get("PCB_BASE_URL")  # set by scripts/run_models_thalys.sh per GPU/port
    for m in cfg["models"]:
        kw = {**defaults, **m}
        if override and kw.get("base_url"):
            kw["base_url"] = override
        specs[kw["name"]] = ModelSpec(**kw)
    return specs


def load_trials(exp: str) -> list[dict]:
    path = DATA_BUILT / f"exp_{exp}" / "trials.jsonl"
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def gen_files(exp: str) -> list[Path]:
    """Generation files of an experiment: gen/*.jsonl plus any gen_s<seed>/*.jsonl (extra sampling seeds)."""
    base = OUTPUTS / f"exp_{exp}"
    return sorted(base.glob("gen/*.jsonl")) + sorted(base.glob("gen_s*/*.jsonl"))


def rec_seed(r: dict) -> int:
    return int(r.get("seed") or 1)


def _done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("error") is None:
            done.add(r["trial_id"])
    return done


def run(exp: str, model_name: str, *, limit: int | None = None, cell: str | None = None,
        specs_path: Path = CONFIG / "models.yaml") -> Path:
    spec = load_model_specs(specs_path)[model_name]
    client = ChatClient(spec)
    trials = load_trials(exp)
    # PCB_GEN_SUBDIR lets a timing re-run write next to (not over) the main results
    out_dir = OUTPUTS / f"exp_{exp}" / os.environ.get("PCB_GEN_SUBDIR", "gen")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{model_name}.jsonl"
    done = _done_ids(out)
    todo = [t for t in trials if t["trial_id"] not in done]
    if cell:
        todo = [t for t in todo if any(c in t["cell"] for c in cell.split(","))]
    if limit is not None:
        todo = todo[:limit]
    lock = threading.Lock()

    seed = int(os.environ["PCB_SEED"]) if os.environ.get("PCB_SEED") else None

    def work(t: dict) -> dict:
        c = client.complete(t["system"], t["user"], max_tokens=t["gen"]["max_tokens"],
                            **({"seed": seed} if seed is not None else {}))
        return {
            "trial_id": t["trial_id"], "model": model_name, "model_id": spec.model_id, "seed": seed or 1,
            "text": c.text, "finish_reason": c.finish_reason,
            "prompt_tokens": c.prompt_tokens, "completion_tokens": c.completion_tokens,
            "latency_ms": round(c.latency_ms, 1), "attempts": c.attempts, "error": c.error,
            "request": c.request, "ts": time.time(),
        }

    n_err = 0
    with out.open("a") as f, ThreadPoolExecutor(spec.concurrency) as ex:
        futs = [ex.submit(work, t) for t in todo]
        for fut in tqdm(as_completed(futs), total=len(futs), desc=f"{exp}/{model_name}"):
            r = fut.result()
            n_err += r["error"] is not None
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
    print(f"[{model_name}] wrote {len(todo)} records ({n_err} errors) -> {out}")
    return out
