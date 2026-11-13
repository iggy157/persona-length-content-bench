"""Build Experiment B trials: persona x required content x length budget."""

from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

from . import CONFIG, DATA_BUILT
from .content_items import budgets_from_floor, floor_length, sample_item_set
from .ipip import load_facets
from .prompts import (
    B_SCENARIOS,
    B_SYSTEM_NO_PERSONA,
    B_SYSTEM_PERSONA,
    render_b_user,
    render_persona_lines,
)
from .textnorm import to_second_person


def persona_lines(spec: dict | None, facets) -> list[str] | None:
    if spec is None:
        return None
    lines = []
    for code in spec["facets"]:
        items = facets[code].plus if spec["pole"] == "plus" else facets[code].minus
        lines.extend(to_second_person(i) for i in items)
    return lines


def build(config_path: Path = CONFIG / "exp_b.yaml", out_dir: Path = DATA_BUILT / "exp_b") -> Path:
    cfg = yaml.safe_load(config_path.read_text())
    rng = random.Random(cfg["seed"])
    unit = cfg["unit"]
    facets = load_facets()
    personas = {str(name): persona_lines(spec, facets) for name, spec in cfg["personas"].items()}

    # item sets: shared across persona and budget conditions
    item_sets: dict[int, list] = {}
    for k in cfg["k_values"]:
        if k == 0:
            continue
        item_sets[k] = [sample_item_set(rng, k) for _ in range(cfg["n_item_sets"])]
    k_ref = max(k for k in cfg["k_values"] if k > 0)
    scenarios = [B_SCENARIOS[i % len(B_SCENARIOS)] for i in range(cfg["n_item_sets"])]

    trials = []
    for pname, plines in personas.items():
        system = (B_SYSTEM_PERSONA.format(persona_lines=render_persona_lines(plines))
                  if plines else B_SYSTEM_NO_PERSONA)
        for k in cfg["k_values"]:
            for si in range(cfg["n_item_sets"]):
                items = item_sets[k][si] if k > 0 else []
                # k = 0 borrows the floor of the k_ref item set with the same index so that the
                # absolute budgets match the content-bearing conditions
                floor = floor_length(item_sets[k_ref][si] if k == 0 else items, unit)
                budgets = budgets_from_floor(floor, cfg["ratios"])
                for ratio, budget in zip(cfg["ratios"], budgets):
                    user = render_b_user([i.instruction for i in items], budget, unit, scenarios[si])
                    trials.append({
                        "trial_id": f"B-{pname}-k{k}-s{si:02d}-r{ratio:g}",
                        "experiment": "b",
                        "cell": f"{pname}|k{k}|r{ratio:g}",
                        "persona": pname,
                        "k": k,
                        "ratio": ratio,
                        "budget": budget,          # 0 = unlimited
                        "floor": floor,
                        "unit": unit,
                        "set_index": si,
                        "system": system,
                        "user": user,
                        "gen": {"max_tokens": cfg["max_tokens"]},
                        "meta": {"items": [i.to_dict() for i in items], "scenario": scenarios[si],
                                 "persona_lines": plines},
                    })

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "trials.jsonl"
    with out.open("w") as f:
        for t in trials:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    summary = {"n_trials": len(trials), "personas": {k: v for k, v in personas.items()},
               "floors": {k: sorted({floor_length(s, unit) for s in sets}) for k, sets in item_sets.items()},
               "config": cfg}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return out


def add_absolute_budgets(config_path: Path = CONFIG / "exp_b.yaml",
                         out_dir: Path = DATA_BUILT / "exp_b") -> Path:
    """Append trials with absolute word budgets (config `extra_budgets`) to the existing trial file.

    The existing ratio-based trials are frozen: they were generated with budgets derived from the
    first floor definition, so the file is never rebuilt. New trials reuse exactly the same
    personas, item sets, scenarios and prompt rendering, read back from the frozen trials.
    Trial ids: B-<persona>-k<k>-s<set>-b<budget>."""
    cfg = yaml.safe_load(config_path.read_text())
    path = out_dir / "trials.jsonl"
    trials = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    have = {t["trial_id"] for t in trials}
    # one template per (persona, k, set): the unlimited-budget trial carries system, items, scenario
    templates = {(t["persona"], t["k"], t["set_index"]): t for t in trials if t.get("ratio") == 0}
    new = []
    for (pname, k, si), tpl in sorted(templates.items()):
        for budget in cfg.get("extra_budgets", []):
            tid = f"B-{pname}-k{k}-s{si:02d}-b{budget}"
            if tid in have:
                continue
            items = [d["instruction"] for d in tpl["meta"]["items"]]
            user = render_b_user(items, budget, tpl["unit"], tpl["meta"]["scenario"])
            new.append({**tpl, "trial_id": tid, "cell": f"{pname}|k{k}|b{budget}",
                        "ratio": None, "budget": budget, "budget_mode": "absolute", "user": user})
    with path.open("a") as f:
        for t in new:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"appended {len(new)} absolute-budget trials ({len(trials)} existing kept as is)")
    return path


def tolerance(budget: int) -> int:
    """±10% of the target, at least ±1 word."""
    return max(1, round(0.1 * budget))


def build_b2(config_path: Path = CONFIG / "exp_b2.yaml", out_dir: Path = DATA_BUILT / "exp_b2") -> Path:
    """Experiment B, design 2/3: one budget list shared by all item sets, more item sets. exp_b stays frozen.
    config `budget_mode`: "around" (two-sided "about N words, lo-hi") or "at_most" (one-sided, as exp_b)."""
    cfg = yaml.safe_load(config_path.read_text())
    mode = cfg.get("budget_mode", "around")
    tag = cfg.get("tag", "B2")
    rng = random.Random(cfg["seed"])
    unit = cfg["unit"]
    facets = load_facets()
    personas = {str(name): persona_lines(spec, facets) for name, spec in cfg["personas"].items()}
    item_sets = {k: [sample_item_set(rng, k) for _ in range(cfg["n_item_sets"])] for k in cfg["k_values"] if k > 0}
    scenarios = [B_SCENARIOS[i % len(B_SCENARIOS)] for i in range(cfg["n_item_sets"])]
    trials = []
    for pname, plines in personas.items():
        system = (B_SYSTEM_PERSONA.format(persona_lines=render_persona_lines(plines))
                  if plines else B_SYSTEM_NO_PERSONA)
        for k in cfg["k_values"]:
            for si in range(cfg["n_item_sets"]):
                items = item_sets[k][si] if k > 0 else []
                floor = floor_length(items, unit) if items else None
                for budget in cfg["budgets"]:
                    t = tolerance(budget) if (budget > 0 and mode == "around") else 0
                    lo, hi = (budget - t, budget + t) if (budget > 0 and mode == "around") else (None, None)
                    user = render_b_user([i.instruction for i in items], budget, unit, scenarios[si], lo, hi)
                    trials.append({
                        "trial_id": f"{tag}-{pname}-k{k}-s{si:02d}-b{budget}",
                        "experiment": tag.lower(), "cell": f"{pname}|k{k}|b{budget}",
                        "persona": pname, "k": k, "ratio": None, "budget": budget, "lo": lo, "hi": hi,
                        "budget_mode": mode, "floor": floor, "unit": unit, "set_index": si,
                        "system": system, "user": user, "gen": {"max_tokens": cfg["max_tokens"]},
                        "meta": {"items": [i.to_dict() for i in items], "scenario": scenarios[si],
                                 "persona_lines": plines},
                    })
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "trials.jsonl"
    with out.open("w") as f:
        for tr in trials:
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")
    summary = {"n_trials": len(trials), "budgets": cfg["budgets"], "n_item_sets": cfg["n_item_sets"], "budget_mode": mode,
               "tolerance": {b: tolerance(b) for b in cfg["budgets"] if b > 0} if mode == "around" else None,
               "floors": {k: sorted({floor_length(s_, unit) for s_ in sets}) for k, sets in item_sets.items()},
               "config": cfg}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return out


def build_b4(config_path: Path = CONFIG / "exp_b4.yaml", out_dir: Path = DATA_BUILT / "exp_b4") -> Path:
    """Experiment B, facet design: for each IPIP-NEO facet, a high-pole persona (its +keyed items), a low-pole
    persona (its -keyed items) and no persona, over 10 facet-specific friend messages (config/scenarios_facets.yaml),
    3 required content items per (facet, scenario), one-sided "at most N" budgets."""
    cfg = yaml.safe_load(config_path.read_text())
    scen = yaml.safe_load((CONFIG / "scenarios_facets.yaml").read_text())
    rng = random.Random(cfg["seed"])
    unit = cfg["unit"]
    facets = load_facets()
    trials = []
    for code in sorted(facets, key=lambda c: (c[0], int(c[1:]))):
        msgs = scen[code]
        assert len(msgs) == cfg["n_scenarios"], (code, len(msgs))
        personas = {"high": persona_lines({"facets": [code], "pole": "plus"}, facets),
                    "low": persona_lines({"facets": [code], "pole": "minus"}, facets), "off": None}
        for si, friend in enumerate(msgs):
            items = sample_item_set(rng, cfg["k"])
            floor = floor_length(items, unit)
            for pname, plines in personas.items():
                system = (B_SYSTEM_PERSONA.format(persona_lines=render_persona_lines(plines))
                          if plines else B_SYSTEM_NO_PERSONA)
                for budget in cfg["budgets"]:
                    user = render_b_user([i.instruction for i in items], budget, unit, friend)
                    trials.append({
                        "trial_id": f"B4-{code}-{pname}-s{si:02d}-b{budget}",
                        "ref_id": f"B4-{code}-{pname}-s{si:02d}-b0",
                        "experiment": "b4", "cell": f"{code}-{pname}|k{cfg['k']}|b{budget}",
                        "facet": code, "persona": pname, "k": cfg["k"], "ratio": None, "budget": budget,
                        "lo": None, "hi": None, "budget_mode": "at_most", "floor": floor, "unit": unit,
                        "set_index": si, "system": system, "user": user, "gen": {"max_tokens": cfg["max_tokens"]},
                        "meta": {"facet": code, "facet_name": facets[code].name, "items": [i.to_dict() for i in items],
                                 "scenario": friend, "persona_lines": plines},
                    })
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "trials.jsonl"
    with out.open("w") as f:
        for tr in trials:
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")
    summary = {"n_trials": len(trials), "budgets": cfg["budgets"], "n_scenarios": cfg["n_scenarios"], "k": cfg["k"],
               "floors": sorted({t["floor"] for t in trials}), "n_facets": len(facets), "config": cfg}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return out

def build_b5(config_path: Path = CONFIG / "exp_b5.yaml", out_dir: Path = DATA_BUILT / "exp_b5") -> Path:
    """Complement of exp_b4 to a full factorial: the same facets, poles, scenarios and budgets with k = 0, 1, 2
    required content items (exp_b4 has k = 3). k = 0 with unlimited budget and no persona is the plain reply."""
    cfg = yaml.safe_load(config_path.read_text())
    scen = yaml.safe_load((CONFIG / "scenarios_facets.yaml").read_text())
    rng = random.Random(cfg["seed"])
    unit = cfg["unit"]
    facets = load_facets()
    trials = []
    for code in sorted(facets, key=lambda c: (c[0], int(c[1:]))):
        msgs = scen[code]
        assert len(msgs) == cfg["n_scenarios"], (code, len(msgs))
        personas = {"high": persona_lines({"facets": [code], "pole": "plus"}, facets),
                    "low": persona_lines({"facets": [code], "pole": "minus"}, facets), "off": None}
        for si, friend in enumerate(msgs):
            for k in cfg["ks"]:
                items = sample_item_set(rng, k)
                floor = floor_length(items, unit)
                for pname, plines in personas.items():
                    system = (B_SYSTEM_PERSONA.format(persona_lines=render_persona_lines(plines))
                              if plines else B_SYSTEM_NO_PERSONA)
                    for budget in cfg["budgets"]:
                        user = render_b_user([i.instruction for i in items], budget, unit, friend)
                        trials.append({
                            "trial_id": f"B5-{code}-{pname}-k{k}-s{si:02d}-b{budget}",
                            "ref_id": f"B5-{code}-{pname}-k{k}-s{si:02d}-b0",
                            "experiment": "b5", "cell": f"{code}-{pname}|k{k}|b{budget}",
                            "facet": code, "persona": pname, "k": k, "ratio": None, "budget": budget,
                            "lo": None, "hi": None, "budget_mode": "at_most", "floor": floor, "unit": unit,
                            "set_index": si, "system": system, "user": user, "gen": {"max_tokens": cfg["max_tokens"]},
                            "meta": {"facet": code, "facet_name": facets[code].name, "items": [i.to_dict() for i in items],
                                     "scenario": friend, "persona_lines": plines},
                        })
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "trials.jsonl"
    with out.open("w") as f:
        for tr in trials:
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")
    summary = {"n_trials": len(trials), "budgets": cfg["budgets"], "n_scenarios": cfg["n_scenarios"], "ks": cfg["ks"],
               "floors": sorted({t["floor"] for t in trials}), "n_facets": len(facets), "config": cfg}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return out
