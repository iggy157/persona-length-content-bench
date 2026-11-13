"""Experiment C: the system-vs-user position baseline (config/exp_c.yaml).

Experiments A and B always place the persona in the system message and the competing instruction in the user
message, so "the persona loses" and "the system message loses" cannot be told apart. Here the *same kind* of
instruction is given in both positions with incompatible values, in both orders, and the reply shows which position
the model follows. Single-instruction controls place one instruction alone in either position (the ones that already
exist in exp_a / exp_b5 are not rebuilt).

Two trial files, so that the existing pipelines apply unchanged:

  exp_c  (friend-texting frame of experiment B)
    length       : system "at most 15 words" vs user "at least 25 words", and swapped; controls at-most / at-least
                   alone in the system message and at-least alone in the user message (at-most alone in the user
                   message = exp_b5 off/k0/b15). 300 facet scenarios, persona off, no content items.
    personality  : system = one pole of a facet, user = the other pole (operator block), and swapped; controls = one
                   pole in the user message alone (system alone = exp_b5 high/low, k0, unlimited). Rated with the
                   observer rating (pcb.rate) like exp_b4/b5.
  exp_ca (chat-partner frame of experiment A; 168 verified DNLI contradiction pairs s1 / s2)
    content      : system "tell your partner s1" vs user "tell your partner s2", and swapped; control = tell alone
                   in the system message (alone in the user message = exp_a nopersona_content).
    setting      : system persona s1 (+ 3 fillers) vs user operator block "Your persona: s2", and swapped;
                   control = persona alone in the user message (alone in the system = exp_a persona_only).
    Scored with the DNLI classifier and the experiment-A judge; meta.s1 / meta.s2 keep the experiment-A roles and
    `system_side` records which of them sits in the system message.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import yaml

from . import CONFIG, DATA_BUILT
from .content_items import ContentItem, make_day, make_room, make_time
from .ipip import load_facets
from .prompts import (
    A_SYSTEM_NO_PERSONA,
    A_SYSTEM_PERSONA,
    A_USER_NO_CONTENT,
    A_USER_WITH_CONTENT,
    B_CONTENT_BLOCK,
    B_SYSTEM_NO_PERSONA,
    B_SYSTEM_PERSONA,
    B_USER,
    render_persona_lines,
)
from .textnorm import to_second_person

# ---------------------------------------------------------------- templates specific to experiment C
C_OPERATOR_IN_SYSTEM = """\
{base}

[Instruction from the operator]
{line}"""

C_LEN_MAX = "Your reply must be at most {n} words."
C_LEN_MIN = "Your reply must be at least {n} words."
C_PERSONALITY_BLOCK = "Your personality:\n{lines}"
C_PERSONA_BLOCK = "Your persona:\n{lines}"
A_TELL = 'In your next message, tell your chat partner the following about yourself: "{content}"'
A_USER_OPERATOR = """\
[Instruction from the operator]
{block}

[Message from your chat partner]
{question}

Write only your reply to your chat partner."""


def _b_user(operator_block: str | None, friend: str) -> str:
    block = (operator_block.rstrip("\n") + "\n") if operator_block else "Reply to your friend.\n"
    return B_USER.format(content_block=block, length_line="", friend_message=friend)


def _facet_lines(facets, code: str, pole: str) -> list[str]:
    items = facets[code].plus if pole == "high" else facets[code].minus
    return [to_second_person(i) for i in items]


def build_c(config_path: Path = CONFIG / "exp_c.yaml", out_dir: Path = DATA_BUILT / "exp_c") -> Path:
    cfg = yaml.safe_load(config_path.read_text())
    scen = yaml.safe_load((CONFIG / "scenarios_facets.yaml").read_text())
    facets = load_facets()
    tight, loose = cfg["length"]["tight"], cfg["length"]["loose"]
    trials: list[dict] = []
    base = {"experiment": "c", "k": 0, "ratio": None, "lo": None, "hi": None, "floor": 0, "unit": "words",
            "gen": {"max_tokens": cfg["max_tokens"]}}
    for code in sorted(facets, key=lambda c: (c[0], int(c[1:]))):
        msgs = scen[code]
        assert len(msgs) == cfg["length"]["n_scenarios"], (code, len(msgs))
        for si, friend in enumerate(msgs):
            common = {"facet": code, "set_index": si, "meta": {"facet": code, "facet_name": facets[code].name, "items": [],
                                                              "scenario": friend}}
            # ---- length: the budget field carries the "at most" value (0 when there is none) so metrics_b applies
            length_cells = {
                "len_sysmax_usermin": (C_OPERATOR_IN_SYSTEM.format(base=B_SYSTEM_NO_PERSONA, line=C_LEN_MAX.format(n=tight)),
                                       _b_user(C_LEN_MIN.format(n=loose), friend), tight, "max", "min"),
                "len_sysmin_usermax": (C_OPERATOR_IN_SYSTEM.format(base=B_SYSTEM_NO_PERSONA, line=C_LEN_MIN.format(n=loose)),
                                       _b_user(C_LEN_MAX.format(n=tight), friend), tight, "min", "max"),
                "len_sysmax_only": (C_OPERATOR_IN_SYSTEM.format(base=B_SYSTEM_NO_PERSONA, line=C_LEN_MAX.format(n=tight)),
                                    _b_user(None, friend), tight, "max", None),
                "len_sysmin_only": (C_OPERATOR_IN_SYSTEM.format(base=B_SYSTEM_NO_PERSONA, line=C_LEN_MIN.format(n=loose)),
                                    _b_user(None, friend), 0, "min", None),
                "len_usermin_only": (B_SYSTEM_NO_PERSONA, _b_user(C_LEN_MIN.format(n=loose), friend), 0, None, "min"),
            }
            for cell, (system, user, budget, sys_side, user_side) in length_cells.items():
                trials.append({**base, **common, "trial_id": f"C-{code}-s{si:02d}-{cell}", "cell": cell, "kind": "length",
                               "persona": "off", "budget": budget, "budget_mode": "at_most", "system": system, "user": user,
                               "system_side": sys_side, "user_side": user_side, "tight": tight, "loose": loose})
            for xt, xl in (cfg["length"].get("extra_tights") or {}).items():
                xt, xl = int(xt), int(xl)
                extra = {f"len{xt}_sysmax_usermin": (C_OPERATOR_IN_SYSTEM.format(base=B_SYSTEM_NO_PERSONA, line=C_LEN_MAX.format(n=xt)),
                                                     _b_user(C_LEN_MIN.format(n=xl), friend), "max", "min"),
                         f"len{xt}_sysmin_usermax": (C_OPERATOR_IN_SYSTEM.format(base=B_SYSTEM_NO_PERSONA, line=C_LEN_MIN.format(n=xl)),
                                                     _b_user(C_LEN_MAX.format(n=xt), friend), "min", "max")}
                for cell, (system, user, sys_side, user_side) in extra.items():
                    trials.append({**base, **common, "trial_id": f"C-{code}-s{si:02d}-{cell}", "cell": cell, "kind": "length",
                                   "persona": "off", "budget": xt, "budget_mode": "at_most", "system": system, "user": user,
                                   "system_side": sys_side, "user_side": user_side, "tight": xt, "loose": xl})
            # ---- personality: `persona` names the system pole (or "off"); `user_pole` the user pole (or None)
            high, low = _facet_lines(facets, code, "high"), _facet_lines(facets, code, "low")
            pers_cells = {
                "pers_syshigh_userlow": (B_SYSTEM_PERSONA.format(persona_lines=render_persona_lines(high)),
                                         _b_user(C_PERSONALITY_BLOCK.format(lines=render_persona_lines(low)), friend), "high", "low"),
                "pers_syslow_userhigh": (B_SYSTEM_PERSONA.format(persona_lines=render_persona_lines(low)),
                                         _b_user(C_PERSONALITY_BLOCK.format(lines=render_persona_lines(high)), friend), "low", "high"),
                "pers_userhigh_only": (B_SYSTEM_NO_PERSONA,
                                       _b_user(C_PERSONALITY_BLOCK.format(lines=render_persona_lines(high)), friend), "off", "high"),
                "pers_userlow_only": (B_SYSTEM_NO_PERSONA,
                                      _b_user(C_PERSONALITY_BLOCK.format(lines=render_persona_lines(low)), friend), "off", "low"),
            }
            for cell, (system, user, sys_pole, user_pole) in pers_cells.items():
                trials.append({**base, **common, "trial_id": f"C-{code}-s{si:02d}-{cell}", "cell": cell, "kind": "personality",
                               "persona": sys_pole, "user_pole": user_pole, "budget": 0, "budget_mode": "at_most",
                               "system": system, "user": user,
                               "meta": {**common["meta"], "persona_lines": high if sys_pole == "high" else (low if sys_pole == "low" else None),
                                        "user_persona_lines": high if user_pole == "high" else low}})
    if cfg.get("content_b"):
        trials += _content_b_trials(cfg, scen, facets, base)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "trials.jsonl"
    with out.open("w") as f:
        for tr in trials:
            f.write(json.dumps(tr, ensure_ascii=False) + "\n")
    cells = sorted({t["cell"] for t in trials})
    (out_dir / "summary.json").write_text(json.dumps({"n_trials": len(trials), "cells": cells, "config": cfg}, indent=2, ensure_ascii=False))
    return out


def _contradicting_pair(rng: random.Random, kinds: list[str]) -> tuple[list[ContentItem], list[ContentItem]]:
    """Two item sets over the same attributes with different values (so they cannot both be true)."""
    from .content_items import _DAYS, _ROOMS, _TIMES
    a, b = [], []
    for kind in kinds:
        if kind == "time":
            (h1, p1), (h2, p2) = rng.sample(_TIMES, 2); a.append(make_time(h1, p1)); b.append(make_time(h2, p2))
        elif kind == "room":
            r1, r2 = rng.sample(_ROOMS, 2); a.append(make_room(r1)); b.append(make_room(r2))
        elif kind == "day":
            d1, d2 = rng.sample(_DAYS, 2); a.append(make_day(d1)); b.append(make_day(d2))
        else:
            raise ValueError(kind)
    return a, b


def _content_b_trials(cfg: dict, scen: dict, facets, base: dict) -> list[dict]:
    """content (friend frame): system "tell your friend all of: set 1" vs user "tell ... set 2" and swapped; persona off,
    unlimited length. meta.items = the USER-side set (so evaluate's content_all is the user side), meta.sys_items = the
    system side; pcb.report_c scores both sides from the text."""
    cb = cfg["content_b"]; rng = random.Random(cb["seed"]); trials = []
    for code in sorted(facets, key=lambda c: (c[0], int(c[1:]))):
        msgs = scen[code]
        assert len(msgs) == cb["n_scenarios"], (code, len(msgs))
        for si, friend in enumerate(msgs):
            s1, s2 = _contradicting_pair(rng, list(cb["kinds"]))
            block = lambda items: B_CONTENT_BLOCK.format(items="\n".join(f"- {i.instruction}" for i in items))
            cells = {"con_sys1_user2": (s1, s2), "con_sys2_user1": (s2, s1)}
            for cell, (sys_items, user_items) in cells.items():
                system = C_OPERATOR_IN_SYSTEM.format(base=B_SYSTEM_NO_PERSONA, line=block(sys_items).rstrip("\n"))
                user = B_USER.format(content_block=block(user_items), length_line="", friend_message=friend)
                trials.append({**base, "k": len(user_items), "facet": code, "set_index": si, "trial_id": f"C-{code}-s{si:02d}-{cell}",
                               "cell": cell, "kind": "content", "persona": "off", "budget": 0, "budget_mode": "at_most",
                               "system": system, "user": user, "system_side": cell[4:8], "user_side": cell[9:],
                               "meta": {"facet": code, "facet_name": facets[code].name, "scenario": friend,
                                        "items": [i.to_dict() for i in user_items], "sys_items": [i.to_dict() for i in sys_items]}})
    return trials
