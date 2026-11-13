"""Programmatic metrics for Experiment B: length compliance and content recall.
Persona expression is measured by the pairwise LLM judge (judge.py), not here."""

from __future__ import annotations

from .content_items import ContentItem, measure


def judge(reply: str, items: list[ContentItem], budget: int, unit: str,
          lo: int | None = None, hi: int | None = None) -> dict:
    """lo/hi given -> two-sided target (exp_b2): compliant inside [lo, hi]; `length_score` = 1 - |len - N| / N
    clipped to [0, 1] (LIFEBench-style deviation score). Otherwise one-sided (exp_b)."""
    length = measure(reply, unit)
    hits = [it.hit(reply) for it in items]
    if budget == 0:
        compliant, overshoot, lscore = True, 0.0, 1.0
    elif lo is not None and hi is not None:
        compliant = lo <= length <= hi
        overshoot = max(0, length - hi) / budget
        lscore = max(0.0, 1 - abs(length - budget) / budget)
    else:
        compliant = length <= budget
        overshoot = max(0, length - budget) / budget
        lscore = 1.0 if compliant else max(0.0, 1 - (length - budget) / budget)
    return {
        "length": length,
        "compliant": compliant,
        "overshoot": overshoot,
        "length_score": lscore,
        "undershoot": (max(0, lo - length) / budget) if (lo is not None and budget) else 0.0,
        "content_hits": hits,
        "content_recall": (sum(hits) / len(items)) if items else None,
        "content_all": all(hits) if items else None,
        "empty_reply": not reply.strip(),
    }
