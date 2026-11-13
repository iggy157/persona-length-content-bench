"""Rule-based normalisation of PersonaChat / Dialogue NLI surface text.

The corpus is lower-cased, tokenised with spaces before punctuation, and has
apostrophes removed ("i m", "don t", "nurses aide"). We repair only what can be
repaired deterministically and leave the rest; residual noise is part of the
stimulus and is identical across models.
"""

from __future__ import annotations

import re

_CONTRACTIONS = [
    (r"\bi m\b", "I'm"),
    (r"\bi ve\b", "I've"),
    (r"\bi ll\b", "I'll"),
    (r"\bi d\b", "I'd"),
    (r"\b(\w+) s\b", r"\1's"),  # that s -> that's (also possessives)
    (r"\b(\w+) re\b", r"\1're"),
    (r"\b(\w+) ve\b", r"\1've"),
    (r"\b(\w+) ll\b", r"\1'll"),
    (r"\b(do|does|did|is|are|was|were|have|has|had|would|should|could|ca|wo|ai) n t\b", r"\1n't"),
    (r"\b(do|does|did|is|are|was|were|have|has|had|would|should|could|ca|wo|ai)n t\b", r"\1n't"),
    (r"\bdon t\b", "don't"),
    (r"\bcan t\b", "can't"),
    (r"\bwon t\b", "won't"),
    (r"\bisn t\b", "isn't"),
]


def normalize(text: str) -> str:
    t = " ".join(text.strip().split())
    for pat, rep in _CONTRACTIONS:
        t = re.sub(pat, rep, t)
    # remove space before punctuation, keep space after
    t = re.sub(r"\s+([.,!?;:])", r"\1", t)
    t = re.sub(r"\s+'\s*", "'", t)
    # standalone i -> I
    t = re.sub(r"\bi\b", "I", t)
    # capitalise sentence starts
    parts = re.split(r"(?<=[.!?])\s+", t)
    parts = [p[:1].upper() + p[1:] if p else p for p in parts]
    t = " ".join(parts)
    if t and t[-1] not in ".!?":
        t += "."
    return t


def to_second_person(item: str) -> str:
    """Convert an IPIP item stem ("Am afraid of many things.") to a persona line
    addressed to the model ("You are afraid of many things.")."""
    s = item.strip().rstrip(".")
    s = re.sub(r"\bmyself\b", "yourself", s)
    s = re.sub(r"\bmy\b", "your", s)
    s = re.sub(r"\bme\b", "you", s)
    s = re.sub(r"\bI\b", "you", s)
    low = s[:1].lower() + s[1:]
    if low.startswith("am "):
        return "You are " + low[3:] + "."
    return "You " + low + "."
