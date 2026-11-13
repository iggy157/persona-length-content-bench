"""Experiment B required-content items and the deterministic length floor.

Each item type has
- a natural instruction phrase shown to the model,
- a regex that counts as "item conveyed" in the output,
- a *minimal rendering*: the shortest string that satisfies the regex.

The length floor of an item set is the length (in the configured unit) of the
minimal renderings joined by single spaces.  Budgets are expressed as multiples
of the floor, so ratio 1.0 means "exactly enough room for the content and
nothing else".  This is a definition, not a tuning choice.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ContentItem:
    kind: str
    instruction: str      # how the item is stated to the model
    pattern: str          # regex (case-insensitive) that detects the item in a reply
    minimal: str          # minimal rendering satisfying the pattern
    value: str            # the slot value, for bookkeeping

    def to_dict(self) -> dict:
        return asdict(self)

    def hit(self, text: str) -> bool:
        return re.search(self.pattern, text, flags=re.IGNORECASE) is not None


_TIMES = [(h, "am") for h in (8, 9, 10, 11)] + [(h, "pm") for h in (1, 2, 3, 4, 5, 6)]
_ROOMS = [101, 102, 103, 104, 105, 201, 202, 203, 204, 205, 301, 302, 303, 304, 305]
_THINGS = ["laptop", "umbrella", "charger", "textbook", "passport", "notebook",
           "headphones", "calculator", "jacket", "ticket"]
_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _num_word(n: int) -> str:
    return {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
            8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}[n]


def make_time(h: int, ap: str) -> ContentItem:
    pat = rf"\b(?:{h}|{_num_word(h)})\s*(?::\s*00)?\s*(?:{ap}|{ap[0]}\.?{ap[1]}\.?|o'?clock)\b"
    return ContentItem(kind="time", instruction=f"it starts at {h} {ap}", pattern=pat,
                       minimal=f"{h}{ap}", value=f"{h}{ap}")


def make_room(n: int) -> ContentItem:
    pat = rf"\b(?:room|rm\.?)?\s*{n}\b"
    # the detector accepts the bare number, so the minimal rendering is the bare number
    return ContentItem(kind="room", instruction=f"it is in room {n}", pattern=pat,
                       minimal=f"{n}", value=str(n))


def make_bring(thing: str) -> ContentItem:
    pat = rf"\b{thing}s?\b"
    # the detector accepts the bare noun, so the minimal rendering is the bare noun
    art = "" if thing.endswith("s") else ("an " if thing[0] in "aeiou" else "a ")
    return ContentItem(kind="bring", instruction=f"they should bring {art}{thing}", pattern=pat,
                       minimal=f"{thing}", value=thing)


def make_day(day: str) -> ContentItem:
    pat = rf"\b{day[:3]}(?:{day[3:]})?\.?\b"
    return ContentItem(kind="day", instruction=f"it has been moved to {day}", pattern=pat,
                       minimal=day, value=day)


KINDS = ("time", "room", "bring", "day")


def from_dict(d: dict) -> ContentItem:
    """Rebuild an item from a stored record via its factory, so pattern and minimal rendering
    always follow the current definitions rather than what was serialised at build time."""
    kind, value = d["kind"], d["value"]
    if kind == "time":
        return make_time(int(value[:-2]), value[-2:])
    if kind == "room":
        return make_room(int(value))
    if kind == "bring":
        return make_bring(value)
    if kind == "day":
        return make_day(value)
    raise ValueError(kind)


def sample_item_set(rng: random.Random, k: int) -> list[ContentItem]:
    """k distinct item kinds, fixed order time -> room -> bring -> day for stable floors."""
    kinds = rng.sample(KINDS, k)
    kinds.sort(key=KINDS.index)
    items = []
    for kind in kinds:
        if kind == "time":
            h, ap = rng.choice(_TIMES)
            items.append(make_time(h, ap))
        elif kind == "room":
            items.append(make_room(rng.choice(_ROOMS)))
        elif kind == "bring":
            items.append(make_bring(rng.choice(_THINGS)))
        else:
            items.append(make_day(rng.choice(_DAYS)))
    return items


def count_words(text: str) -> int:
    return len([t for t in re.split(r"\s+", text.strip()) if re.search(r"\w", t)])


def count_chars(text: str) -> int:
    return len(text.strip())


def measure(text: str, unit: str) -> int:
    return count_words(text) if unit == "words" else count_chars(text)


def floor_length(items: list[ContentItem], unit: str) -> int:
    """Shortest reply (in `unit`) that satisfies every item's detector: the minimal renderings
    joined by single spaces. Each `minimal` must be the shortest string its own pattern accepts."""
    return measure(" ".join(i.minimal for i in items), unit)


def budgets_from_floor(floor: int, ratios: list[float]) -> list[int]:
    return [max(1, math.ceil(floor * r)) if r > 0 else 0 for r in ratios]  # 0 = unlimited
