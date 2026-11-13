"""Parse the IPIP-NEO 300 facet key page into facets with +/- keyed items."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import DATA_RAW

KEY_FILE = DATA_RAW / "ipip" / "newNEOFacetsKey.htm"

_FACET_HEAD = re.compile(r"^([NEOAC])([1-6]):\s*([A-Z][A-Z \-/]+)$")


@dataclass
class Facet:
    code: str  # e.g. "A3"
    name: str  # e.g. "ALTRUISM"
    plus: list[str] = field(default_factory=list)
    minus: list[str] = field(default_factory=list)

    @property
    def domain(self) -> str:
        return self.code[0]


def _page_lines(path: Path = KEY_FILE) -> list[str]:
    raw = path.read_text(encoding="latin-1")
    raw = re.sub(r"<script.*?</script>", "", raw, flags=re.DOTALL)
    # an item may wrap onto two lines inside one <td>; join the lines of each cell
    raw = re.sub(r"<td([^>]*)>(.*?)</td>", lambda m: f"<td{m.group(1)}>" + re.sub(r"\s+", " ", m.group(2)) + "</td>", raw, flags=re.DOTALL)
    txt = html.unescape(re.sub(r"<[^>]+>", "\n", raw))
    return [ln.strip() for ln in txt.splitlines() if ln.strip()]


def load_facets(path: Path = KEY_FILE) -> dict[str, Facet]:
    facets: dict[str, Facet] = {}
    cur: Facet | None = None
    mode: str | None = None
    for ln in _page_lines(path):
        m = _FACET_HEAD.match(ln)
        if m:
            cur = Facet(code=m.group(1) + m.group(2), name=m.group(3).strip())
            facets[cur.code] = cur
            mode = None
            continue
        if cur is None:
            continue
        if ln.startswith("+") and "keyed" in ln:
            mode = "plus"
            continue
        if (ln.startswith(("–", "-"))) and "keyed" in ln:
            mode = "minus"
            continue
        if ln.startswith(("(", "=")):  # alpha line fragments
            continue
        if re.fullmatch(r"\(?\.?\d\d\)?", ln) or re.fullmatch(r"\(Alpha", ln):
            continue
        if mode == "plus" and ln.rstrip('"')[-1] in ".!":
            cur.plus.append(ln)
        elif mode == "minus" and ln.rstrip('"')[-1] in ".!":
            cur.minus.append(ln)
    # sanity: 30 facets, 10 items each
    bad = {k: (len(v.plus), len(v.minus)) for k, v in facets.items() if len(v.plus) + len(v.minus) != 10}
    if len(facets) != 30 or bad:
        raise RuntimeError(f"IPIP parse problem: n_facets={len(facets)} bad={bad}")
    return facets
