"""Reproduce every table and figure of the paper from outputs/exp_*/scores.csv.gz and ratings.csv.gz.

    uv run python scripts/make_results.py

Writes results/tables/*.csv (and .md) and results/figs/*.png. Table and figure numbers follow the paper:
  table3_models            evaluated models (family, parameters, output speed)
  table4_baseline          timing baseline: which side was followed when the same kind of instruction was given twice
  table5_single_pair_triple each model at its breaking point with one, two and three instructions
  table6_robustness        robustness under multiple constraints (harmonic mean) and its breakdown
  table7_breaking_point    at the breaking point: kept share, baseline, and their difference per instruction
  table8_examples          replies of five models to the same scene within 5 words
  cells                    the full grid (model x persona on/off x content items x word limit)
  fig1 .. fig5             the figures of the paper
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from pcb.content_items import from_dict

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
TAB = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figs"
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)

JUDGES = ["gpt-oss-120b", "llama-3.3-70b"]
BUDGETS = [3, 5, 8, 10, 15, 20, 30, 0]          # 0 = unlimited
BL = ["3", "5", "8", "10", "15", "20", "30", "∞"]
KEEP_MIN = 0.8                                  # breaking point: both the limit and the content kept at >= 80 %
LOOSE = {3: 25, 5: 25, 8: 25, 10: 25, 15: 25, 30: 50}
MODELS = pd.DataFrame([   # name, family, total B, active B, output tokens/s (A100 80GB, vLLM, 64 concurrent sequences)
    ("gemma-4-31b-it", "Gemma 4 31B", "Gemma", 31.3, 31.3, 396), ("gemma-4-12b-it", "Gemma 4 12B", "Gemma", 12.0, 12.0, 780),
    ("gemma-4-e4b-it", "Gemma 4 E4B", "Gemma", 8.0, 4.0, 1158), ("gemma-4-e2b-it", "Gemma 4 E2B", "Gemma", 5.1, 2.0, 1158),
    ("qwen3.5-27b", "Qwen3.5 27B", "Qwen", 27.8, 27.8, 310), ("qwen3.5-9b", "Qwen3.5 9B", "Qwen", 9.7, 9.7, 787),
    ("qwen3.5-4b", "Qwen3.5 4B", "Qwen", 4.7, 4.7, 879), ("qwen3.5-2b", "Qwen3.5 2B", "Qwen", 2.3, 2.3, 1672),
    ("mistral-small-3.2-24b", "Mistral Small 3.2 24B", "Mistral", 24.0, 24.0, 711), ("ministral-3-8b", "Ministral 3 8B", "Mistral", 8.9, 8.9, 1697),
    ("ministral-3-3b", "Ministral 3 3B", "Mistral", 4.2, 4.2, 3279), ("olmo-3-7b-instruct", "OLMo 3 7B", "OLMo", 7.3, 7.3, 1382),
    ("phi-4-mini", "Phi-4-mini 3.8B", "Phi", 3.8, 3.8, 2347), ("granite-4.2-3b", "Granite 4.2 3B", "Granite", 3.7, 3.7, 1731),
    ("smollm3-3b", "SmolLM3 3B", "SmolLM", 3.1, 3.1, 2179)], columns=["model", "name", "family", "total_b", "active_b", "speed"])
ORDER = list(MODELS.model); NAME = dict(zip(MODELS.model, MODELS.name)); SPEED = dict(zip(MODELS.name, MODELS.speed))
FAM_COLOR = {"Gemma": "#1f77b4", "Qwen": "#ff7f0e", "Mistral": "#2ca02c", "OLMo": "#9467bd", "Phi": "#8c564b", "Granite": "#e377c2", "SmolLM": "#7f7f7f"}
FAMILY = dict(zip(MODELS.name, MODELS.family))
COL = {"pers": "#4C78A8", "len": "#F58518", "con": "#E45756"}
LAB = {"pers": "性格ペルソナ", "len": "語数上限", "con": "発話内容"}


def jp_font() -> None:
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ("IPAexGothic", "Noto Sans CJK JP", "IPAGothic"):
        if cand in names:
            plt.rcParams["font.family"] = cand
            return


def md(df: pd.DataFrame, path: Path, fmt: str = "{:.0f}") -> None:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    path.write_text("\n".join(lines) + "\n")


def save(df: pd.DataFrame, stem: str, fmt: str = "{:.0f}") -> None:
    df.to_csv(TAB / f"{stem}.csv", index=False)
    md(df, TAB / f"{stem}.md", fmt)


# ---------------------------------------------------------------- inputs
def load_scores(exps: tuple[str, ...]) -> pd.DataFrame:
    parts = []
    for exp in exps:
        sc = pd.read_csv(OUT / f"exp_{exp}" / "scores.csv.gz", low_memory=False)
        sc = sc[sc.error.isna() & ~sc.truncated.astype(bool)].copy()
        sc["exp"] = exp
        parts.append(sc)
    return pd.concat(parts, ignore_index=True)


def load_ratings(exps: tuple[str, ...]) -> pd.DataFrame:
    """Facet score per rated reply, averaged over the two judges."""
    parts = [pd.read_csv(OUT / f"exp_{exp}" / "ratings.csv.gz") for exp in exps]
    r = pd.concat(parts, ignore_index=True)
    keys = ["model", "seed", "trial_id", "facet", "persona", "k", "budget", "set_index", "length"]
    return r.groupby(keys).facet_score.mean().reset_index().rename(columns={"facet_score": "S", "set_index": "scenario"})


def expression(r: pd.DataFrame) -> pd.DataFrame:
    """Persona expression = facet score minus the score of the no-persona reply to the same scene (seed 1), signed
    toward the instructed pole."""
    off = r[(r.persona == "off") & (r.seed == 1)].set_index(["model", "facet", "scenario", "k", "budget"]).S
    on = r[r.persona != "off"].copy()
    on["S_off"] = [off.get(key, np.nan) for key in zip(on.model, on.facet, on.scenario, on.k, on.budget)]
    on = on.dropna(subset=["S_off"])
    on["expr"] = np.where(on.persona == "high", on.S - on.S_off, on.S_off - on.S)
    return on


# ---------------------------------------------------------------- main experiment grid
def cell_table(sc: pd.DataFrame, ex: pd.DataFrame) -> pd.DataFrame:
    sc = sc.copy(); sc["pers"] = np.where(sc.persona == "off", "off", "on")
    t = sc.groupby(["model", "pers", "k", "budget"]).agg(n=("length", "size"), mean_len=("length", "mean"),
                                                        length_kept=("compliant", "mean"), content_all=("content_all", "mean")).reset_index()
    e = ex.groupby(["model", "k", "budget"]).agg(expr=("expr", "mean"), n_rated=("expr", "size")).reset_index(); e["pers"] = "on"
    return t.merge(e, on=["model", "pers", "k", "budget"], how="left")


def cell(t: pd.DataFrame, m: str, pers: str, k: int, b: int) -> pd.Series:
    return t[(t.model == m) & (t.pers == pers) & (t.k == k) & (t.budget == b)].iloc[0]


# ---------------------------------------------------------------- timing baseline (exp_c)
def baseline_length(c: pd.DataFrame, b5: pd.DataFrame) -> dict[tuple[str, int], dict]:
    """user_win = share that kept the per-turn side's limit (both orders averaged); keyed by (model, N)."""
    out = {}
    c = c[c.kind == "length"].copy()
    for m in ORDER:
        for n_ in (3, 5, 8, 10, 15, 30):
            pre = "len_" if n_ == 15 else f"len{n_}_"
            c1 = c[(c.model == m) & (c.cell == pre + "sysmax_usermin")]; c2 = c[(c.model == m) & (c.cell == pre + "sysmin_usermax")]
            if c1.empty or c2.empty:
                continue
            loose = LOOSE[n_]
            sys_win = np.mean([(c1.length <= n_).mean(), (c2.length >= loose).mean()])
            user_win = np.mean([(c1.length >= loose).mean(), (c2.length <= n_).mean()])
            out[(m, n_)] = {"sys": sys_win, "user": user_win, "neither": 1 - sys_win - user_win}
    return out


def baseline_content(c: pd.DataFrame) -> dict[str, dict]:
    trials = {t["trial_id"]: t for t in (json.loads(l) for l in (ROOT / "data" / "built" / "exp_c" / "trials.jsonl").read_text().splitlines())
              if t["kind"] == "content"}
    d = c[c.trial_id.isin(trials)].copy()
    d["user_all"] = [all(from_dict(x).hit(str(r.text)) for x in trials[r.trial_id]["meta"]["items"]) for r in d.itertuples()]
    d["sys_all"] = [all(from_dict(x).hit(str(r.text)) for x in trials[r.trial_id]["meta"]["sys_items"]) for r in d.itertuples()]
    out = {}
    for m in ORDER:
        g = d[d.model == m]
        out[m] = {"sys": (g.sys_all & ~g.user_all).mean(), "user": (g.user_all & ~g.sys_all).mean(), "both": (g.sys_all & g.user_all).mean()}
    return out


def baseline_persona(rc: pd.DataFrame, rb: pd.DataFrame, base: dict[str, float]) -> dict[str, dict]:
    """Degree to which the reply leans to the system-side pole (expression toward it divided by the persona-only
    expression, clipped to 0..1) and to the per-turn side pole, averaged over the conflict cells of both orders."""
    trials = {t["trial_id"]: t for t in (json.loads(l) for l in (ROOT / "data" / "built" / "exp_c" / "trials.jsonl").read_text().splitlines())
              if t["kind"] == "personality"}
    c = rc[rc.trial_id.isin(trials)].copy()
    c["cell"] = c.trial_id.map(lambda t: trials[t]["cell"])
    off = rb[(rb.persona == "off") & (rb.k == 0) & (rb.budget == 0) & (rb.seed == 1)].set_index(["model", "facet", "scenario"]).S
    c["S_off"] = [off.get(k_, np.nan) for k_ in zip(c.model, c.facet, c.scenario)]
    c = c.dropna(subset=["S_off"])
    conf = c[c.cell.str.startswith("pers_sys")].copy()
    conf["expr_pole"] = np.where(conf.persona == "high", 1, -1) * (conf.S - conf.S_off)
    out = {}
    for m in ORDER:
        d = conf[conf.model == m]
        sw = np.clip(d.expr_pole / base[m], 0, 1); uw = np.clip(-d.expr_pole / base[m], 0, 1)
        out[m] = {"sys": sw.mean(), "user": uw.mean(), "neither": 1 - sw.mean() - uw.mean()}
    return out


# ---------------------------------------------------------------- figures
def fig_bands(t9: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.4), dpi=170, sharey=True)
    names = list(t9["name"]); y = np.arange(len(names))[::-1]
    for ax, (s_, u_, n_, title) in zip(axes, [("p_sys", "p_user", "p_nei", "性格ペルソナ"), ("l_sys", "l_user", "l_nei", "語数上限"), ("c_sys", "c_user", None, "発話内容")]):
        s = t9[s_].values; u = t9[u_].values; n = t9[n_].values if n_ else np.clip(100 - s - u, 0, None)
        left = np.zeros(len(names))
        for p, lab, col in zip([s, u, n], ["システムプロンプト側", "ターンごとのプロンプト側", "どちらでもない"], ["#4C78A8", "#F58518", "#BAB0AC"]):
            ax.barh(y, p, left=left, color=col, label=lab, height=0.7)
            for yi, l, v in zip(y, left, p):
                if v >= 8:
                    ax.text(l + v / 2, yi, f"{v:.0f}", ha="center", va="center", fontsize=7, color="white")
            left += p
        ax.set_xlim(0, 100); ax.set_xlabel("%"); ax.set_title(title, fontsize=10)
    axes[0].set_yticks(y); axes[0].set_yticklabels(names, fontsize=8)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig1_baseline_bands.png"); plt.close(fig)


def fig_diff(d: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.4), dpi=170, sharey=True)
    y = np.arange(len(d))[::-1]
    for ax, key in zip(axes, ("dp", "dl", "dc")):
        v = d[key].values; k = {"dp": "pers", "dl": "len", "dc": "con"}[key]
        ax.barh(y, v, color=COL[k], height=0.7)
        for yi, vv in zip(y, v):
            ax.text(vv + (2 if vv >= 0 else -2), yi, f"{vv:+.0f}", va="center", ha="left" if vv >= 0 else "right", fontsize=7)
        ax.axvline(0, color="k", lw=0.8); ax.set_xlim(-60, 110); ax.set_title(LAB[k], fontsize=10)
        ax.set_xlabel("3 者で守った割合 − 基準実験の値（ポイント）", fontsize=8); ax.grid(axis="x", alpha=.3)
    axes[0].set_yticks(y); axes[0].set_yticklabels(d["name"], fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "fig2_breaking_point_diff.png"); plt.close(fig)


def fig_surfaces(cells: pd.DataFrame, base: dict[str, float]) -> None:
    reps = [("phi-4-mini", "Phi-4-mini 3.8B", "性格ペルソナを最も強く残す"), ("gemma-4-31b-it", "Gemma 4 31B", "語数上限を最も強く残す"),
            ("qwen3.5-2b", "Qwen3.5 2B", "発話内容を最も強く残す")]

    def grid(m, pers, col, ks):
        Z = np.full((len(ks), len(BUDGETS)), np.nan)
        for i, k in enumerate(ks):
            for j, b in enumerate(BUDGETS):
                r = cells[(cells.model == m) & (cells.pers == pers) & (cells.k == k) & (cells.budget == b)]
                if len(r):
                    Z[i, j] = r[col].iloc[0]
        return Z

    fig = plt.figure(figsize=(8.4, 7.4), dpi=170); X = np.arange(len(BUDGETS))
    for r_, (m, name, tag) in enumerate(reps):
        panels = [("性格の表出（残存率）", np.minimum(grid(m, "on", "expr", [0, 1, 2, 3]) / base[m], 1) * 100, None, [0, 1, 2, 3]),
                  ("語数上限を守った割合", grid(m, "on", "length_kept", [0, 1, 2, 3]) * 100, grid(m, "off", "length_kept", [0, 1, 2, 3]) * 100, [0, 1, 2, 3]),
                  ("発話内容をすべて伝えた割合", grid(m, "on", "content_all", [1, 2, 3]) * 100, grid(m, "off", "content_all", [1, 2, 3]) * 100, [1, 2, 3])]
        for c_, (title, Zon, Zoff, ks) in enumerate(panels):
            ax = fig.add_subplot(3, 3, r_ * 3 + c_ + 1, projection="3d")
            Xg, Yg = np.meshgrid(X, ks)
            ax.plot_surface(Xg, Yg, Zon, cmap="viridis", vmin=0, vmax=100, alpha=0.9, edgecolor="k", lw=0.2)
            if Zoff is not None:
                ax.plot_wireframe(Xg, Yg, Zoff, color="red", lw=0.8)
            ax.set_xticks(X); ax.set_xticklabels(BL, fontsize=6); ax.set_yticks(ks); ax.set_yticklabels([str(k) for k in ks], fontsize=6)
            ax.set_zlim(0, 100); ax.tick_params(axis="z", labelsize=6)
            ax.set_xlabel("語数上限", fontsize=7, labelpad=-2); ax.set_ylabel("項目数", fontsize=7, labelpad=-2)
            ax.view_init(elev=28, azim=-130)
            if r_ == 0:
                ax.set_title(title, fontsize=9, pad=4)
        fig.text(0.012, 0.84 - r_ * 0.325, f"{name}\n{tag}", fontsize=8, rotation=90, va="center", ha="center")
    fig.tight_layout(w_pad=0.5, h_pad=0.2, rect=(0.03, 0, 1, 1))
    fig.savefig(FIG / "fig3_surfaces.png"); plt.close(fig)


def frontier(ax, sp, v):
    o = np.argsort(-sp); best = -1e9; fx, fy = [], []
    for i in o:
        if v[i] > best:
            fx.append(sp[i]); fy.append(v[i]); best = v[i]
    ax.plot(fx[::-1], fy[::-1], "k--", lw=1)


def fig_speed(rob: pd.DataFrame, d: pd.DataFrame, th: int = 70) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=170)
    off_ = {"Gemma 4 12B": (6, 6), "Mistral Small 3.2 24B": (-6, -11), "Qwen3.5 27B": (6, -9), "Qwen3.5 9B": (6, 3), "Qwen3.5 4B": (6, -8),
            "Gemma 4 E2B": (6, -7), "Granite 4.2 3B": (6, -8), "Gemma 4 E4B": (-6, 6), "Ministral 3 8B": (6, 4), "Qwen3.5 2B": (6, -8)}
    ha = {"Mistral Small 3.2 24B": "left", "Gemma 4 E4B": "right"}
    names = list(rob["name"]); sp = np.array([SPEED[n] for n in names]); v = rob.robust.values
    for n, x, yv in zip(names, sp, v):
        c = FAM_COLOR[FAMILY[n]]
        ax.scatter(x, yv, s=38, zorder=3, facecolors=c if yv >= th else "white", edgecolors=c, linewidths=1.2)
        ax.annotate(n, (x, yv), textcoords="offset points", xytext=off_.get(n, (6, 2)), fontsize=7, ha=ha.get(n, "left"))
    frontier(ax, sp, v)
    ax.axhline(th, color="#E45756", lw=1, ls=":"); ax.text(3350, th + 1.5, f"耐性 {th}", color="#E45756", fontsize=8, ha="right", va="bottom")
    ax.set_xlabel("出力速度（トークン/秒）"); ax.set_ylabel("多制約下の耐性（調和平均）"); ax.set_ylim(0, 100); ax.grid(alpha=.3)
    ax.legend(handles=[Line2D([], [], marker="o", color="k", ls="", label=f"耐性 {th} 以上"), Line2D([], [], marker="o", mfc="white", color="k", ls="", label=f"耐性 {th} 未満"),
                       Line2D([], [], color="k", ls="--", label="それより速いモデルに耐性で上回られない点")], fontsize=7, loc="lower left", frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig4_robustness_speed.png"); plt.close(fig)

    rb = dict(zip(rob["name"], rob.robust))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), dpi=170, sharey=True)
    for ax, key in zip(axes, ("dp", "dl", "dc")):
        vv = d[key].values; sp = d["name"].map(SPEED).values; k = {"dp": "pers", "dl": "len", "dc": "con"}[key]
        for n, x, yv in zip(d["name"], sp, vv):
            c = FAM_COLOR[FAMILY[n]]
            ax.scatter(x, yv, s=28, zorder=3, facecolors=c if rb[n] >= th else "white", edgecolors=c, linewidths=1.2)
            ax.annotate(n, (x, yv), textcoords="offset points", xytext=(4, 3), fontsize=6)
        frontier(ax, sp, vv); ax.axhline(0, color="k", lw=0.6)
        ax.set_title(LAB[k], fontsize=10); ax.set_xlabel("出力速度（トークン/秒）"); ax.grid(alpha=.3)
    axes[0].set_ylabel("基準実験の値との差（ポイント）")
    axes[0].legend(handles=[Line2D([], [], marker="o", color="k", ls="", label=f"耐性 {th} 以上"), Line2D([], [], marker="o", mfc="white", color="k", ls="", label=f"耐性 {th} 未満")],
                   fontsize=7, loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "fig5_diff_speed.png"); plt.close(fig)


# ---------------------------------------------------------------- main
def main() -> None:
    jp_font()
    save(MODELS.rename(columns={"name": "モデル", "family": "系統", "total_b": "総(B)", "active_b": "実効(B)", "speed": "速度(tok/s)"}).drop(columns="model"), "table3_models", "{:g}")

    sc = load_scores(("b4", "b5"))
    sc = sc[["model", "trial_id", "seed", "persona", "k", "budget", "set_index", "length", "compliant", "content_all"]].copy()
    sc["k"] = sc.k.astype(int); sc["budget"] = sc.budget.astype(int)
    rb = load_ratings(("b4", "b5"))
    ex = expression(rb)
    cells = cell_table(sc, ex)
    cells.insert(1, "name", cells.model.map(NAME))
    cells.to_csv(TAB / "cells.csv", index=False)
    base = {m: cell(cells, m, "on", 0, 0).expr for m in ORDER}       # persona-only expression (points)

    # --- table 6: robustness
    on = cells[cells.pers == "on"]
    rows = []
    for m in ORDER:
        g = on[(on.model == m) & (on.k >= 1)].copy()
        g["rp"] = np.minimum(g.expr / base[m], 1.0)
        X = g[["rp", "length_kept", "content_all"]].clip(lower=1e-6)
        g["hm"] = 3 / (1 / X).sum(axis=1)
        r = {"model": m, "name": NAME[m], "robust": g.hm.mean() * 100, "pers": g.rp.mean() * 100, "len": g.length_kept.mean() * 100, "con": g.content_all.mean() * 100}
        for b, lab in zip(BUDGETS, BL):
            r[f"b{lab}"] = g[g.budget == b].hm.mean() * 100
        for k in (1, 2, 3):
            r[f"k{k}"] = g[g.k == k].hm.mean() * 100
        rows.append(r)
    rob = pd.DataFrame(rows).sort_values("robust", ascending=False)
    save(rob.drop(columns="model").rename(columns={"name": "モデル", "robust": "耐性", "pers": "性格", "len": "語数", "con": "内容"}), "table6_robustness")

    # --- timing baseline (exp_c)
    c = load_scores(("c",))
    rc = load_ratings(("c",))
    bl = baseline_length(c, sc); bc = baseline_content(c); bp = baseline_persona(rc, rb, base)

    # --- breaking point, table 5, table 7, table 4
    t4, t7, t9 = [], [], []
    for m in ORDER:
        ret = lambda r: min(r.expr / base[m], 1.0) * 100
        N = next(b for b in BUDGETS if cell(cells, m, "off", 3, b).length_kept >= KEEP_MIN and cell(cells, m, "off", 3, b).content_all >= KEEP_MIN)
        c3 = cell(cells, m, "on", 3, N); pl = cell(cells, m, "on", 0, N); pc = cell(cells, m, "on", 3, 0); lc = cell(cells, m, "off", 3, N)
        kp, kl, kc = ret(c3), c3.length_kept * 100, c3.content_all * 100
        tp, tl, tc = bp[m]["sys"] * 100, bl[(m, N)]["user"] * 100, bc[m]["user"] * 100
        t4.append({"モデル": NAME[m], "崩壊点": N, "性格のみ(点)": base[m], "語数のみ": cell(cells, m, "off", 0, N).length_kept * 100,
                   "内容のみ": cell(cells, m, "off", 3, 0).content_all * 100, "性格vs語数:性格": ret(pl), "性格vs語数:語数": pl.length_kept * 100,
                   "性格vs内容:性格": ret(pc), "性格vs内容:内容": pc.content_all * 100, "語数vs内容:語数": lc.length_kept * 100, "語数vs内容:内容": lc.content_all * 100,
                   "3者:性格": kp, "3者:語数": kl, "3者:内容": kc})
        t7.append({"model": m, "name": NAME[m], "N": N, "kp": kp, "tp": tp, "dp": kp - tp, "kl": kl, "tl": tl, "dl": kl - tl, "kc": kc, "tc": tc, "dc": kc - tc})
        t9.append({"name": NAME[m], "p_sys": bp[m]["sys"] * 100, "p_user": bp[m]["user"] * 100, "p_nei": bp[m]["neither"] * 100, "N": N,
                   "l_sys": bl[(m, N)]["sys"] * 100, "l_user": bl[(m, N)]["user"] * 100, "l_nei": bl[(m, N)]["neither"] * 100,
                   "c_sys": bc[m]["sys"] * 100, "c_user": bc[m]["user"] * 100})
    t4 = pd.DataFrame(t4); t7 = pd.DataFrame(t7); t9 = pd.DataFrame(t9)
    t4.to_csv(TAB / "table5_single_pair_triple.csv", index=False)
    md(t4.assign(**{"性格のみ(点)": t4["性格のみ(点)"].map("{:.2f}".format)}), TAB / "table5_single_pair_triple.md")
    save(t9.rename(columns={"name": "モデル", "p_sys": "性格:システム側", "p_user": "性格:ターン側", "p_nei": "性格:どちらでもない", "N": "語数:N",
                            "l_sys": "語数:システム側", "l_user": "語数:ターン側", "l_nei": "語数:どちらでもない", "c_sys": "内容:システム側", "c_user": "内容:ターン側"}), "table4_baseline")
    save(t7.drop(columns="model").rename(columns={"name": "モデル", "N": "崩壊点", "kp": "性格:3者", "tp": "性格:基準", "dp": "性格:差", "kl": "語数:3者", "tl": "語数:基準",
                                                   "dl": "語数:差", "kc": "内容:3者", "tc": "内容:基準", "dc": "内容:差"}), "table7_breaking_point")

    # --- table 8: the same scene within 5 words
    b4 = load_scores(("b4",))
    key = dict(facet="A5", persona="low", set_index=2, seed=3, budget=5)
    rows = []
    for m in ["gemma-4-31b-it", "phi-4-mini", "ministral-3-8b", "qwen3.5-2b", "granite-4.2-3b"]:
        r = b4[(b4.model == m) & (b4["set_index"] == key["set_index"]) & (b4.seed == key["seed"]) & (b4.budget == key["budget"]) & (b4.persona == key["persona"])
               & b4.trial_id.str.startswith(f"B4-{key['facet']}-")].iloc[0]
        e = ex[(ex.model == m) & (ex.trial_id == r.trial_id) & (ex.seed == r.seed)].expr
        rows.append({"モデル": NAME[m], "返答": " ".join(str(r.text).split()), "語数": int(r.length), "5語以内": "○" if r.compliant else "×",
                     "3項目": "○" if r.content_all else "×", "性格の表出量": f"{float(e.iloc[0]):+.1f}" if len(e) else ""})
    save(pd.DataFrame(rows), "table8_examples", "{}")

    # --- figures
    fig_bands(t9); fig_diff(t7); fig_surfaces(cells, base); fig_speed(rob, t7)
    print("tables:", sorted(p.name for p in TAB.iterdir())); print("figs:", sorted(p.name for p in FIG.iterdir()))


if __name__ == "__main__":
    main()
