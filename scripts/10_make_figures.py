"""Step 10 — result figures for the paper (PNG 300 dpi + vector PDF) from the result tables.

    python scripts/10_make_figures.py

Colours: validated categorical palette (slots 1–4, CVD-safe on adjacent pairs; slots 3–4
are below 3:1 contrast on white, so every figure also uses hatching and a legend, and
exact values are in the tables). Hatching keeps the figures readable in greyscale print.

  fig_retrieval_main            nDCG@10 per system, with condensed-list scores for non-pooled runs
  fig_field_ablation            metadata fields, standard and condensed-list scores side by side
  fig_language_ablation         bilingual vs English-only vs Arabic-only metadata, by query language
  fig_hallucination_abstention  ungrounded citations and false abstentions per model x condition
  fig_k_ablation                number of retrieved records (models in generation.k_ablation_models)
"""
from __future__ import annotations

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from _common import init, parser, results_path  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
HATCH = ["", "//", "..", "xx"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

SYSTEM_NAMES = {"bm25": "BM25", "dense:minilm_en": "MiniLM (English only)", "dense:mminilm": "Multilingual MiniLM",
                "dense:me5_base": "mE5-base", "dense:bge_m3": "BGE-M3",
                "hybrid:bm25+me5_base": "BM25 + mE5 (RRF)", "hybrid:bm25+bge_m3": "BM25 + BGE-M3 (RRF)",
                "bge_m3": "BGE-M3"}
CONDITION_NAMES = {"norag_plain": "No retrieval, plain prompt", "norag_grounded": "No retrieval, grounded prompt",
                   "rag_plain": "Retrieval, plain prompt", "rag_grounded": "Retrieval, grounded prompt"}
METADATA_NAMES = {"all": "Arabic + English metadata", "en": "English metadata only", "ar": "Arabic metadata only"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 100, "hatch.color": "white", "hatch.linewidth": 0.8,
})


def model_name(m: str) -> str:
    m = re.sub(r"-instruct.*$", "", str(m))
    fam, _, size = m.partition(":")
    fam = {"llama3.2": "Llama 3.2", "qwen2.5": "Qwen2.5"}.get(fam, fam)
    return f"{fam} {size.upper()}".strip()


def ci(s: str) -> tuple[float, float]:
    m = re.findall(r"[-\d.]+", str(s))
    return (float(m[0]), float(m[1])) if len(m) >= 2 else (np.nan, np.nan)


def save(fig, cfg, name):
    for ext in ("png", "pdf"):
        fig.savefig(results_path(cfg, "figures", f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def pct_axis(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))


# ---------------------------------------------------------------------------- retrieval
def retrieval_main(cfg, metric):
    p = results_path(cfg, "tables", "retrieval_main.csv")
    if not p.exists():
        return
    df = pd.read_csv(p).sort_values(metric).reset_index(drop=True)
    lo_hi = np.array([ci(x) for x in df[f"{metric}_ci"]])
    has_pool = "pooled" in df
    marks = [("" if pl == "yes" else " †" if pl == "no" else " ‡") for pl in df["pooled"]] if has_pool else [""] * len(df)
    labels = [SYSTEM_NAMES.get(s, s) + mk for s, mk in zip(df["system"], marks)]
    fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(df) + 1.1))
    y = np.arange(len(df))
    ax.barh(y, df[metric], height=0.55, color=SERIES[0], edgecolor="white", linewidth=1)
    ax.errorbar(df[metric], y, xerr=[df[metric] - lo_hi[:, 0], lo_hi[:, 1] - df[metric]],
                fmt="none", ecolor=INK, elinewidth=1, capsize=3)
    cond = f"{metric}_cond"
    has_cond = cond in df and has_pool
    if has_cond:
        other = df[df["pooled"] != "yes"]
        ax.scatter(other[cond], other.index, marker="D", s=34, facecolor="white", edgecolor=INK, linewidth=1.2, zorder=4)
    right = lo_hi[:, 1].copy()
    if has_cond:                                     # keep the value label clear of the condensed marker
        right = np.where(df["pooled"] != "yes", np.fmax(right, df[cond].values + 0.015), right)
    for yi, (v, xr) in enumerate(zip(df[metric], right)):
        ax.text(xr + 0.012, yi, f"{v:.3f}", va="center", color=INK, fontsize=8)
    ax.set_yticks(y, labels)
    ax.set_xlabel(f"nDCG@10, mean and 95% bootstrap CI ({int(df['n_queries'].iloc[0])} answerable queries)")
    ax.set_xlim(0, min(1.0, float(np.nanmax(right)) + 0.1))
    ax.grid(axis="y", visible=False)
    if has_cond:
        fig.legend(handles=[Patch(facecolor=SERIES[0], label="nDCG@10 (unjudged = non-relevant)"),
                            Line2D([], [], marker="D", ls="", mfc="white", mec=INK, label="condensed-list nDCG@10")],
                   ncol=2, fontsize=8, loc="lower center", bbox_to_anchor=(0.55, -0.06))
        fig.text(0.01, -0.11, "† not in the judgment pool    ‡ not pooled for Arabic queries", fontsize=7.5, color=INK2)
    save(fig, cfg, "fig_retrieval_main")


def field_ablation(cfg, metric):
    p = results_path(cfg, "tables", "retrieval_field_ablation.csv")
    if not p.exists():
        return
    df = pd.read_csv(p)
    order = cfg["ablation_templates"]
    cond = f"{metric}_cond"
    panels = [(metric, "A. Standard scoring (unjudged = non-relevant)")]
    if cond in df:
        panels.append((cond, "B. Condensed lists (unjudged removed)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels) + 1.6, 2.9), sharey=True, sharex=True,
                             squeeze=False)
    systems = list(dict.fromkeys(df["system"]))
    for ax, (col, title) in zip(axes[0], panels):
        for i, system in enumerate(systems):
            grp = df[df["system"] == system].set_index("template").reindex(order)
            y = np.arange(len(order)) + (i - (len(systems) - 1) / 2) * 0.22
            cic = f"{col}_ci"
            if cic in grp:
                err = np.array([ci(s) for s in grp[cic]])
                ax.errorbar(grp[col], y, xerr=[grp[col] - err[:, 0], err[:, 1] - grp[col]], fmt="none",
                            ecolor=SERIES[i], elinewidth=1.5, capsize=0)
            ax.plot(grp[col], y, "o", color=SERIES[i], ms=7, mec="white", mew=1.5)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.set_xlabel("nDCG@10 (95% CI)")
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(np.arange(len(order)), [" + ".join(cfg["templates"][t]) for t in order])
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=SERIES[i], mec="white", ms=7,
                               label=SYSTEM_NAMES.get(s, s)) for i, s in enumerate(systems)],
               ncol=len(systems), fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, cfg, "fig_field_ablation")


def language_ablation(cfg, metric):
    p = results_path(cfg, "tables", "retrieval_language_ablation.csv")
    if not p.exists():
        return
    df = pd.read_csv(p)
    df = df[df["query_language"].isin(["ar", "en"])]
    systems = list(dict.fromkeys(df["system"]))
    variants = [v for v in ["all", "en", "ar"] if v in set(df["metadata_languages"])]
    fig, axes = plt.subplots(1, len(systems), figsize=(3.6 * len(systems) + 0.6, 3.0), sharey=True, squeeze=False)
    width = 0.8 / len(variants)
    for ax, system in zip(axes[0], systems):
        sub = df[df["system"] == system]
        x = np.arange(2)
        for i, v in enumerate(variants):
            cells = [sub[(sub["metadata_languages"] == v) & (sub["query_language"] == q)] for q in ("ar", "en")]
            vals = [c[metric].mean() for c in cells]
            pos = x + (i - (len(variants) - 1) / 2) * width
            ax.bar(pos, vals, width * 0.9, color=SERIES[i], hatch=HATCH[i], edgecolor="white", linewidth=0.8)
            for xp, val, c in zip(pos, vals, cells):
                # value label; * = drop significant after Holm in both the full and the condensed list,
                # (*) = significant in the full list only
                mark = ""
                if "p_value_holm" in c and len(c) and c["p_value_holm"].notna().any():
                    full_sig = c["p_value_holm"].iloc[0] < 0.05
                    cond_sig = c.get("p_value_cond_holm", pd.Series([np.nan])).iloc[0] < 0.05
                    mark = "*" if full_sig and cond_sig else "(*)" if full_sig else ""
                ax.text(xp, val + 0.015, f"{val:.2f}{mark}", ha="center", va="bottom", fontsize=6.5, color=INK)
        ax.set_xticks(x, ["Arabic queries", "English queries"])
        ax.set_title(SYSTEM_NAMES.get(system, system), color=INK, fontsize=9, loc="left")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, 1)
    axes[0][0].set_ylabel("nDCG@10")
    fig.legend(handles=[Patch(facecolor=SERIES[i], hatch=HATCH[i], edgecolor="white", label=METADATA_NAMES.get(v, v))
                        for i, v in enumerate(variants)],
               ncol=len(variants), fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, cfg, "fig_language_ablation")


# ---------------------------------------------------------------------------- generation
def hallucination(cfg):
    p = results_path(cfg, "tables", "generation_main.csv")
    if not p.exists():
        return
    df = pd.read_csv(p)
    models = list(dict.fromkeys(df["model"]))
    conds = [c for c in cfg["generation"]["conditions"] if c in set(df["condition"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.5))
    width = 0.8 / len(conds)
    for ax, col, cicol, title in ((axes[0], "hallucination_rate", "hallucination_ci",
                                   "A. Answers with an ungrounded citation"),
                                  (axes[1], "false_abstention_rate", "false_abstention_ci",
                                   "B. Abstentions on answerable queries")):
        x = np.arange(len(models))
        for i, c in enumerate(conds):
            sub = df[df["condition"] == c].set_index("model").reindex(models)
            vals = sub[col].values.astype(float)
            err = np.array([ci(s) for s in sub[cicol]])
            pos = x + (i - (len(conds) - 1) / 2) * width
            ax.bar(pos, np.nan_to_num(vals), width * 0.9, color=SERIES[i], hatch=HATCH[i], edgecolor="white", linewidth=0.8)
            ok = ~np.isnan(vals)
            ax.errorbar(pos[ok], vals[ok], yerr=[vals[ok] - err[ok, 0], err[ok, 1] - vals[ok]],
                        fmt="none", ecolor=INK, elinewidth=0.8, capsize=2)
            for xp in pos[~ok]:                      # every answer was an abstention: no rate to show
                ax.text(xp, 0.015, "n/a", rotation=90, ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xticks(x, [model_name(m) for m in models])
        ax.set_ylim(0, 1.02)
        pct_axis(ax)
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.grid(axis="x", visible=False)
    fig.legend(handles=[Patch(facecolor=SERIES[i], hatch=HATCH[i], edgecolor="white", label=CONDITION_NAMES.get(c, c))
                        for i, c in enumerate(conds)],
               ncol=len(conds), fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.text(0.01, -0.05, "Ungrounded = a cited work not found in the catalog, or attributed to the wrong authors "
             "(among answered queries). n/a: the model abstained on every query. Error bars: 95% Wilson intervals.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, cfg, "fig_hallucination_abstention")


def k_ablation(cfg):
    p = results_path(cfg, "tables", "generation_k_ablation.csv")
    if not p.exists():
        return
    df = pd.read_csv(p)
    keep = [m for m in cfg["generation"].get("k_ablation_models", []) if m in set(df["model"])] or \
        [m for m, g in df.groupby("model") if g["k"].nunique() > 1]
    df = df[df["model"].isin(keep)]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.9))
    for ax, col, cicol, label in ((axes[0], "hallucination_rate", "ci", "A. Answers with an ungrounded citation"),
                                  (axes[1], "false_abstention_rate", "fa_ci", "B. Abstentions on answerable queries")):
        for i, model in enumerate(keep):
            grp = df[df["model"] == model].sort_values("k")
            if cicol in grp:
                err = np.array([ci(s) for s in grp[cicol]])
                ax.fill_between(grp["k"], err[:, 0], err[:, 1], color=SERIES[i], alpha=0.12, linewidth=0)
            ax.plot(grp["k"], grp[col], marker="o", ms=7, mec="white", mew=1.5, lw=2, color=SERIES[i])
        ax.set_xlabel("records given to the model (k)")
        ax.set_xticks(sorted(df["k"].unique()))
        ax.set_ylim(0, 1)
        pct_axis(ax)
        ax.set_title(label, color=INK, fontsize=9, loc="left")
    fig.legend(handles=[Line2D([], [], marker="o", color=SERIES[i], mec="white", ms=7, lw=2, label=model_name(m))
                        for i, m in enumerate(keep)],
               ncol=len(keep), fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.text(0.01, -0.07, "Retrieval with the grounded prompt; shaded bands are 95% Wilson intervals.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, cfg, "fig_k_ablation")


def main() -> None:
    cfg, _, log = init(parser(__doc__))
    metric = cfg["evaluation"]["main_metric"]
    retrieval_main(cfg, metric)
    field_ablation(cfg, metric)
    language_ablation(cfg, metric)
    hallucination(cfg)
    k_ablation(cfg)
    log.info(f"figures → {results_path(cfg, 'figures')}")


if __name__ == "__main__":
    main()
