"""
plot_contamination_paths.py

Evolution of MSE_true and Accuracy across contamination levels, read from a
formatted summary workbook.

Produces two figures, one per metric, each pairing the uniform panel with the
margin-targeted panel.  Both panels of a figure share the same y-limits, so the
two contamination schemes are directly comparable.

Usage
-----
    python plot_contamination_paths.py [summary.xlsx] [outdir]
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ settings

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "FILE_NAME.xlsx")
OUTDIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
SHEET = "g1"
EXT = "pdf"

METRICS = ["MSE_true", "Accuracy"]
RULES = ["uniform", "margin"]

# One curve per loss function: a single tuning constant each, the values
# reported in the paper's simulation tables.
MODELS = {
    "logistic":       dict(label="Logistic",           color="black", ls="--", marker="o"),
    "hinge":          dict(label="Hinge",              color="C0",    ls="--", marker="s"),
    "Modified_Huber": dict(label="Modified Huber",     color="C1",    ls="-.", marker="^"),
    "BY_exp_0.5":     dict(label="BY-exp ($d=0.5$)",   color="C2",    ls="-",  marker="D"),
    "BY_orig_0.75":   dict(label="BY-orig ($c=0.75$)", color="C3",    ls="-",  marker="v"),
}

# Losses whose sigmoid-transformed scores carry no probabilistic meaning:
# hinge is improper under any link, Modified Huber is proper composite only
# under its own linear link.  Set True to plot them on MSE_true regardless.
SHOW_IMPROPER_ON_PROB_METRIC = False
IMPROPER = {"hinge", "Modified_Huber"}
PROB_METRICS = {"MSE_true", "Brier", "LogLoss"}

ERRORBAND = 1.0          # half-width of the shaded band, in standard errors
FIGSIZE = (9.6, 4.0)     # per figure (two panels)

METRIC_LABEL = {
    "MSE_true": r"MSE$_{\mathrm{true}}$",
    "Accuracy": "Accuracy",
    "Brier": "Brier score",
    "LogLoss": "Log loss",
    "AUC": "AUC",
}
RULE_LABEL = {"uniform": "uniform flipping", "margin": "margin-targeted flipping"}


# ------------------------------------------------------------------ data prep

def load(src=SRC, sheet=SHEET):
    df = pd.read_excel(src, sheet_name=sheet)
    df = df[df["model"].notna()].copy()          # drop the band header rows
    for col in ("mean", "se", "share"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def series(df, metric, rule, model):
    """Contamination path for one (metric, rule, model): shares, means, SEs.

    The clean cell (share = 0) is common to both rules and is prepended, so
    every path starts from the uncontaminated model.
    """
    clean = df[(df["metric"] == metric)
               & (df["contamination"] == "clean")
               & (df["model"] == model)]
    cont = df[(df["metric"] == metric)
              & (df["contamination"] == "mislabeled")
              & (df["flip_rule"] == rule)
              & (df["model"] == model)]
    out = pd.concat([clean, cont]).sort_values("share")
    return out["share"].to_numpy(), out["mean"].to_numpy(), out["se"].to_numpy()


def plotted(metric):
    """Models shown for a given metric."""
    return [m for m in MODELS
            if not (metric in PROB_METRICS and m in IMPROPER
                    and not SHOW_IMPROPER_ON_PROB_METRIC)]


def limits(df, metric, rules=RULES, pad=0.05):
    """Common y-limits for a metric, over every rule and model plotted."""
    lo, hi = np.inf, -np.inf
    for rule in rules:
        for model in plotted(metric):
            _, mu, se = series(df, metric, rule, model)
            if mu.size == 0:
                continue
            lo = min(lo, np.nanmin(mu - ERRORBAND * se))
            hi = max(hi, np.nanmax(mu + ERRORBAND * se))
    span = hi - lo
    return lo - pad * span, hi + pad * span


# ------------------------------------------------------------------ plotting

def panel(ax, df, metric, rule, ylim, xshares, show_ylabel=True):
    for model in plotted(metric):
        sty = MODELS[model]
        x, mu, se = series(df, metric, rule, model)
        if mu.size == 0:
            continue
        ax.plot(x, mu, color=sty["color"], linestyle=sty["ls"],
                marker=sty["marker"], markersize=4.5, linewidth=1.7,
                label=sty["label"], zorder=3)
        ax.fill_between(x, mu - ERRORBAND * se, mu + ERRORBAND * se,
                        color=sty["color"], alpha=0.14, linewidth=0, zorder=2)

    ax.set_title(RULE_LABEL.get(rule, rule).capitalize(), fontsize=10.5)
    ax.set_xlabel(r"contamination share $\varepsilon$")
    if show_ylabel:
        ax.set_ylabel(METRIC_LABEL.get(metric, metric))
    ax.set_xlim(min(xshares) - 0.005, max(xshares) + 0.005)
    ax.set_xticks(xshares)
    ax.set_ylim(*ylim)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def make_figure(df, metric, ylim, xshares, outdir=OUTDIR, ext=EXT):
    """One figure per metric: the two contamination schemes side by side."""
    fig, axes = plt.subplots(1, len(RULES), figsize=FIGSIZE, sharey=True)
    axes = np.atleast_1d(axes)
    for i, (ax, rule) in enumerate(zip(axes, RULES)):
        panel(ax, df, metric, rule, ylim, xshares, show_ylabel=(i == 0))

    fig.suptitle(METRIC_LABEL.get(metric, metric), fontsize=12)

    # One legend for the figure, taking the panel that shows every curve.
    fullest = max(axes, key=lambda a: len(a.get_legend_handles_labels()[1]))
    handles, labels = fullest.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))

    outdir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^0-9A-Za-z]+", "_", metric).strip("_").lower()
    path = outdir / f"paths_{slug}.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"wrote {path}")
    return fig


def main():
    df = load()
    xshares = sorted(
        df.loc[df["contamination"].isin(["clean", "mislabeled"]), "share"].unique())
    for metric in METRICS:
        make_figure(df, metric, limits(df, metric), xshares)
    plt.show()


if __name__ == "__main__":
    main()