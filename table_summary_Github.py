#!/usr/bin/env python3
"""Turn a formatted summary workbook into booktabs LaTeX tables.

Reads a workbook produced by format_summary.py (band rows and all) and writes
one table per (signal function, metric) combination: rows are loss functions,
columns are contamination shares, and each contamination rule gets its own
panel with the clean column repeated at the left.

Usage
-----
Set SOURCE below and run this file. Or pass paths on the command line:

    tabulate_summary.py in_formatted.xlsx [out.tex]

Output goes to "<source stem>.tex" by default and is also printed to stdout.
"""

import os
import re
import sys
import pandas as pd

# ---------------------------------------------------------------- run config
SOURCE = "FILE_NAME.xlsx"      # a workbook written by format_summary.py
OUTPUT = "NEW_NAME.docx"      # blank -> "<source stem>.tex"
FLOAT = True     # wrap each table in \begin{table}...\end{table}
# -----------------------------------------------------------------------------

# Manuscript labels. Losses absent from this mapping are left out of the
# tables: LogReg is a context baseline, not one of the compared objectives.
MODELS = [
    ("logistic", "Logistic"),
    ("hinge", "Hinge"),
    ("Modified_Huber", "Modified Huber"),
    (None, None),                      # rule between the two families
    ("BY_exp_0.5", "BY-CH"),
    ("BY_orig_0.75", "BY"),
]

# Panels, in order, as (contamination, flip_rule, title). flip_rule None means
# the rule column is empty for that contamination.
PANELS = [
    ("mislabeled", "uniform", "uniform mislabeling"),
    ("mislabeled", "margin", "margin-targeted mislabeling"),
    ("bad leverage", None, "bad leverage points"),
]

METRICS = {
    # metric: (label suffix, caption noun, direction phrase, mean dp, se dp)
    "MSE_true": ("mse", "MSE against the true probabilities", "lower is better", 3, 4),
    "Brier": ("brier", "Brier score", "lower is better", 3, 4),
    "Accuracy": ("acc", "Accuracy", "higher is better", 3, 4),
    "AUC": ("auc", "AUC", "higher is better", 3, 4),
    "LogLoss": ("logloss", "Log loss", "lower is better", 3, 4),
}

# The metric each table cross-references for its simulation settings: the
# first metric present in a sheet carries the full description, the rest
# point back to it.
PRIMARY_METRIC = "MSE_true"


def read_sheet(path, sheet):
    """Read one formatted sheet, dropping the coloured band rows."""
    df = pd.read_excel(path, sheet_name=sheet)
    df = df[df["model"].notna()].copy()
    df["share"] = df["share"].astype(float)
    return df


def tex_escape(text):
    """Underscores in sheet names would break out of text mode."""
    return text.replace("_", r"\_")


def slug(text):
    """Sheet name as a label fragment: no underscores, no spaces."""
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()


def design_of(sheet):
    """g1 -> 1. Sheets that aren't a signal function keep their own name."""
    m = re.fullmatch(r"g(\d+)", sheet.strip())
    return int(m.group(1)) if m else None


def cell(mean, se, mdp, sdp):
    if pd.isna(mean):
        return "---"
    if pd.isna(se):
        return f"{mean:.{mdp}f}"
    return f"{mean:.{mdp}f} ({se:.{sdp}f})"


def panel_rows(df, metric, contamination, flip_rule, shares, mdp, sdp):
    """One panel's body lines, or None if this panel isn't in the data."""
    sub = df[(df["metric"] == metric) & (df["contamination"] == contamination)]
    if flip_rule is None:
        sub = sub[sub["flip_rule"].isna()]
    else:
        sub = sub[sub["flip_rule"] == flip_rule]
    if sub.empty:
        return None

    clean = df[(df["metric"] == metric) & (df["contamination"] == "clean")]
    lines = []
    for key, label in MODELS:
        if key is None:
            lines.append(r"\midrule")
            continue
        if key not in set(df["model"]):
            continue
        cells = []
        for s in shares:
            src = clean if s == 0.0 else sub
            row = src[(src["model"] == key) & (src["share"] == s)]
            if row.empty:
                cells.append("---")
            else:
                cells.append(cell(row["mean"].iloc[0], row["se"].iloc[0], mdp, sdp))
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    # a trailing rule from the family separator would collide with \bottomrule
    while lines and lines[-1] == r"\midrule":
        lines.pop()
    return lines


def fmt_share(s):
    return "$0$" if s == 0 else f"${s:.2f}$"


def caption_for(metric, design, sheet, is_primary):
    noun, direction = METRICS[metric][1], METRICS[metric][2]
    where = f"design $g_{design}$" if design else f"the {tex_escape(sheet)} data"
    if is_primary:
        body = (f"{noun} under mislabeling, {where}. "
                f"Mean over replications with the standard error in "
                f"parentheses; {direction}.")
    else:
        ref = f"tab:flip-g{design}-{METRICS[PRIMARY_METRIC][0]}" if design \
            else f"tab:flip-{slug(sheet)}-{METRICS[PRIMARY_METRIC][0]}"
        body = (f"{noun} under mislabeling, {where}. Simulation settings and "
                f"data identical to Table~\\ref{{{ref}}}; {direction}.")
    return body


def table_for(df, metric, sheet, is_primary):
    design = design_of(sheet)
    suffix, _, _, mdp, sdp = METRICS[metric]
    label = f"tab:flip-g{design}-{suffix}" if design \
        else f"tab:flip-{slug(sheet)}-{suffix}"

    shares = sorted(set(df.loc[df["metric"] == metric, "share"]))
    ncol = len(shares)

    panels = []
    for contamination, flip_rule, title in PANELS:
        rows = panel_rows(df, metric, contamination, flip_rule, shares, mdp, sdp)
        if rows is not None:
            panels.append((title, rows))
    if not panels:
        return None

    out = []
    if FLOAT:
        out += [r"\begin{table}[htbp]", r"\centering"]
    out += [
        rf"\caption{{{caption_for(metric, design, sheet, is_primary)}}}",
        rf"\label{{{label}}}",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{l*{{{ncol}}}{{c}}}}",
        r"\toprule",
        rf"\textbf{{Loss function}} & \multicolumn{{{ncol}}}{{c}}"
        rf"{{\textbf{{Contamination share $\varepsilon$}}}} \\",
        rf"\cmidrule(lr){{2-{ncol + 1}}}",
        " & " + " & ".join(fmt_share(s) for s in shares) + r" \\",
        r"\midrule",
    ]

    for i, (title, rows) in enumerate(panels):
        out.append(r"\midrule")
        if i:
            out.append(r"\addlinespace[4pt]")
        out.append(rf"\multicolumn{{{ncol + 1}}}{{l}}{{\textit{{Panel "
                   rf"{chr(65 + i)}: {title}}}}} \\[2pt]")
        out += rows

    out += [r"\bottomrule", r"\end{tabular}"]
    if FLOAT:
        out.append(r"\end{table}")
    return "\n".join(out)


def build(src, dst):
    sheets = [s for s in pd.ExcelFile(src).sheet_names if s != "legend"]
    blocks = []
    for sheet in sheets:
        df = read_sheet(src, sheet)
        metrics = [m for m in df["metric"].unique() if m in METRICS]
        primary = PRIMARY_METRIC if PRIMARY_METRIC in metrics else metrics[0]
        for metric in sorted(metrics, key=lambda m: m != primary):
            tab = table_for(df, metric, sheet, metric == primary)
            if tab:
                blocks.append(tab)
                print(f"{sheet} / {metric}: table written")
    tex = "\n\n".join(blocks) + "\n"
    with open(dst, "w") as fh:
        fh.write(tex)
    print(f"wrote {dst} ({len(blocks)} tables)")
    return tex


def main():
    args = sys.argv[1:]
    src = args[0] if args else SOURCE
    if not src:
        print(__doc__)
        print("Nothing to do: set SOURCE at the top of this file, "
              "or pass a path on the command line.")
        return
    dst = (args[1] if len(args) > 1 else None) or OUTPUT \
        or os.path.splitext(src)[0] + ".tex"
    build(src, dst)


if __name__ == "__main__":
    main()
