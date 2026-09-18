#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
 
import csv
import secrets
 
import numpy as np
 
import CORE_Config_LogReg as bc
 
# ==========================================================================
# CONFIGURATION -- everything a run varies is in this block
# ==========================================================================
 
# --- 1. signals -----------------------------------------------------------
# Register any signal beyond those in CORE here. fn receives an (n, active)
# matrix -- already sliced to its active columns -- and returns an (n,) array.
# Then add the key to FUNCTIONS to actually run it.
 
def _g3(X):
    """Example: additive, four active covariates, two of them non-monotone."""
    return (3 * X[:, 0] - 2 * X[:, 1]
            + 4 * np.sin(2 * np.pi * X[:, 2])
            + 6 * (X[:, 3] - 0.5) ** 2)
 
 
CUSTOM_SIGNALS = (
    # (key, function, covariates, active columns)
    (3, _g3, "uniform", 4),
)
 
FUNCTIONS = (1,2)        # which signals to run; add 3 to include _g3 above
 
# --- 2. covariate dimension ----------------------------------------------
PREDICTORS = 5           # total p; columns beyond a signal's active set are noise
 
# --- 3. seeding -----------------------------------------------------------
BASE_SEED = None          # None -> draw a fresh block; int -> reproduce a run
SEED_BLOCK = 10_000        # seeds reserved per run; must exceed ITERATIONS
SEED_BLOCKS = 1_000_000   # how many distinct blocks to draw from
DIAG_REPS = 200           # replications used by flip_diagnostics
 
# --- 4. sample sizes and Monte Carlo effort -------------------------------
OBSERVATIONS = 500        # training rows
TEST_OBS = 2000           # test rows, drawn clean
ITERATIONS = 1000         # replications per cell
ROUNDS = 100              # boosting rounds per fit
 
# --- 5. contamination design ---------------------------------------------
FLIP_RULES = ("uniform", "margin", #"boundary"   # drop "boundary" to save time
              )
SHARES = (0.05, 0.10, 0.15, 0.20)
BAYES = 0.10
PREVALENCE = 0.5
RHO = 0.0                 # equicorrelation; 0.0 keeps the original draw path
INCLUDE_BAD_LEVERAGE = False
 
# --- 6. metrics -----------------------------------------------------------
# Any subset of CORE.METRIC_ORDER, in the order the tables should show them.
# Accuracy and AUC are not computed at all unless one of them is listed
# (CORE's extras flag); the other three are computed and then filtered, which
# costs nothing measurable next to a fit. Dropping a metric removes it from
# the printed tables, the long CSV and the summary alike.
METRICS = ("MSE_true",
           #"Brier",
           #"LogLoss",
           "Accuracy",
           #"AUC"
           )
DEGRADE_ON = "MSE_true"   # metric used by the paired degradation table
 
# --- 7. output ------------------------------------------------------------
VERBOSE_TABLES = True
RUN_DIAGNOSTICS = True
COMBINE_TO_XLSX = True
 
LONG_PATH = "results_flip_g{g}_p{p}_s{seed}_LogReg.csv"
SUMMARY_PATH = "summary_flip_g{g}_p{p}_s{seed}_LogReg.csv"
WORKBOOK = "summary_flip_all_g_p{p}_s{seed}_LogReg.xlsx"
 
# --- 8. model set ---------------------------------------------------------
# CORE's own CONFIGURATION block already lists the models; leave this empty to
# use it as it stands. Anything named here overrides it for THIS RUN ONLY --
# CORE is never edited, and a typo raises rather than being ignored. Accepted
# keys are CORE_Config.MODEL_CONFIG_KEYS; the useful ones are
#
#   FIT_LOGREG, FIT_XGB_LOGISTIC, FIT_HINGE, FIT_MODIFIED_HUBER   True/False
#   BY_EXP_TUNING, BY_ORIG_TUNING       tuples of c; () drops the family
#   BY_EXP_RESCALE, BY_ORIG_RESCALE, BY_FLOOR
#   REFERENCE                           model the paired differences use
 
MODELS = dict(
    # BY_ORIG_TUNING=(0.75, 1.0),
    # FIT_LOGREG=False,
)
 
# ======================= end of configuration =============================
 
 
def apply_config():
    """Register the custom signals and check the design is coherent."""
    for key, fn, cov, active in CUSTOM_SIGNALS:
        bc.register_signal(key, fn, covariates=cov, active=active)
 
    names = bc.configure_models(**MODELS)
    if not names:
        raise ValueError("MODELS leaves nothing to fit")
    if bc.REFERENCE not in names:
        raise ValueError(f"REFERENCE={bc.REFERENCE!r} is not among the fitted "
                         f"models {names}")
 
    for g in FUNCTIONS:
        if g not in bc.SIGNAL:
            raise ValueError(f"g={g} is in FUNCTIONS but not registered")
        d = bc.ACTIVE.get(g)
        if d is not None and PREDICTORS < d:
            raise ValueError(
                f"g{g} reads {d} covariates; PREDICTORS={PREDICTORS} is too narrow")
 
    unknown = [m for m in METRICS if m not in bc.METRIC_ORDER]
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; expected a subset of "
                         f"{bc.METRIC_ORDER}")
    if not METRICS:
        raise ValueError("METRICS is empty; keep at least one")
    if DEGRADE_ON not in METRICS:
        raise ValueError(f"DEGRADE_ON={DEGRADE_ON!r} is not in METRICS")
 
    kinds = {bc.COVARIATES[g] for g in FUNCTIONS}
    if len(kinds) > 1:
        print("NOTE: FUNCTIONS mixes covariate laws " + str(sorted(kinds))
              + "; standard_normal and random consume the stream differently, "
                "so the signals are NOT paired on X. Split the run instead.")
 
 
# ------------------------------------------------------------------ metrics
def extras_wanted():
    """CORE computes Accuracy and AUC only when extras is True."""
    return any(m in ("Accuracy", "AUC") for m in METRICS)
 
 
def run_cell(**kwargs):
    """compare_generate, restricted to METRICS and ordered as listed.
 
    Filtering here rather than inside CORE keeps the objectives, the fits and
    the RNG stream identical to a full run: the same models see the same data,
    only fewer columns are carried downstream. Runs with different METRICS are
    therefore still comparable row by row.
    """
    raw = bc.compare_generate(**kwargs)
    return {model: {m: d[m] for m in METRICS if m in d}
            for model, d in raw.items()}
 
 
def draw_base_seed(block=SEED_BLOCK, n_blocks=SEED_BLOCKS):
    """A random base seed restricted to a grid of multiples of `block`.
 
    compare_generate consumes the seeds base_seed, ..., base_seed + iterations
    - 1, and every signal in FUNCTIONS re-uses that same set. If base seeds
    were drawn freely, two runs could share some replications and not others:
    the pooled results would be neither independent nor paired, and nothing in
    the output would reveal it. A grid coarser than the number of seeds a run
    consumes leaves only two cases, fully disjoint or exactly identical, and
    the second is visible because the seed is written into both files.
 
    Entropy comes from the OS, not from any generator this study seeds.
    """
    need = max(ITERATIONS, DIAG_REPS)
    if block <= need:
        raise ValueError(
            f"SEED_BLOCK={block} must exceed the {need} seeds a run consumes "
            f"(ITERATIONS={ITERATIONS}, DIAG_REPS={DIAG_REPS})")
    return secrets.randbelow(n_blocks) * block
 
 
# ------------------------------------------------- carrying the seed to disk
def add_seed_column():
    """Prepend base_seed to CORE's summary field list, once per process.
 
    save_summary reads bc.SUMMARY_FIELDS at call time and fills any missing
    key with restval="", so naming the column here and putting the value on
    each record is enough; no CORE logic is duplicated.
    """
    if "base_seed" not in bc.SUMMARY_FIELDS:
        bc.SUMMARY_FIELDS = ["base_seed"] + bc.SUMMARY_FIELDS
 
 
def collect_seeded(rows, raw, dgp, kind, share, base_seed, flip_rule=""):
    """bc.collect, with the base seed prepended to each appended record."""
    start = len(rows)
    bc.collect(rows, raw, dgp, kind, share, flip_rule)
    rows[start:] = [(base_seed, *r) for r in rows[start:]]
    return rows
 
 
def summarise_seeded(raw, dgp, kind, share, base_seed, clean=None,
                     flip_rule=""):
    """bc.summarise, with the base seed attached to each record."""
    recs = bc.summarise(raw, dgp, kind, share, clean=clean, flip_rule=flip_rule)
    for rec in recs:
        rec["base_seed"] = base_seed
    return recs
 
 
def save_csv_seeded(rows, path):
    """bc.save_csv with the extra leading column in the header."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["base_seed"] + bc.LONG_FIELDS)
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {path}")
 
 
# --------------------------------------------------------------- diagnostics
def flip_diagnostics(dgp, seed, obs=OBSERVATIONS, shares=SHARES,
                     reps=DIAG_REPS):
    """What each rule actually does, before any model is fitted.
 
    Reports the mean |Bayes margin| of the flipped points and the share of
    flips that were already wrong before flipping (negative margin -- those
    are REPAIRED, not corrupted, and under the uniform rule there are always
    a few).
 
    Run on this run's seeds, so the table describes the data actually fitted.
    X and y at seed + i match replication i, and so do the margin and boundary
    index sets, which are deterministic functions of them. The uniform index
    set does not match: generate() draws the test set before calling
    flip_index and this function does not, so the two consume the stream
    differently. Both are uniform draws from the same population, so the
    statistics are unbiased for the rule; they simply do not name the same
    rows.
    """
    print(f"\n{'=' * 72}\nFlip diagnostics (g{dgp.g}, p={dgp.p}, BE={dgp.bayes})"
          f"\n{'=' * 72}")
    print(f"  {'rule':10s} {'share':>7s} {'mean |margin|':>14s} "
          f"{'mean margin':>12s} {'% repaired':>11s}")
    for rule in FLIP_RULES:
        for share in shares:
            absm, sgn, rep = [], [], []
            for i in range(reps):
                rng = np.random.default_rng(seed + i)
                X = bc.draw_X(rng, obs, dgp.p, dgp.g, dgp.rho)
                eps = bc.draw_eps(rng, obs, dgp)
                y = bc.labels(X, eps, dgp)
                k = max(1, int(round(share * obs)))
                idx = bc.flip_index(rng, X, y, dgp, k, rule)
                m = bc.bayes_margin(X, y, dgp)[idx]
                absm.append(np.abs(m).mean())
                sgn.append(m.mean())
                rep.append((m < 0).mean())
            print(f"  {rule:10s} {share:7.0%} {np.mean(absm):14.4f} "
                  f"{np.mean(sgn):12.4f} {np.mean(rep):11.1%}")
    print()
 
 
# ------------------------------------------------------------- one signal
def run_signal(function, base_seed):
    """Everything the original main() did, for one signal function.
 
    Returns (rows, summary) and writes the two per-signal CSVs. The seeds are
    the caller's, not derived from `function`, so each signal sees the same
    replication streams.
    """
    dgp = bc.make_dgp(function, PREDICTORS, BAYES, PREVALENCE, RHO)
    print(f"\n{'#' * 72}\n### signal g{function}  (p={PREDICTORS})\n{'#' * 72}")
    bc.describe(dgp)
    print("\nFreeze this constant once the design is final:")
    print("CALIBRATED.update({\n" + bc.freeze_line(dgp) + "\n})")
 
    if RUN_DIAGNOSTICS:
        flip_diagnostics(dgp, seed=base_seed)
 
    rows, summary = [], []
    common = dict(iterations=ITERATIONS, observations=OBSERVATIONS,
                  predictors=PREDICTORS, n=ROUNDS, g=function,
                  base_seed=base_seed, test_obs=TEST_OBS, verbose=False,
                  extras=extras_wanted(), bayes=BAYES, prevalence=PREVALENCE,
                  rho=RHO)
 
    # One clean run, shared by every rule. The flip rule is irrelevant when
    # nothing is flipped, so running it once is not an approximation.
    clean = run_cell(**common)
    print(f"\n{'=' * 72}\n=== g{function} | clean ===")
    if VERBOSE_TABLES:
        bc.show(clean)
    collect_seeded(rows, clean, dgp, "clean", 0.0, base_seed)
    summary += summarise_seeded(clean, dgp, "clean", 0.0, base_seed)
 
    for rule in FLIP_RULES:
        for share in SHARES:
            res = run_cell(**common, mislabeled=share, flip_rule=rule)
            print(f"\n=== g{function} | {share:.0%} mislabeled [{rule}] ===")
            if VERBOSE_TABLES:
                bc.show(res)
                bc.degradation(clean, res, DEGRADE_ON)
            collect_seeded(rows, res, dgp, "mislabeled", share, base_seed,
                           flip_rule=rule)
            summary += summarise_seeded(res, dgp, "mislabeled", share, base_seed,
                                        clean=clean, flip_rule=rule)
 
    if INCLUDE_BAD_LEVERAGE:
        for share in SHARES:
            res = run_cell(**common, lp=share, bad_leverage=1)
            print(f"\n=== g{function} | {share:.0%} bad leverage ===")
            if VERBOSE_TABLES:
                bc.show(res)
                bc.degradation(clean, res, DEGRADE_ON)
            collect_seeded(rows, res, dgp, "bad leverage", share, base_seed)
            summary += summarise_seeded(res, dgp, "bad leverage", share,
                                        base_seed, clean=clean)
 
    tag = dict(g=function, p=PREDICTORS, seed=base_seed)
    save_csv_seeded(rows, LONG_PATH.format(**tag))
    bc.save_summary(summary, SUMMARY_PATH.format(**tag))
    return rows, summary
 
 
# --------------------------------------------------------------- workbook
def write_workbook(paths, path):
    """Drop the per-signal summary CSVs into one workbook, one sheet each.
 
    Deliberately unformatted: this only saves the manual step of opening the
    files. Run format_summary.py over the CSVs instead if you want the colour
    bands.
    """
    try:
        import pandas as pd
    except ImportError:
        print("pandas not available; skipping the workbook")
        return
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for function, csv_path in paths.items():
            pd.read_csv(csv_path).to_excel(xl, sheet_name=f"g{function}",
                                           index=False)
    print(f"wrote {len(paths)} sheets to {path}")
 
 
# ---------------------------------------------------------------------- main
def main():
    apply_config()
    base_seed = draw_base_seed() if BASE_SEED is None else int(BASE_SEED)
    add_seed_column()
 
    models = bc.model_names()
    per_signal = 1 + len(FLIP_RULES) * len(SHARES) \
        + (len(SHARES) if INCLUDE_BAD_LEVERAGE else 0)
    cells = per_signal * len(FUNCTIONS)
    active = {g: (bc.ACTIVE.get(g) or PREDICTORS) for g in FUNCTIONS}
 
    print(f"{'=' * 72}")
    print(f"base_seed = {base_seed}"
          + ("" if BASE_SEED is None else "   (fixed by hand)"))
    print(f"  replications use seeds {base_seed} .. {base_seed + ITERATIONS - 1}")
    print(f"  the same seeds are re-used for every signal in {FUNCTIONS}")
    print(f"  set BASE_SEED = {base_seed} to reproduce this run exactly")
    print(f"  settings: p={PREDICTORS} obs={OBSERVATIONS} test_obs={TEST_OBS} "
          f"rounds={ROUNDS} iterations={ITERATIONS} rho={RHO}")
    print("  active covariates: "
          + ", ".join(f"g{g}={d} (+{PREDICTORS - d} noise)"
                      for g, d in active.items()))
    print(f"  flip rules: {FLIP_RULES}   metrics: {METRICS}")
    print(f"  {cells} cells x {ITERATIONS} replications x {len(models)} models "
          f"= {cells * ITERATIONS * len(models)} fits")
    print(f"{'=' * 72}")
    bc.describe_models()
    print(f"{'=' * 72}")
 
    summary_paths = {}
    for function in FUNCTIONS:
        run_signal(function, base_seed)
        summary_paths[function] = SUMMARY_PATH.format(
            g=function, p=PREDICTORS, seed=base_seed)
 
    if COMBINE_TO_XLSX:
        write_workbook(summary_paths,
                       WORKBOOK.format(p=PREDICTORS, seed=base_seed))
 
    print(f"\nbase_seed = {base_seed} (first column of every file)")
 
 
if __name__ == "__main__":
    main()
