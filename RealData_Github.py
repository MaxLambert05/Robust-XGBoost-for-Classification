#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 11:06:04 2026

@author: Max

Real-data study, uniform label flipping only.
"""

import csv
import secrets
from collections import namedtuple
from functools import partial

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

import CORE
from CORE import (BY_EXP_RESCALE, BY_EXP_TUNING, BY_ORIG_RESCALE,
                  BY_ORIG_TUNING, BY_exp, BY_orig, COMMON, PROB_SCALE,
                  RAW_SCALE, flip, modified_huber, summarise, summary_key,
                  SUMMARY_FIELDS, ROUND_TO, show, degradation)

# =========================================================================
#                              CONFIGURATION
# =========================================================================

# ---- the one-per-run switch --------------------------------------------
DATASET_BREAST = True      # True  -> sklearn breast cancer (Wisconsin, n=569)
                           # False -> OpenML id 37, Pima diabetes  (n=768)

# ---- replication design -------------------------------------------------
REPLICATIONS = 100         # repeated stratified splits
TEST_FRAC = 0.30           # held-out share; contamination NEVER touches it
N_ROUNDS = 100             # boosting rounds, as in the simulation

# ---- contamination grid -------------------------------------------------
EPS_GRID = (0.05, 0.10, 0.20, 0.30)     # share of TRAINING labels flipped
FLIP_RULE = "uniform"      # the only rule in this script; written to the
                           # flip_rule column so the output stays schema-
                           # compatible with the simulation files

NESTED_FLIPS = False       # False reproduces CORE: an independent draw at
                           # each epsilon. True makes the flip sets nested
                           # (eps=.10 set is a subset of the eps=.20 set),
                           # which removes one source of noise from the
                           # degradation *curve* but breaks comparability
                           # with the simulation tables. Leave False unless
                           # you change the simulation too.

# ---- Hessian handling ---------------------------------------------------
HESSIAN_RULE = "reject"    # "reject" (primary) | "floor" | "raw"

MH_LINK = "linear"         # "linear" | "blank"

# ---- metrics ------------------------------------------------------------
# The single switch for which columns exist. Comment a line out and that
# metric disappears from the console print, the degradation tables, the long
# CSV and the summary CSV; nothing else needs touching. Order here is the
# order it prints in. An unrecognised name raises rather than being ignored.
METRICS = (
    #"Brier",
    "Accuracy",
    # "LogLoss",
    # "AUC"
)

CLIP = 1e-12
# ---- dataset-specific options -------------------------------------------
PIMA_ZEROS_AS_NAN = False  # Pima encodes missing glucose / blood pressure /
                           # skin thickness / insulin / BMI as 0, which are
                           # physiologically impossible. False keeps the raw
                           # benchmark coding (what most published comparisons
                           # use); True recodes to NaN, which XGBoost routes
                           # natively. This changes the difficulty of the task
                           # and therefore every level in the table, though it
                           # is applied identically to all ten models.

BREAST_MALIGNANT_POSITIVE = False   # False keeps sklearn's coding (1=benign,
                           # prevalence .627). True flips to malignant=1
                           # (prevalence .373). Brier and Accuracy are
                           # symmetric under this relabelling, so it changes
                           # only how prevalence reads in the write-up and how
                           # far base_score = 0.5 sits from the marginal rate.

# ---- seeding ------------------------------------------------------------
# Same convention as the other daughters: a random but reproducible base seed
# on a block grid, so two runs are either fully disjoint or fully identical,
# never partially overlapping. Paste a printed seed here to reproduce a run.
BASE_SEED = None           # None -> draw one; or set an int

# ---- output -------------------------------------------------------------
WRITE_CSV = True
OUT_PREFIX = "real"        # files: <prefix>_<dataset>_long.csv / _summary.csv


# =========================================================================
#                                DATA
# =========================================================================

RealDgp = namedtuple("Dgp", "g p rho bayes prevalence tau sd")
# Field names deliberately mirror CORE.Dgp so summarise() and summary_key()
# work unmodified. `g` carries the dataset name; the CSV header renames it.


def load_dataset():
    """Returns (X, y, name, feature_names). X float64 ndarray, y in {0, 1}."""
    if DATASET_BREAST:
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer(return_X_y=False, as_frame=True)
        X = d.data.to_numpy(dtype=np.float64)
        y = d.target.to_numpy().astype(int)          # 1 = benign, 0 = malignant
        if BREAST_MALIGNANT_POSITIVE:
            y = 1 - y
        name = "breast_cancer"
        cols = list(d.data.columns)
    else:
        from sklearn.datasets import fetch_openml
        d = fetch_openml(data_id=37, as_frame=True)   # cached in ~/scikit_learn_data
        X = d.data.to_numpy(dtype=np.float64)
        y = (d.target == "tested_positive").astype(int).to_numpy()
        cols = list(d.data.columns)
        if PIMA_ZEROS_AS_NAN:
            impossible = ["plas", "pres", "skin", "insu", "mass"]
            for c in impossible:
                if c in cols:
                    j = cols.index(c)
                    X[X[:, j] == 0.0, j] = np.nan
        name = "pima_diabetes"
    return X, y, name, cols


def describe_dataset(X, y, name, cols):
    prev = float(y.mean())
    majority = max(prev, 1.0 - prev)
    n_nan = int(np.isnan(X).sum())
    print(f"dataset      {name}")
    print(f"  n = {X.shape[0]}   p = {X.shape[1]}   "
          f"P(y=1) = {prev:.4f}   majority-class rate = {majority:.4f}")
    if n_nan:
        print(f"  {n_nan} missing cells routed natively by XGBoost")
    print(f"  split        {1 - TEST_FRAC:.0%}/{TEST_FRAC:.0%} stratified, "
          f"{REPLICATIONS} repetitions")
    print(f"  Accuracy floor for a constant predictor is {majority:.4f}; "
          f"read the Accuracy column against that, not against 0.5.")
    return prev


# =========================================================================
#                        OBJECTIVES / HESSIAN RULE
# =========================================================================

def _apply_rule(obj, rule):
    """Wrap a CORE objective with the chosen negative-Hessian rule.

    reject  zero BOTH g_i and h_i where h_i < 0. The observation is dropped
            from the split search entirely, so exact and hist agree.
    floor   zero h_i only, keep g_i. CORE's built-in floor=0 behaviour.
    raw     no intervention; tree_method-dependent and reported as an ablation.
    """
    if rule == "floor":
        return partial(obj, floor=0.0)
    if rule == "raw":
        return partial(obj, floor=-np.inf)
    if rule != "reject":
        raise ValueError(f"unknown HESSIAN_RULE={rule!r}")

    raw_obj = partial(obj, floor=-np.inf)

    def rejected(predt, dtrain):
        g, h = raw_obj(predt, dtrain)
        bad = h < 0.0
        if bad.any():
            g, h = g.copy(), h.copy()
            g[bad] = 0.0
            h[bad] = 0.0
        return g, h

    return rejected


def train_models_real(X_tr, y_tr, n=N_ROUNDS, seed=0, rule=HESSIAN_RULE):
    """Mirror of CORE.train_models, with the Hessian rule made explicit.

    Model names, order and hyperparameters are identical to CORE so that the
    summary sort key and the Excel layout carry over without edits.
    """
    train = xgb.DMatrix(X_tr, label=y_tr)
    p = {**COMMON, "seed": seed}
    models = {}

    models["logistic"] = xgb.train(
        {**p, "objective": "binary:logistic", "base_score": PROB_SCALE}, train, n)
    models["hinge"] = xgb.train(
        {**p, "objective": "binary:hinge", "base_score": RAW_SCALE}, train, n)
    models["Modified_Huber"] = xgb.train(
        {**p, "base_score": RAW_SCALE}, train, n, obj=modified_huber)

    for c in BY_EXP_TUNING:
        models[f"BY_exp_{c}"] = xgb.train(
            {**p, "base_score": RAW_SCALE}, train, n,
            obj=_apply_rule(partial(BY_exp, c=c, rescale=BY_EXP_RESCALE), rule))

    for c in BY_ORIG_TUNING:
        models[f"BY_orig_{c}"] = xgb.train(
            {**p, "base_score": RAW_SCALE}, train, n,
            obj=_apply_rule(partial(BY_orig, c=c, rescale=BY_ORIG_RESCALE), rule))

    return models


# =========================================================================
#                        PREDICTION / METRICS
# =========================================================================

def predict_real(X_te, models):
    """Scores, probabilities and 0/1 predictions, with the per-loss link.

    Differs from CORE.predict in one respect that matters: CORE pushes every
    non-logistic margin through expit, which is the wrong link for Modified
    Huber and a meaningless one for hinge. Here the link is explicit and
    probabilities is None where no link exists.
    """
    test = xgb.DMatrix(X_te)
    scores, probs, preds = {}, {}, {}

    for name, model in models.items():
        # float64 at source. XGBoost returns float32, and np.float32(1 - 1e-12)
        # rounds to exactly 1.0, so an upper clip at 1 - 1e-12 is a silent
        # no-op in single precision. That is harmless for Brier but sends
        # log1p(-p) to -inf, and (1 - y) * -inf to NaN, for any probability
        # that lands exactly on 1. The Modified Huber linear link produces
        # exactly 1.0 for every margin >= 1, which here is most of the test
        # set, so this fired on every replication.
        if name == "logistic":
            pr = np.asarray(model.predict(test), dtype=np.float64)
            s = np.log(np.clip(pr, 1e-12, 1 - 1e-12)
                       / np.clip(1 - pr, 1e-12, 1 - 1e-12))
        else:
            s = np.asarray(model.predict(test, output_margin=True),
                           dtype=np.float64)
            if name == "hinge":
                pr = None                                  # improper: no link
            elif name == "Modified_Huber":
                pr = (None if MH_LINK == "blank"
                      else (np.clip(s, -1.0, 1.0) + 1.0) / 2.0)
            else:
                pr = 1.0 / (1.0 + np.exp(-s))              # BY_* keep the logit link
        scores[name] = s
        probs[name] = pr
        preds[name] = 1 * (s > 0)

    return scores, probs, preds


def evaluate_real(pred_tuple, y_te, metrics=METRICS):
    """Compute exactly the metrics named in METRICS, in that order.

    NaN, not a missing key, is used where a loss has no probability, so that
    every model carries the same metric set and show()/summarise() do not have
    to special-case ragged dictionaries. The Excel step already renders these
    as an em dash.

    LogLoss warning. Under MH_LINK = "linear" the Modified Huber probability
    is exactly 0 or 1 wherever |margin| >= 1, which is most of the test set.
    Log loss is then -log(CLIP) on every saturated-and-wrong point, so the
    reported number is roughly 27.6 * (saturated error rate): a function of
    the clip constant, not of the model. Brier is bounded and has no such
    dependence. Report Brier; do not report LogLoss for Modified Huber under
    the linear link, whatever the clip is set to.
    """
    scores, probs, preds = pred_tuple
    out = {}
    for name in preds:
        pr = probs[name]
        p = None if pr is None else np.clip(pr, CLIP, 1.0 - CLIP)

        m = {}
        for metric in metrics:
            if metric == "Accuracy":
                m[metric] = float(accuracy_score(y_te, preds[name]))
            elif metric == "AUC":
                m[metric] = float(roc_auc_score(y_te, scores[name]))
            elif metric == "Brier":
                m[metric] = (np.nan if p is None
                             else float(np.mean((p - y_te) ** 2)))
            elif metric == "LogLoss":
                m[metric] = (np.nan if p is None
                             else float(np.mean(-(y_te * np.log(p)
                                                  + (1 - y_te) * np.log1p(-p)))))
            else:
                raise ValueError(
                    f"unknown metric {metric!r} in METRICS; expected one of "
                    f"Brier, Accuracy, LogLoss, AUC")
        out[name] = m
    return out


# =========================================================================
#                          CONTAMINATION
# =========================================================================

def flip_index_real(rng, k, n, order=None):
    """Which k training observations to mislabel: uniformly at random.

    Independent draw without replacement at each epsilon, or, with
    NESTED_FLIPS, the first k entries of one fixed permutation.
    """
    if NESTED_FLIPS:
        return order[:k]                        # prefix of one fixed shuffle
    return rng.choice(n, size=k, replace=False)


# =========================================================================
#                          REPLICATION LOOP
# =========================================================================

def run(X, y, eps=0.0, replications=REPLICATIONS, base_seed=0, verbose=True):
    """Returns {model: {metric: array over replications}}.

    The split for replication i is a deterministic function of base_seed + i
    and of nothing else, so the clean arm and every contaminated arm see the
    IDENTICAL train/test partition at each i. That is what makes the paired
    differences against logistic, and the increase-over-clean columns, exact
    rather than merely same-distribution -- the same property the simulation
    gets from fixing the draw order in generate().
    """
    raw = {}

    for i in range(replications):
        if verbose:
            print(i + 1, end=" " if (i + 1) % 20 else "\n", flush=True)

        seed = base_seed + i
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_FRAC,
                                     random_state=seed)
        tr, te = next(sss.split(X, y))
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        if eps > 0:
            rng = np.random.default_rng(seed)
            k = max(1, int(round(eps * len(y_tr))))
            order = rng.permutation(len(y_tr)) if NESTED_FLIPS else None
            y_tr = flip(y_tr, flip_index_real(rng, k, len(y_tr), order=order))

        ev = evaluate_real(
            predict_real(X_te, train_models_real(X_tr, y_tr, N_ROUNDS, seed)),
            y_te)

        for name, metrics in ev.items():
            for metric, value in metrics.items():
                raw.setdefault(name, {}).setdefault(metric, []).append(value)

    if verbose:
        print()
    return {m: {k: np.asarray(v) for k, v in d.items()} for m, d in raw.items()}


# =========================================================================
#                              EXPORT
# =========================================================================

LONG_FIELDS_REAL = ["dataset", "p", "prevalence", "hessian_rule", "mh_link",
                    "contamination", "share", "flip_rule",
                    "model", "metric", "replication", "value"]


def collect_real(rows, raw, dgp, kind, share, flip_rule=""):
    for name, metrics in raw.items():
        for metric, values in metrics.items():
            for i, value in enumerate(values):
                rows.append((dgp.g, dgp.p, dgp.prevalence, HESSIAN_RULE, MH_LINK,
                             kind, share, flip_rule, name, metric, i, value))
    return rows


def save_csv_real(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(LONG_FIELDS_REAL)
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {path}")


def save_summary_real(rows, path, reference="logistic", digits=ROUND_TO):
    """CORE.save_summary with `g` renamed to `dataset` and the two
    DGP-only columns (tau, sd) dropped. Everything else, including the sort
    key, is CORE's, so the workbook macro needs no change."""
    drop = {"tau", "sd", "bayes", "rho"}
    fields_in = [f for f in SUMMARY_FIELDS if f not in drop]
    fields_out = ["dataset" if f == "g" else
                  (f"diff_vs_{reference}" if f == "diff_vs_ref" else f)
                  for f in fields_in]

    rows = sorted(rows, key=summary_key)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields_out, restval="")
        w.writeheader()
        for rec in rows:
            out = {}
            for f_in, f_out in zip(fields_in, fields_out):
                v = rec.get(f_in, "")
                if digits is not None and isinstance(v, float):
                    v = round(v, digits)
                out[f_out] = v
            w.writerow(out)
    print(f"wrote {len(rows)} summary rows to {path}")


# =========================================================================
#                               MAIN
# =========================================================================

def main():
    if not METRICS:
        raise ValueError("METRICS is empty; uncomment at least one metric")

    base_seed = BASE_SEED
    if base_seed is None:
        base_seed = secrets.randbelow(1_000_000) * 1_000
    print(f"base_seed = {base_seed}   (paste into BASE_SEED to reproduce)\n")

    X, y, name, cols = load_dataset()
    prevalence = describe_dataset(X, y, name, cols)
    dgp = RealDgp(g=name, p=X.shape[1], rho="", bayes="",
                  prevalence=round(prevalence, 6), tau="", sd="")

    print(f"\n  Hessian rule {HESSIAN_RULE}   Modified Huber link {MH_LINK}   "
          f"metrics {', '.join(METRICS)}   flip rule {FLIP_RULE}")

    long_rows, sum_rows = [], []

    print("\n=== clean (nominal) ===")
    clean = run(X, y, eps=0.0, base_seed=base_seed)
    show(clean)
    collect_real(long_rows, clean, dgp, "clean", 0.0)
    sum_rows += summarise(clean, dgp, "clean", 0.0)

    for eps in EPS_GRID:
        print(f"\n=== mislabeled {eps:.0%}, rule = {FLIP_RULE} ===")
        raw = run(X, y, eps=eps, base_seed=base_seed)
        show(raw)
        for metric in METRICS:
            degradation(clean, raw, metric=metric)
        collect_real(long_rows, raw, dgp, "mislabeled", eps, FLIP_RULE)
        sum_rows += summarise(raw, dgp, "mislabeled", eps, clean=clean,
                              flip_rule=FLIP_RULE)

    if WRITE_CSV:
        print()
        save_csv_real(long_rows, f"{OUT_PREFIX}_{name}_long_mcw0.csv")
        save_summary_real(sum_rows, f"{OUT_PREFIX}_{name}_summary_mcw0_.csv")


if __name__ == "__main__":
    main()