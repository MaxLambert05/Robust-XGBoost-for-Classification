#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared core for the robust-loss XGBoost study.

"""

import csv
import re
from collections import namedtuple
from functools import lru_cache, partial

import numpy as np
import xgboost as xgb
from scipy.optimize import brentq
from scipy.special import expit, logit, ndtr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acc, roc_auc_score as auc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOG2 = float(np.log(2.0))
SCALE = np.sqrt(3.0) / np.pi      # sd -> logistic scale parameter

# ==========================================================================
# CONFIGURATION -- everything a run varies is in this block
# ==========================================================================

# --- 1. booster hyperparameters -------------------------------------------
# Identical for every boosted model; only the objective differs.
COMMON = dict(
    max_depth=6, eta=0.3,
    reg_lambda=1.0, min_child_weight=1.0, gamma=0.0,
    subsample=1.0, colsample_bytree=1.0,
    tree_method="exact", nthread=1,
)
PROB_SCALE = 0.5   # base_score for objectives with a logit link -> margin 0
RAW_SCALE = 0.0    # base_score for hinge and custom objectives  -> margin 0

# --- 2. which models to fit -----------------------------------------------
# Order here is the order they are fitted, printed and sorted in (see
# FAMILY_ORDER). Every flag is independent; an empty tuning tuple drops that
# family entirely. Override from a daughter script with configure_models().
FIT_LOGREG = False             # plain logistic regression, no boosting
FIT_XGB_LOGISTIC = True       # XGBoost, binary:logistic  -- the reference
FIT_HINGE = True              # XGBoost, binary:hinge
FIT_MODIFIED_HUBER = True     # XGBoost, custom modified-Huber objective

BY_EXP_TUNING = (0.5,)   # exponential rho; any c > 0
BY_ORIG_TUNING = (0.75,)          # original rho; needs c > log 2 = 0.6931

# --- 3. loss parameters ----------------------------------------------------
BY_EXP_RESCALE = True     # x exp(sqrt(c)); required for cross-loss comparability
BY_ORIG_RESCALE = False   # rho'(0) = 1 already, so no rescaling is needed
BY_FLOOR = 0.0            # Hessian floor applied AFTER summing both terms

# --- 4. the logistic regression baseline ----------------------------------
# The classical (non-robust) estimator the BY losses were proposed against.
# It reads the raw covariates, so under a non-linear signal it is misspecified
# by construction -- that is the point of carrying it: it separates "the loss
# is robust" from "the learner is flexible".
LOGREG_NAME = "LogReg"
LOGREG_STANDARDISE = True     # z-score the covariates before fitting
LOGREG_KWARGS = dict(         # C = 1e12 is the unpenalised MLE in every
    C=1e12,                   # scikit-learn version (penalty=None is
    solver="lbfgs",           # default optimization method for logreg, deprecated in 1.8); drop C to a finite value
    max_iter=2000,            # if a design ever separates and lbfgs runs away
)

# --- 5. inference ----------------------------------------------------------
REFERENCE = "logistic"    # model every paired difference is taken against

# ======================= end of configuration =============================


# ------------------------------------------------------------ shared pieces
def _deviance(y, s):
    """Logistic deviance d(y, s), computed stably."""
    return np.logaddexp(0.0, np.where(y == 1, -s, s))


# ------------------------------------------- Bianco--Yohai, exponential rho
def _G1(t, c):
    """G'(t). Closed form collapses to exp(-sqrt(-log t)) on the lower branch."""
    out = np.full_like(t, np.exp(-np.sqrt(c)))
    m = t <= np.exp(-c)
    if m.any():
        u = np.sqrt(-np.log(np.clip(t[m], 1e-300, None)))
        out[m] = np.exp(-u)
    return out


def _G2(t, c):
    """G''(t) = exp(-u) / (2 u t) with u = sqrt(-log t)."""
    out = np.zeros_like(t)
    m = t <= np.exp(-c)
    if m.any():
        tm = np.clip(t[m], 1e-300, None)
        u = np.sqrt(-np.log(tm))
        out[m] = np.exp(-u) / (2.0 * u * tm)
    return out


def BY_exp(predt, dtrain, c=0.5, rescale=True, floor=0):
    """Bianco--Yohai objective with the exponential rho: rho(d(y,s)) + C(s)."""
    y = dtrain.get_label().astype(np.float64)
    s = np.asarray(predt, dtype=np.float64)

    p = expit(s)
    t = _deviance(y, s)
    ts = np.sqrt(np.maximum(t, 1e-12))

    rho1 = np.where(t <= c, np.exp(-np.sqrt(c)), np.exp(-ts))
    rho2 = np.where(t <= c, 0.0, -np.exp(-ts) / (2.0 * ts))

    d1 = p - y                 # d/ds  of the deviance
    d2 = p * (1.0 - p)         # d2/ds2
    F1 = p * (1.0 - p)
    F2 = F1 * (1.0 - 2.0 * p)

    dG = _G1(p, c) - _G1(1.0 - p, c)

    grad = rho1 * d1 + F1 * dG
    hess = (rho2 * d1**2 + rho1 * d2
            + F1**2 * (_G2(p, c) + _G2(1.0 - p, c))
            + F2 * dG)

    if rescale:
        k = np.exp(np.sqrt(c))
        grad, hess = k * grad, k * hess

    return grad, np.maximum(hess, floor)   # clip AFTER summing both terms


# ---------------------------------------------- Bianco--Yohai, original rho
def _G1o(t, c):
    """G'(t) = (1 + log(t)/c)_+ ; identically 0 for t <= exp(-c)."""
    out = np.zeros_like(t)
    m = t > np.exp(-c)
    if m.any():
        out[m] = 1.0 + np.log(t[m]) / c
    return out


def _G2o(t, c):
    """G''(t) = 1/(c t) on t > exp(-c), else 0. Bounded above by exp(c)/c."""
    out = np.zeros_like(t)
    m = t > np.exp(-c)
    if m.any():
        out[m] = 1.0 / (c * t[m])
    return out


def _Go(t, c):
    """G(t) itself -- only needed for reporting the objective value."""
    out = np.zeros_like(t)
    m = t > np.exp(-c)
    if m.any():
        tm = t[m]
        out[m] = tm + (tm * np.log(tm) - tm) / c + np.exp(-c) / c
    return out


def _check_c(c):
    """rho'(log 2) = 1 - log(2)/c is the weight at margin 0.

    At c = log 2 it is exactly zero: every gradient and Hessian vanishes at the
    starting margin and the model never leaves its base score.
    """
    if c <= LOG2:
        raise ValueError(f"BY_orig needs c > log 2 = {LOG2:.4f}, got c={c}")


def BY_orig_loss(y, s, c):
    """phi_BY(s; y), normalised so phi -> 0 as the margin becomes correct."""
    _check_c(c)
    t = _deviance(y, s)
    p = expit(s)
    rho = np.where(t <= c, t - t**2 / (2.0 * c), c / 2.0)
    G1val = 1.0 - 1.0 / c + np.exp(-c) / c            # G(1)
    return rho + _Go(p, c) + _Go(1.0 - p, c) - G1val


def BY_orig(predt, dtrain, c=1.0, rescale=False, floor=0):
    """Original Bianco--Yohai (1996) objective: rho(d(y,s)) + C(s), with

        rho(t) = t - t^2/(2c)  for t <= c,   c/2  otherwise.
    """
    _check_c(c)
    y = dtrain.get_label().astype(np.float64)
    s = np.asarray(predt, dtype=np.float64)

    p = expit(s)
    t = _deviance(y, s)

    rho1 = np.maximum(0.0, 1.0 - t / c)              # rho'
    rho2 = np.where(t < c, -1.0 / c, 0.0)            # rho''

    d1 = p - y
    d2 = p * (1.0 - p)
    F1 = p * (1.0 - p)
    F2 = F1 * (1.0 - 2.0 * p)

    dG = _G1o(p, c) - _G1o(1.0 - p, c)

    grad = rho1 * d1 + F1 * dG
    hess = (rho2 * d1**2 + rho1 * d2
            + F1**2 * (_G2o(p, c) + _G2o(1.0 - p, c))
            + F2 * dG)

    if rescale:
        k = c / (c - LOG2)
        grad, hess = k * grad, k * hess

    return grad, np.maximum(hess, floor)


# ---------------------------------------------------------- modified Huber
def modified_huber(predt, dtrain):
    y01 = dtrain.get_label()
    y = 2.0 * y01 - 1.0
    z = y * predt

    grad = np.zeros_like(predt)
    hess = np.zeros_like(predt)

    mid = (z > -1) & (z < 1)
    left = z <= -1

    grad[mid] = -2.0 * y[mid] * (1.0 - z[mid])
    hess[mid] = 2.0
    grad[left] = -4.0 * y[left]
    hess[left] = 0.0

    return grad, hess


# ------------------------------------------------------------ signal functions
def _g0(X):
    return X @ np.full(X.shape[1], 3.0)


def _g1(X):
    if X.shape[1] < 5:
        raise ValueError("g=1 needs at least 5 predictors")
    return (2 * X[:, 0] - 2 * X[:, 1] + 8 * (X[:, 2] - 0.5) ** 2
            + np.exp(X[:, 3]) + 0.5 * np.cos(8 * np.pi * X[:, 4]) * np.exp(2 * X[:, 4]))


def _g2(X):
    if X.shape[1] < 3:
        raise ValueError("g=2 needs at least 3 predictors")
    p = X.shape[1]
    sign = (-1.0) ** np.arange(1, p + 1)
    y1 = (1 + sign * 0.8 * X + np.sin(6 * X)).sum(axis=1)
    y2 = (1 + X[:, :3] / 3).sum(axis=1)
    return y1 * y2


SIGNAL = {0: _g0, 1: _g1, 2: _g2}
COVARIATES = {0: "normal", 1: "uniform", 2: "uniform"}   # g1, g2 live on [0,1]^p

# How many covariates each signal actually reads. None means "all p", which is
# only correct for g0, whose linear form is defined for any width. g1 and g2
# are five-variable functions: with ACTIVE set, raising p adds pure noise
# columns and leaves the DGP itself untouched, so p is a nuisance-dimension
# axis rather than a redefinition of the signal.
ACTIVE = {0: None, 1: 5, 2: 5}


def signal(X, g):
    """Evaluate signal g on the columns it is defined over.

    The active columns are the FIRST ACTIVE[g] of X, so the relevant
    covariates are X[:, :ACTIVE[g]] and everything beyond is noise. At
    p == ACTIVE[g] the slice is the whole matrix and this is a no-op, which
    is what keeps p = 5 results bit-identical to the pre-ACTIVE code.
    """
    d = ACTIVE.get(g)
    return SIGNAL[g](X if d is None else X[:, :d])


def register_signal(g, fn, covariates="uniform", active=None):
    """Add or replace a signal function without editing this module.

    fn receives an (n, active) matrix and returns an (n,) array. Register
    before the first make_dgp/calibrate call for that g; the caches are
    cleared here anyway, so re-registering mid-process is safe as long as no
    already-fitted results depend on the old definition.
    """
    if covariates not in ("uniform", "normal"):
        raise ValueError(f"covariates={covariates!r}; expected 'uniform' or 'normal'")
    if active is not None and int(active) < 1:
        raise ValueError("active must be a positive number of columns")
    SIGNAL[g] = fn
    COVARIATES[g] = covariates
    ACTIVE[g] = None if active is None else int(active)
    _reference_signal.cache_clear()
    calibrate.cache_clear()
    make_dgp.cache_clear()
    return g


# NOT USED------------------------------------------------------- correlated covariates
@lru_cache(maxsize=None)
def _cholesky(p, rho):
    lo = -1.0 / (p - 1)
    if not (lo < rho < 1.0):
        raise ValueError(f"rho={rho} outside ({lo:.4f}, 1) for p={p}")
    R = np.full((p, p), rho)
    np.fill_diagonal(R, 1.0)
    return np.linalg.cholesky(R)


def draw_X(rng, n, p, g, rho=0.0):
    if rho == 0.0:
        if COVARIATES[g] == "normal":
            return rng.standard_normal((n, p))
        return rng.random((n, p))

    L = _cholesky(p, float(rho))
    Z = rng.standard_normal((n, p)) @ L.T
    if COVARIATES[g] == "normal":
        return Z
    return ndtr(Z)


def pearson_from_latent(rho):
    return (6.0 / np.pi) * np.arcsin(rho / 2.0)


# --------------------------------------------------------------- calibration
@lru_cache(maxsize=None)
def _reference_signal(g, p, rho=0.0, m=200_000, seed=987654321):
    """Large cached draw of signal(X), used to calibrate tau and sd.

    Fixed seed, so this is deterministic quadrature rather than data: it is
    drawn once per (g, p, rho) for the whole process, never fitted on, and
    never contaminated. Changing m, seed, or the draw order invalidates any
    hard-coded constants in CALIBRATED.

    Only the active columns are drawn. Under independence that is trivially
    the right marginal law, and under the Gaussian copula the equicorrelation
    matrix is consistent under marginalisation, so the first d of p columns
    have exactly the law of a d-column draw at the same rho. Two consequences:
    tau and sd no longer depend on p (a p = 100 design has the same DGP as the
    p = 5 one plus noise columns), and the draw costs m*d rather than m*p.
    At p == ACTIVE[g] this is the original draw, unchanged.
    """
    d = ACTIVE.get(g) or p
    rng = np.random.default_rng(seed)
    return SIGNAL[g](draw_X(rng, m, d, g, rho))


def _bayes_error(sig, tau, sd):
    p_star = expit((sig - tau) / (sd * SCALE))
    return float(np.mean(np.minimum(p_star, 1.0 - p_star)))


# Frozen calibration constants. Key: (g, p, rho, bayes, prevalence).
# A miss falls through to brentq rather than silently using the wrong values.
CALIBRATED = {}


@lru_cache(maxsize=None)
def calibrate(g, p=5, bayes=0.10, prevalence=0.5, rho=0.0, check=True):
    """tau gives P(y=1) = prevalence; sd gives the target Bayes error.

    Bayes error is continuous and strictly increasing in sd (sd -> 0 pushes
    every p* to 0 or 1, sd -> inf pushes every p* to 1/2), so the root is
    unique and brentq is safe on a wide bracket.
    """
    key = (g, p, float(rho), float(bayes), float(prevalence))
    sig = _reference_signal(g, p, rho)

    if key in CALIBRATED:
        tau, sd = CALIBRATED[key]
        if check:
            hit = _bayes_error(sig, tau, sd)
            if abs(hit - bayes) > 1e-4:
                raise ValueError(
                    f"frozen constants for {key} give Bayes error {hit:.6f}, "
                    f"target {bayes}; the reference draw must have changed")
        return tau, sd

    tau = float(np.quantile(sig, 1.0 - prevalence))
    sd = float(brentq(lambda s: _bayes_error(sig, tau, s) - bayes, 1e-6, 1e6))
    return tau, sd


Dgp = namedtuple("Dgp", "g p rho bayes prevalence tau sd")


@lru_cache(maxsize=None)
def make_dgp(g, p=5, bayes=0.10, prevalence=0.5, rho=0.0):
    if g not in SIGNAL:
        raise ValueError(f"unknown g={g}; expected one of {sorted(SIGNAL)}")
    d = ACTIVE.get(g)
    if d is not None and p < d:
        raise ValueError(f"g={g} reads {d} covariates; p={p} is too narrow")
    tau, sd = calibrate(g, p, bayes, prevalence, rho)
    return Dgp(g=g, p=p, rho=float(rho), bayes=float(bayes),
               prevalence=float(prevalence), tau=tau, sd=sd)


def describe(dgp, verbose=True):
    sig = _reference_signal(dgp.g, dgp.p, dgp.rho)
    p_star = expit((sig - dgp.tau) / (dgp.sd * SCALE))
    be = float(np.mean(np.minimum(p_star, 1 - p_star)))
    d = ACTIVE.get(dgp.g) or dgp.p
    if verbose:
        print(f"DGP g={dgp.g}  p={dgp.p}  active={d} (noise={dgp.p - d})  "
              f"covariates={COVARIATES[dgp.g]}  "
              f"rho={dgp.rho:g}"
              + (f" (Pearson {pearson_from_latent(dgp.rho):.4f})"
                 if dgp.rho else ""))
        print(f"  tau = {dgp.tau:.6f}   sd = {dgp.sd:.6f}")
        print(f"  P(y=1) = {p_star.mean():.4f}   Bayes error = {be:.4f} "
              f"(target {dgp.bayes})")
    return be


def freeze_line(dgp):
    """Printable CALIBRATED entry for this DGP, to paste in once and forget."""
    return (f"    ({dgp.g}, {dgp.p}, {dgp.rho!r}, {dgp.bayes!r}, "
            f"{dgp.prevalence!r}): ({dgp.tau!r}, {dgp.sd!r}),")


# ------------------------------------------------------------ data generation
def draw_eps(rng, n, dgp):
    """Logistic errors with standard deviation dgp.sd."""
    return rng.logistic(0.0, dgp.sd * SCALE, size=n)


def eta(X, dgp):
    """True Bayes logit, (g(x) - tau) / s. logit(p*) by construction."""
    return (signal(X, dgp.g) - dgp.tau) / (dgp.sd * SCALE)


def true_p(X, dgp):
    return expit(eta(X, dgp))


def labels(X, eps, dgp):
    return 1 * ((signal(X, dgp.g) + eps) > dgp.tau)


def contaminate(X, idx_rows, idx_cols, lvl, typ):
    """Displace selected cells by lvl * sd_j, away from the column centre."""
    X = X.copy()
    scale = X.std(axis=0)
    centre = X.mean(axis=0)          # 0 for normals, 0.5 for uniforms
    if typ == "all":
        sub = X[idx_rows]
        X[idx_rows] = sub + np.where(sub > centre, 1.0, -1.0) * np.abs(lvl * scale)
    elif typ == "random":
        cur = X[idx_rows, idx_cols]
        X[idx_rows, idx_cols] = (cur + np.where(cur > centre[idx_cols], 1.0, -1.0)
                                 * np.abs(lvl * scale[idx_cols]))
    else:
        raise ValueError(f"unknown typ={typ!r}; expected 'all' or 'random'")
    return X


def flip(y, idx):
    y = y.copy()
    y[idx] = 1 - y[idx]
    return y


FLIP_RULES = ("uniform", "margin", "boundary")


def bayes_margin(X, y, dgp):
    """Signed Bayes margin of the REALISED label: m_i = (2 y_i - 1) * eta_i.

    Large positive m means the label agrees with the Bayes rule and does so
    confidently. Small |m| means the point sits on the decision boundary and
    its label was close to a coin flip. Negative m means the draw already went
    against the odds -- such a point is an outlier before any contamination,
    and flipping it would REPAIR it rather than corrupt it.
    """
    return np.where(y == 1, 1.0, -1.0) * eta(X, dgp)


def flip_index(rng, X, y, dgp, k, rule):
    """Which k observations to mislabel.

    uniform   the original design: an index is chosen without regard to how
              informative it is. Roughly 2 * BayesError of the flips land on
              near-ambiguous points and do almost nothing, so the nominal
              contamination share overstates the effective one.
    margin    the k largest Bayes margins: confidently correct points, whose
              flip is maximally inconsistent with the rest of the sample. This
              is the margin-space analogue of bad leverage, and unlike x-space
              displacement a tree cannot quarantine it without also carving
              out the well-fitted neighbours it sits among.
    boundary  the k smallest |margin|: a control. These labels were nearly
              coin flips already, so a bounded rho should gain almost nothing
              over logistic here. If it does, the effect is not robustness.

    margin and boundary are deterministic given the data and consume no RNG,
    which is intentional: nothing downstream draws again, so the clean data
    for a given base_seed are identical across all three rules and the
    comparison is exactly paired.
    """
    if rule == "uniform":
        return rng.choice(len(y), size=k, replace=False)

    m = bayes_margin(X, y, dgp)
    if rule == "margin":
        return np.argsort(-m, kind="stable")[:k]
    if rule == "boundary":
        return np.argsort(np.abs(m), kind="stable")[:k]
    raise ValueError(f"unknown flip_rule={rule!r}; expected one of {FLIP_RULES}")


def generate(dgp, obs, mislabeled=0.0, lp=0.0, lp_lvl=10, lp_typ="all",
             goodbad=0, seed=None, test_obs=2000, flip_rule="uniform"):
    """One replication. Contamination hits the training set only.

    Draw order (X_train, eps_train, X_test, eps_test) is fixed and precedes
    every contamination decision, so the underlying clean data are identical
    across contamination types, shares and flip rules at a given seed. That is
    what makes degradation() and the `increase` column paired differences.
    """
    if lp > 0 and mislabeled > 0:
        raise ValueError("use either mislabeling or leverage contamination, not both")

    rng = np.random.default_rng(seed)

    X_train = draw_X(rng, obs, dgp.p, dgp.g, dgp.rho)
    eps_train = draw_eps(rng, obs, dgp)
    X_test = draw_X(rng, test_obs, dgp.p, dgp.g, dgp.rho)
    eps_test = draw_eps(rng, test_obs, dgp)

    y_test = labels(X_test, eps_test, dgp)

    if lp > 0:
        k = max(1, int(round(lp * obs)))
        idx_rows = rng.choice(obs, size=k, replace=False)
        idx_cols = rng.choice(dgp.p, size=k, replace=True)
        X_train = contaminate(X_train, idx_rows, idx_cols, lp_lvl, lp_typ)
        y_train = labels(X_train, eps_train, dgp)    # labels AFTER displacement
        if goodbad:
            y_train = flip(y_train, idx_rows)        # bad leverage
    else:
        y_train = labels(X_train, eps_train, dgp)
        if mislabeled > 0:
            k = max(1, int(round(mislabeled * obs)))
            y_train = flip(y_train, flip_index(rng, X_train, y_train, dgp, k,
                                               flip_rule))

    p_test = true_p(X_test, dgp)

    return X_train, X_test, y_train, y_test, p_test


# ------------------------------------------------------------ model registry
# A ModelSpec is everything train_models and predict need to know about one
# model, so neither of them has to special-case a name:
#
#   kind    "xgb" -> fitted with xgb.train on a DMatrix
#           "sk"  -> any scikit-learn estimator, fitted on the raw X
#   link    "prob"   -> the model outputs probabilities; margins are recovered
#                       as logit(p)
#           "margin" -> the model outputs margins; probabilities are expit(s)
#   params  extra booster parameters, merged over COMMON (kind="xgb")
#   obj     custom objective, or None for a built-in one (kind="xgb")
#   build   callable(seed) -> unfitted estimator (kind="sk")
ModelSpec = namedtuple("ModelSpec", "name kind link params obj build",
                       defaults=(None, None, None))

MODEL_CONFIG_KEYS = (
    "FIT_LOGREG", "FIT_XGB_LOGISTIC", "FIT_HINGE", "FIT_MODIFIED_HUBER",
    "BY_EXP_TUNING", "BY_ORIG_TUNING",
    "BY_EXP_RESCALE", "BY_ORIG_RESCALE", "BY_FLOOR",
    "LOGREG_NAME", "LOGREG_STANDARDISE", "LOGREG_KWARGS",
    "COMMON", "PROB_SCALE", "RAW_SCALE", "REFERENCE",
)


def configure_models(**kwargs):
    """Override the model-set configuration from a daughter script.

    Only the names in MODEL_CONFIG_KEYS are accepted, so a typo raises here
    rather than silently doing nothing. model_specs() reads these globals at
    call time, so a change takes effect for every later fit without any object
    having to be rebuilt.
    """
    unknown = [k for k in kwargs if k not in MODEL_CONFIG_KEYS]
    if unknown:
        raise ValueError(f"unknown model settings {unknown}; expected a subset "
                         f"of {list(MODEL_CONFIG_KEYS)}")
    for c in kwargs.get("BY_ORIG_TUNING", BY_ORIG_TUNING):
        _check_c(c)                   # fail now, not 3000 fits from now, and
    globals().update(kwargs)          # before anything here has been changed
    return model_names()


def make_logreg(seed=0):
    """The unfitted logistic regression baseline.

    Standardising is cosmetic for the unpenalised fit -- the MLE is
    equivariant under affine rescaling of the columns -- but it keeps lbfgs
    well conditioned, and it is what makes the penalised variant meaningful if
    LOGREG_KWARGS is ever switched to one.
    """
    clf = LogisticRegression(**LOGREG_KWARGS)
    return make_pipeline(StandardScaler(), clf) if LOGREG_STANDARDISE else clf


def model_specs():
    """The models to fit, in order, built from the CONFIGURATION block."""
    specs = []

    if FIT_LOGREG:
        specs.append(ModelSpec(LOGREG_NAME, "sk", "prob", build=make_logreg))
    if FIT_XGB_LOGISTIC:
        specs.append(ModelSpec("logistic", "xgb", "prob",
                               params={"objective": "binary:logistic",
                                       "base_score": PROB_SCALE}))
    if FIT_HINGE:
        specs.append(ModelSpec("hinge", "xgb", "margin",
                               params={"objective": "binary:hinge",
                                       "base_score": RAW_SCALE}))
    if FIT_MODIFIED_HUBER:
        specs.append(ModelSpec("Modified_Huber", "xgb", "margin",
                               params={"base_score": RAW_SCALE},
                               obj=modified_huber))

    for c in BY_EXP_TUNING:
        specs.append(ModelSpec(f"BY_exp_{c}", "xgb", "margin",
                               params={"base_score": RAW_SCALE},
                               obj=partial(BY_exp, c=c,
                                           rescale=BY_EXP_RESCALE,
                                           floor=BY_FLOOR)))
    for c in BY_ORIG_TUNING:
        specs.append(ModelSpec(f"BY_orig_{c}", "xgb", "margin",
                               params={"base_score": RAW_SCALE},
                               obj=partial(BY_orig, c=c,
                                           rescale=BY_ORIG_RESCALE,
                                           floor=BY_FLOOR)))
    return specs


def model_names():
    return [s.name for s in model_specs()]


def describe_models(verbose=True):
    """One line per model; what a run is about to fit."""
    specs = model_specs()
    if verbose:
        print(f"models ({len(specs)}), reference = {REFERENCE}:")
        for s in specs:
            how = ("scikit-learn" if s.kind == "sk"
                   else s.params.get("objective", "custom objective"))
            print(f"  {s.name:18s} {s.kind:4s} {s.link:7s} {how}")
    return specs


# ------------------------------------------------------------------ modelling
def train_models(X_tr, y_tr, n=100, seed=0):
    """Fit every model in model_specs() on the same training sample."""
    train = xgb.DMatrix(X_tr, label=y_tr)
    p = {**COMMON, "seed": seed}
    models = {}

    for spec in model_specs():
        if spec.kind == "sk":
            models[spec.name] = spec.build(seed).fit(X_tr, y_tr)
        elif spec.kind == "xgb":
            models[spec.name] = xgb.train({**p, **spec.params}, train, n,
                                          obj=spec.obj)
        else:
            raise ValueError(f"unknown kind={spec.kind!r} for {spec.name}")

    return models


def predict(X_te, model_dictionary):
    """Margins, probabilities and labels, dispatched on each model's spec.

    Float64 at source: XGBoost returns float32, and clipping a float32 array
    at 1 - 1e-12 is a silent no-op that turns log1p(-p) into -inf downstream.
    """
    test = xgb.DMatrix(X_te)
    spec = {s.name: s for s in model_specs()}
    scores, probabilities, predictions = {}, {}, {}

    for name, model in model_dictionary.items():
        kind, link = spec[name].kind, spec[name].link

        if kind == "sk":
            prob = np.asarray(model.predict_proba(X_te)[:, 1], dtype=np.float64)
        elif link == "prob":
            prob = np.asarray(model.predict(test), dtype=np.float64)
        else:
            # hinge needs output_margin; custom objectives already return margins
            s = np.asarray(model.predict(test, output_margin=True),
                           dtype=np.float64)

        if link == "prob":
            s = logit(np.clip(prob, 1e-12, 1 - 1e-12))
        else:
            prob = expit(s)

        scores[name] = s
        probabilities[name] = prob
        predictions[name] = 1 * (s > 0)

    return scores, probabilities, predictions


def evaluation(pred_tuple, y_te, p_star, extras=True):
    """Primary metrics measure the fitted probability, not the classification.

    MSE_true is the squared error against the Bayes probability p*, i.e. the
    functional analogue of MSE(beta_hat, beta) in robust regression. It equals
    the Brier score minus the irreducible Bernoulli term E[p*(1-p*)], which is
    identical across models, so it ranks models the same way with less noise.
    """
    scores, probabilities, predictions = pred_tuple
    out = {}
    for name in predictions:
        p = np.clip(probabilities[name], 1e-12, 1 - 1e-12)
        m = {
            "MSE_true": float(np.mean((p - p_star) ** 2)),
            #"Brier":    float(np.mean((p - y_te) ** 2)),
            #"LogLoss":  float(np.mean(-(y_te * np.log(p) + (1 - y_te) * np.log1p(-p)))),
        }
        if extras:                       # reported for comparability only
            m["Accuracy"] = float(acc(y_te, predictions[name]))
            #m["AUC"] = float(auc(y_te, scores[name]))
        out[name] = m
    return out


# --------------------------------------------------------------- Monte Carlo
def compare_generate(iterations=100, observations=500, predictors=5, n=100,
                     mislabeled=0.0, lp=0.0, lp_lvl=10, lp_typ="all", g=0,
                     bad_leverage=0, base_seed=0, test_obs=2000, verbose=True,
                     extras=True, bayes=0.10, prevalence=0.5, rho=0.0,
                     flip_rule="uniform"):
    """Returns {model: {metric: array of per-iteration values}}."""
    dgp = make_dgp(g, predictors, bayes, prevalence, rho)
    raw = {}
    for i in range(iterations):
        if verbose:
            print(i + 1, end=" " if (i + 1) % 20 else "\n", flush=True)

        X_tr, X_te, y_tr, y_te, p_star = generate(
            dgp, observations, mislabeled, lp, lp_lvl, lp_typ, bad_leverage,
            seed=base_seed + i, test_obs=test_obs, flip_rule=flip_rule)

        ev = evaluation(predict(X_te, train_models(X_tr, y_tr, n, seed=base_seed + i)),
                        y_te, p_star, extras=extras)

        for name, metrics in ev.items():
            for metric, value in metrics.items():
                raw.setdefault(name, {}).setdefault(metric, []).append(value)

    return {m: {k: np.asarray(v) for k, v in d.items()} for m, d in raw.items()}


LOWER_IS_BETTER = {"MSE_true", "Brier", "LogLoss"}


def show(raw, reference=None):
    """Mean +/- standard error, plus paired difference against the reference."""
    reference = REFERENCE if reference is None else reference
    models = list(raw)
    metrics = list(raw[models[0]])
    for metric in metrics:
        direction = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
        print(f"\n{metric}  ({direction})")
        print(f"  {'model':18s} {'mean':>9s} {'se':>8s} {'vs ' + reference:>18s}")
        for name in models:
            v = raw[name][metric]
            se = v.std(ddof=1) / np.sqrt(len(v))
            if name == reference:
                tail = ""
            else:
                dif = v - raw[reference][metric]
                d_se = dif.std(ddof=1) / np.sqrt(len(dif))
                if d_se == 0 and dif.mean() == 0:
                    tail = "identical"
                else:
                    star = "*" if abs(dif.mean()) > 2 * d_se else " "
                    tail = f"{dif.mean():+8.4f} ({d_se:.4f}){star}"
            print(f"  {name:18s} {v.mean():9.4f} {se:8.4f} {tail:>18s}")


def degradation(clean, contaminated, metric="MSE_true"):
    """Paired increase in `metric` caused by contamination, per model."""
    print(f"\n{metric}: degradation from clean")
    print(f"  {'model':22s} {'clean':>9s} {'contam':>9s} {'increase':>10s} {'se':>8s}")
    for name in clean:
        a, b = clean[name][metric], contaminated[name][metric]
        dif = b - a
        se = dif.std(ddof=1) / np.sqrt(len(dif))
        print(f"  {name:22s} {a.mean():9.4f} {b.mean():9.4f} "
              f"{dif.mean():+10.4f} {se:8.4f}")


# ------------------------------------------------------------------- export
LONG_FIELDS = ["g", "p", "rho", "bayes", "prevalence", "contamination", "share",
               "flip_rule", "model", "metric", "replication", "value"]


def collect(rows, raw, dgp, kind, share, flip_rule=""):
    """Append one run to a long-format list of records for export."""
    for name, metrics in raw.items():
        for metric, values in metrics.items():
            for i, value in enumerate(values):
                rows.append((dgp.g, dgp.p, dgp.rho, dgp.bayes, dgp.prevalence,
                             kind, share, flip_rule, name, metric, i, value))
    return rows


def save_csv(rows, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(LONG_FIELDS)
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows to {path}")


# ----------------------------------------------------------- summary export
METRIC_ORDER = ["MSE_true", "Brier", "LogLoss", "Accuracy", "AUC"]
CONTAM_ORDER = ["clean", "mislabeled", "good leverage", "bad leverage"]
FLIP_ORDER = ["", "uniform", "margin", "boundary"]
# Sort order of the model column. Unlisted names sort last, so a model
# added from a daughter script still appears; add it here to place it.
FAMILY_ORDER = {LOGREG_NAME: 0, "logistic": 1, "hinge": 2, "Modified_Huber": 3,
                "BY_exp": 4, "BY_orig": 5}

SUMMARY_FIELDS = ["metric", "bayes", "rho", "contamination", "flip_rule", "share",
                  "model", "g", "p", "prevalence", "tau", "sd",
                  "mean", "se", "diff_vs_ref", "diff_se", "significant",
                  "clean_mean", "increase", "increase_se"]

ROUND_TO = 6      # digits kept in the summary file; None writes full precision


def _rank(value, order):
    return order.index(value) if value in order else len(order)


def _model_key(name):
    """Sort as train_models builds them; BY variants by numeric c, not by string."""
    m = re.match(r"^(BY_exp|BY_orig)_([0-9.]+)$", name)
    if m:
        return (FAMILY_ORDER[m.group(1)], float(m.group(2)), name)
    return (FAMILY_ORDER.get(name, 99), 0.0, name)


def summary_key(rec):
    """metric > Bayes error > rho > contamination > flip rule > level > loss."""
    return (_rank(rec["metric"], METRIC_ORDER),
            rec["bayes"],
            rec["rho"],
            _rank(rec["contamination"], CONTAM_ORDER),
            _rank(rec.get("flip_rule", ""), FLIP_ORDER),
            rec["share"],
            _model_key(rec["model"]))


def summarise(raw, dgp, kind, share, reference=None, clean=None,
              flip_rule=""):
    """One record per (model, metric): everything show and degradation print."""
    reference = REFERENCE if reference is None else reference
    rows = []
    for name, metrics in raw.items():
        for metric, v in metrics.items():
            n = len(v)
            rec = dict(g=dgp.g, p=dgp.p, rho=dgp.rho, bayes=dgp.bayes,
                       prevalence=dgp.prevalence, tau=dgp.tau, sd=dgp.sd,
                       contamination=kind, share=share, flip_rule=flip_rule,
                       model=name, metric=metric,
                       mean=float(v.mean()),
                       se=float(v.std(ddof=1) / np.sqrt(n)))

            if name != reference and reference in raw:
                d = v - raw[reference][metric]
                d_se = float(d.std(ddof=1) / np.sqrt(n))
                rec["diff_vs_ref"] = float(d.mean())
                rec["diff_se"] = d_se
                rec["significant"] = int(abs(d.mean()) > 2 * d_se) if d_se > 0 else ""

            if clean is not None and metric in clean.get(name, {}):
                a = clean[name][metric]
                if len(a) != n:
                    raise ValueError(f"clean/contaminated length mismatch for {name}")
                inc = v - a
                rec["clean_mean"] = float(a.mean())
                rec["increase"] = float(inc.mean())
                rec["increase_se"] = float(inc.std(ddof=1) / np.sqrt(n))

            rows.append(rec)
    return rows


def save_summary(rows, path, reference=None, digits=ROUND_TO):
    """Write the summary sorted metric > bayes > rho > contamination > rule > level."""
    reference = REFERENCE if reference is None else reference
    rows = sorted(rows, key=summary_key)
    fields = [f"diff_vs_{reference}" if f == "diff_vs_ref" else f
              for f in SUMMARY_FIELDS]

    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        for rec in rows:
            out = {}
            for f in SUMMARY_FIELDS:
                v = rec.get(f, "")
                if digits is not None and isinstance(v, float):
                    v = round(v, digits)
                out[f"diff_vs_{reference}" if f == "diff_vs_ref" else f] = v
            w.writerow(out)
    print(f"wrote {len(rows)} summary rows to {path}")