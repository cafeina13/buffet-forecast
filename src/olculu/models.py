"""Guest model (layer 1) and per-dish quantile consumption model (layer 2)."""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from olculu.features import DISH_FEATURES, GUEST_FEATURES

GUEST_QUANTILES = [0.1, 0.5, 0.9]
DISH_QUANTILES = [0.1, 0.5, 0.8, 0.9, 0.95]

PARAMS = dict(
    n_estimators=220, learning_rate=0.05, num_leaves=15, min_child_samples=25,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.9, verbose=-1,
)


def fit_quantiles(X: pd.DataFrame, y, quantiles, seed=0, **overrides) -> dict:
    params = {**PARAMS, **overrides}
    return {
        q: lgb.LGBMRegressor(objective="quantile", alpha=q, random_state=seed, **params).fit(X, y)
        for q in quantiles
    }


def predict_quantiles(models: dict, X: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Predictions with one column per quantile, sorted per row to remove crossing.

    Returns the matrix and the number of rows where sorting changed something.
    """
    qs = sorted(models)
    raw = np.column_stack([models[q].predict(X) for q in qs])
    fixed = np.sort(raw, axis=1)
    n_crossed = int((np.abs(fixed - raw) > 1e-9).any(axis=1).sum())
    return fixed, n_crossed


GUEST_PARAMS = dict(n_estimators=200, learning_rate=0.04, num_leaves=7, min_child_samples=15)


def fit_guest_models(services: pd.DataFrame, features=GUEST_FEATURES, target="rate", seed=0):
    """Default: quantiles of the share of present guests who come to the buffet.

    target="attendance" fits guest counts directly (used for the occupancy-only ablation).
    """
    return fit_quantiles(services[features], services[target], GUEST_QUANTILES, seed=seed, **GUEST_PARAMS)


def predict_guests(models: dict, services: pd.DataFrame) -> tuple[np.ndarray, int]:
    """Guest quantiles = predicted rate quantiles x guests present (from the presence rule)."""
    rate, crossed = predict_quantiles(models, services[GUEST_FEATURES])
    return rate * services["present"].to_numpy()[:, None], crossed


def oof_guest_median(services: pd.DataFrame, folds=5, seed=0) -> np.ndarray:
    """Out-of-fold median guests, grouped by week so a fold never sees its own week."""
    weeks = services["date"].dt.isocalendar().week.to_numpy()
    out = np.zeros(len(services))
    for tr, te in GroupKFold(folds).split(services, groups=weeks):
        m = fit_quantiles(services.iloc[tr][GUEST_FEATURES], services.iloc[tr]["rate"], [0.5], seed=seed,
                          **GUEST_PARAMS)[0.5]
        out[te] = m.predict(services.iloc[te][GUEST_FEATURES]) * services.iloc[te]["present"].to_numpy()
    return out


def fit_dish_models(frame: pd.DataFrame, seed=0):
    return fit_quantiles(frame[DISH_FEATURES], frame["ratio"], DISH_QUANTILES, seed=seed)


def interpolate_quantile(qmat: np.ndarray, quantiles, level):
    """Linear interpolation between predicted quantiles for any level within their range."""
    level = float(np.clip(level, min(quantiles), max(quantiles)))
    return np.array([np.interp(level, quantiles, row) for row in qmat])


def pinball(y, pred, q):
    d = y - pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))
