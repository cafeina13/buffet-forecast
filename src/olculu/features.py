"""Feature tables for the guest model and the consumption model.

True demand never enters any feature; see tests/test_leakage.py.
"""

import numpy as np
import pandas as pd

from olculu.config import MEALS, NATIONALITIES
from olculu import embeddings

MEAL_CODE = {m: i for i, m in enumerate(MEALS)}
EVENTS = ["match", "conference", "themed"]
NAT_COLS = [f"nat_{n}" for n in NATIONALITIES]

# Layer 1 has two parts:
#   1. A rule, not learned: guests physically in the hotel at a meal, from front-office counts.
#      Breakfast: in house - very early departures
#      Lunch:     in house - departures before lunch - guests on excursions
#      Dinner:    in house - late-night arrivals - a la carte bookings
#   2. A model of the share of those guests who come to the buffet (GUEST_FEATURES).
PRESENCE_COLS = ["in_house", "arrivals_night", "departures_early", "departures_very_early",
                 "excursions", "alacarte"]
GUEST_FEATURES = [
    "meal_code", "weekday", "occupancy", "child_share", *NAT_COLS,
    *[f"event_{e}" for e in EVENTS], "temperature",
]
GUEST_FEATURES_OCC_ONLY = ["meal_code", "in_house", "occupancy"]


def present_guests(meal, in_house, arrivals_night, departures_early, departures_very_early,
                   excursions, alacarte):
    """Guests in the hotel at the meal (works on scalars or arrays; meal may be an array)."""
    meal = np.asarray(meal)
    out = np.where(meal == "breakfast", in_house - departures_very_early,
                   np.where(meal == "lunch", in_house - departures_early - excursions,
                            in_house - arrivals_night - alacarte))
    return np.maximum(out, 1.0)

EMB_DIMS = 8
DISH_FEATURES = [
    "meal_code", "group_code", "weekday", "temperature", "child_share", "first_meal_share",
    *NAT_COLS, *[f"event_{e}" for e in EVENTS], "n_similar",
    "hist_per100", "knn_per100", *[f"emb_{i}" for i in range(EMB_DIMS)],
]

# Columns that only exist for evaluation and must never be features.
FORBIDDEN = {"true_demand_kg"}


def prepare_services(services: pd.DataFrame) -> pd.DataFrame:
    s = services.copy()
    s["date"] = pd.to_datetime(s["date"])
    s["meal_code"] = s["meal"].map(MEAL_CODE)
    for e in EVENTS:
        s[f"event_{e}"] = (s["event"] == e).astype(int)
    s["present"] = present_guests(s["meal"], *(s[c].to_numpy() for c in PRESENCE_COLS))
    s["rate"] = s["attendance"] / s["present"]
    return s


def censoring_corrected(servings: pd.DataFrame, hourly: pd.DataFrame) -> pd.Series:
    """Consumption estimate that treats sold-out services as lower bounds.

    For a service that ran out at time t, observed consumption covers only the guests
    who arrived before t. We scale it up by 1 / F(t), where F is the cumulative share
    of the service's guests that had arrived by t (from the hourly flow). The factor is
    capped at 2 so a very early stock-out cannot produce an extreme estimate.
    """
    h = hourly.copy()
    h["date"] = pd.to_datetime(h["date"])
    h = h.sort_values(["date", "meal", "slot"])
    h["cum"] = h.groupby(["date", "meal"])["share"].cumsum()

    out = servings["observed_kg"].astype(float).copy()
    sold = servings["stockout_time"].notna()
    if sold.any():
        rows = servings.loc[sold, ["date", "meal", "stockout_time"]].copy()
        rows["date"] = pd.to_datetime(rows["date"])
        # Cumulative share of guests arrived before the stock-out slot ends.
        merged = rows.reset_index().merge(h, on=["date", "meal"])
        merged = merged[merged["slot"] < merged["stockout_time"]]
        frac = merged.groupby("index")["cum"].max().reindex(rows.index).fillna(0.05)
        factor = np.minimum(1 / frac.clip(lower=0.05), 2.0)
        out.loc[sold] = servings.loc[sold, "observed_kg"] * factor
    return out


def dish_static(catalogue: dict, backend: str = "tfidf") -> tuple[pd.DataFrame, np.ndarray]:
    dishes = catalogue["dishes"]
    vecs = embeddings.embed(dishes, backend)
    red = embeddings.reduce(vecs, EMB_DIMS)
    groups = list(catalogue["groups"].keys())
    df = pd.DataFrame({
        "dish_id": [d["id"] for d in dishes],
        "group_code": [groups.index(d["group"]) for d in dishes],
    })
    for i in range(red.shape[1]):
        df[f"emb_{i}"] = red[:, i]
    for i in range(red.shape[1], EMB_DIMS):
        df[f"emb_{i}"] = 0.0
    return df, vecs


def history_tables(servings: pd.DataFrame, services: pd.DataFrame, target_col: str,
                   dish_ids: list[str], neighbours: np.ndarray) -> pd.DataFrame:
    """Own and nearest-dish historical consumption per 100 guests, using only earlier days."""
    att = services.set_index(["date", "meal"])["attendance"]
    v = servings.copy()
    v["per100"] = v[target_col] / att.reindex(pd.MultiIndex.from_frame(v[["date", "meal"]])).to_numpy() * 100

    all_dates = pd.DatetimeIndex(sorted(services["date"].unique()))
    rows = []
    for meal in MEALS:
        vm = v[v["meal"] == meal]
        piv = vm.pivot_table(index="date", columns="dish_id", values="per100", aggfunc="mean")
        piv = piv.reindex(index=all_dates, columns=dish_ids)
        # Expanding mean of each dish's past servings, then shift one day: only strictly earlier data.
        hist = piv.expanding().mean().ffill().shift(1)
        hist_np = hist.to_numpy()
        knn = np.nanmean(hist_np[:, neighbours], axis=2) if hist_np.size else hist_np
        for i, dish in enumerate(dish_ids):
            rows.append(pd.DataFrame({
                "date": all_dates, "meal": meal, "dish_id": dish,
                "hist_per100": hist_np[:, i], "knn_per100": knn[:, i],
            }))
    return pd.concat(rows, ignore_index=True)


def build_dish_frame(servings, services, hourly, catalogue, static, neighbours, censoring=True):
    """One row per served dish with features and the training target (kg per 100 guests)."""
    v = servings.copy()
    v["date"] = pd.to_datetime(v["date"])
    s = prepare_services(services)
    target_col = "target_kg"
    v[target_col] = censoring_corrected(v, hourly) if censoring else v["observed_kg"].astype(float)

    dish_ids = [d["id"] for d in catalogue["dishes"]]
    hist = history_tables(v, s, target_col, dish_ids, neighbours)

    v["n_similar"] = v.groupby(["date", "meal", "group"])["dish_id"].transform("count") - 1
    ctx_cols = ["date", "meal", "meal_code", "weekday", "temperature", "child_share", "first_meal_share",
                "attendance", *NAT_COLS, *[f"event_{e}" for e in EVENTS]]
    df = (v.merge(s[ctx_cols], on=["date", "meal"])
           .merge(static, on="dish_id")
           .merge(hist, on=["date", "meal", "dish_id"], how="left"))
    return df
