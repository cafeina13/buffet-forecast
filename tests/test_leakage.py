"""True demand must never reach the models, and no feature may use future data."""

import numpy as np
import pandas as pd
import pytest

from olculu import embeddings
from olculu.config import load_dishes, load_synthetic
from olculu.features import DISH_FEATURES, FORBIDDEN, GUEST_FEATURES, build_dish_frame, dish_static
from olculu.synth import generate


@pytest.fixture(scope="module")
def season():
    cfg, catalogue = load_synthetic(), load_dishes()
    # A short season keeps the test fast.
    cfg["season"]["end"] = "2026-05-31"
    tables = generate(cfg, catalogue, seed=7)
    static, vecs = dish_static(catalogue, "tfidf")
    return cfg, catalogue, tables, static, embeddings.nearest(vecs, 5)


def test_true_demand_is_not_a_feature():
    assert not FORBIDDEN & set(DISH_FEATURES)
    assert not FORBIDDEN & set(GUEST_FEATURES)
    assert not any("true" in f for f in DISH_FEATURES + GUEST_FEATURES)


def test_features_do_not_change_when_true_demand_changes(season):
    _, catalogue, t, static, nb = season
    a = build_dish_frame(t["servings"], t["services"], t["hourly"], catalogue, static, nb)
    shuffled = t["servings"].copy()
    shuffled["true_demand_kg"] = np.random.default_rng(0).permutation(shuffled["true_demand_kg"].to_numpy())
    b = build_dish_frame(shuffled, t["services"], t["hourly"], catalogue, static, nb)
    pd.testing.assert_frame_equal(a[DISH_FEATURES + ["target_kg"]], b[DISH_FEATURES + ["target_kg"]])


def test_history_features_use_only_earlier_days(season):
    _, catalogue, t, static, nb = season
    base = build_dish_frame(t["servings"], t["services"], t["hourly"], catalogue, static, nb)
    cut = pd.Timestamp("2026-05-15")
    # Wipe everything observed from the cut-off onwards; features up to the cut-off must not move.
    future = t["servings"].copy()
    late = pd.to_datetime(future["date"]) >= cut
    future.loc[late, ["observed_kg"]] = future.loc[late, "observed_kg"] * 10
    changed = build_dish_frame(future, t["services"], t["hourly"], catalogue, static, nb)
    cols = ["hist_per100", "knn_per100"]
    keep = base["date"] <= cut
    pd.testing.assert_frame_equal(base.loc[keep, cols].reset_index(drop=True),
                                  changed.loc[keep, cols].reset_index(drop=True))


def test_sold_out_services_are_scaled_up_not_down(season):
    _, catalogue, t, static, nb = season
    f = build_dish_frame(t["servings"], t["services"], t["hourly"], catalogue, static, nb)
    sold = f["stockout_time"].notna()
    assert sold.any()
    assert (f.loc[sold, "target_kg"] >= f.loc[sold, "observed_kg"] - 1e-9).all()
    assert np.allclose(f.loc[~sold, "target_kg"], f.loc[~sold, "observed_kg"])
