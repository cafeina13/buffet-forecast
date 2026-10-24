"""Sensitivity of the baseline (all-learned) guest median model to excursions / a la carte, vs the generator."""
import sys
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from olculu.features import prepare_services
from olculu.config import load_synthetic
b = lgb.Booster(model_file=str(ROOT / "experiments/2026-10-24-baseline/models/guests_q0.5.txt"))
feats = b.feature_name()
print(feats)
s = prepare_services(pd.read_csv(ROOT / "data/test/services.csv"))
s = s[s["date"] < "2026-10-11"]
cfg = load_synthetic()
for meal, col, att in [("lunch", "excursions", cfg["attendance"]["lunch"]), ("dinner", "alacarte", cfg["attendance"]["dinner"]),
                       ("dinner", "arrivals_night", cfg["attendance"]["dinner"])]:
    X = s[s["meal"] == meal][feats].copy()
    d = X[col].std()
    lo, hi = X.copy(), X.copy()
    lo[col] -= d; hi[col] += d
    slope = np.mean((b.predict(hi) - b.predict(lo)) / (2 * d))
    print(meal, col, "model slope", round(slope, 3), "generator slope", -att, "ratio", round(att / -slope, 2) if slope else None)
print("--- other perturbation sizes")
for meal, col, att in [("lunch", "excursions", 0.62), ("dinner", "alacarte", 0.82)]:
    X = s[s["meal"] == meal][feats].copy()
    for name, a, c in [("p5-p95", X[col].quantile(.05), X[col].quantile(.95)), ("0-max", 0, X[col].max()),
                       ("x0.5-x2", None, None), ("+-10 guests", None, None)]:
        lo, hi = X.copy(), X.copy()
        if name == "x0.5-x2":
            lo[col] *= 0.5; hi[col] *= 2; delta = (hi[col] - lo[col]).to_numpy()
        elif name == "+-10 guests":
            lo[col] -= 10; hi[col] += 10; delta = 20
        else:
            lo[col] = a; hi[col] = c; delta = c - a
        slope = np.mean((b.predict(hi) - b.predict(lo)) / delta)
        print(meal, col, name, "ratio", round(att / -slope, 2))
