"""Audit 3, part 3: rounding vs calibration, baseline sensitivity, parity case export (no training).

Run: PYTHONIOENCODING=utf-8 uv run python audit/scripts/audit3/part3/checks2.py <scratch_dir>
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from olculu import embeddings  # noqa: E402
from olculu.config import load_dishes, load_synthetic  # noqa: E402
from olculu.features import DISH_FEATURES, GUEST_FEATURES, build_dish_frame, dish_static, prepare_services  # noqa: E402

scratch = Path(sys.argv[1])
MD = ROOT / "experiments/2026-10-24-audit2-fixes/models"
cfg, cat = load_synthetic(), load_dishes()
dishes = {d["id"]: d for d in cat["dishes"]}
te = {t: pd.read_csv(ROOT / "data/test" / f"{t}.csv") for t in ["services", "servings", "hourly"]}
s_te = prepare_services(te["services"])
cutoff = pd.Timestamp("2026-10-11")
gb = {q: lgb.Booster(model_file=str(MD / f"guests_q{q}.txt")) for q in [0.1, 0.5, 0.9]}
s_te["att_pred"] = gb[0.5].predict(s_te[GUEST_FEATURES]) * s_te["present"]
static, vecs = dish_static(cat, "tfidf")
f_te = build_dish_frame(te["servings"], te["services"], te["hourly"], cat, static, embeddings.nearest(vecs, 5))
f_te = f_te.merge(s_te[["date", "meal", "att_pred"]], on=["date", "meal"])
ev = f_te[f_te["date"] < cutoff].copy()
DQ = [0.1, 0.5, 0.8, 0.9, 0.95]
db = {q: lgb.Booster(model_file=str(MD / f"dishes_q{q}.txt")) for q in DQ}
qkg = np.sort(np.column_stack([db[q].predict(ev[DISH_FEATURES]) for q in DQ]), axis=1) * ev["att_pred"].to_numpy()[:, None] / 100
y = ev["true_demand_kg"].to_numpy()
units = ev["dish_id"].map(lambda k: dishes[k]["kg_per_unit"]).to_numpy()
res = {}
for r in [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]:
    raw = np.array([np.interp(1 - r, DQ, row) for row in qkg])
    rounded = np.maximum(units * 0.5, np.ceil(raw / units * 2) / 2 * units)
    res[r] = {"exceed_unrounded": round(float(np.mean(y > raw)), 4), "exceed_rounded": round(float(np.mean(y > rounded)), 4),
              "round_up_mean_kg": round(float(np.mean(rounded - raw)), 3)}
print("rounding vs calibration:", json.dumps(res, indent=0))

# Baseline (count-based) guest model sensitivity to excursions / a la carte
BM = ROOT / "experiments/2026-10-24-baseline/models/guests_q0.5.txt"
bb = lgb.Booster(model_file=str(BM))
feats = bb.feature_name()
o = s_te[s_te["date"] < cutoff].copy()
o["arrivals_day"] = te["services"].loc[o.index, "arrivals_day"]
for meal, col, rate in [("lunch", "excursions", 0.62), ("dinner", "alacarte", 0.82)]:
    m = o[o["meal"] == meal]
    base = bb.predict(m[feats])
    for d in [20, 50, 100]:
        x = m[feats].copy()
        x[col] = x[col] + d
        ch = bb.predict(x) - base
        print(f"baseline {meal} {col}+{d}: model mean change {ch.mean():.2f}, generator {-rate * d:.1f}, ratio {(-rate * d) / ch.mean() if ch.mean() else float('inf'):.2f}")

# Parity cases: every open-test row, every quantile, LightGBM predictions
cases = []
for web, frame, feats_, boosters in [("model-dishes.json", ev[DISH_FEATURES], DISH_FEATURES, db),
                                     ("model-guests.json", s_te[GUEST_FEATURES], GUEST_FEATURES, gb)]:
    X = frame.to_numpy(dtype=float)
    for i, q in enumerate(sorted(boosters)):
        cases.append({"web": web, "qi": i, "pred": boosters[q].predict(frame).tolist()})
    cases.append({"web": web, "rows": [[None if np.isnan(v) else v for v in r] for r in X.tolist()]})
(scratch / "parity_all.json").write_text(json.dumps(cases), encoding="utf-8")
print("rows", len(ev), len(s_te))
