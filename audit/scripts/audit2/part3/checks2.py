"""Audit 2 part 3: sensitivity of the earlier count-based guest model; knn ablation; misc."""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from olculu import embeddings  # noqa: E402
from olculu.config import load_dishes  # noqa: E402
from olculu.features import DISH_FEATURES, GUEST_FEATURES, build_dish_frame, dish_static, prepare_services  # noqa: E402
from olculu.models import DISH_QUANTILES, interpolate_quantile  # noqa: E402
from olculu.newsvendor import round_up_half_unit, simulate  # noqa: E402

cat = load_dishes()
te = {t: pd.read_csv(ROOT / "data" / "test" / f"{t}.csv") for t in ["services", "servings", "hourly"]}
s = prepare_services(te["services"])
op = s[s.date < "2026-10-11"].copy()

# 1. Earlier count-based guest model (baseline record): response to excursions / a la carte
B = ROOT / "experiments" / "2026-10-24-baseline" / "models"
feats = pd.read_csv(B / "guest_rows_sample.csv", nrows=1).columns.tolist()
b5 = lgb.Booster(model_file=str(B / "guests_q0.5.txt"))
for col, meal, rate in [("excursions", "lunch", 0.62), ("alacarte", "dinner", 0.82)]:
    X = op.loc[op.meal == meal, feats].copy()
    base = b5.predict(X)
    for d in [10, 20, 40, 80]:
        X2 = X.copy(); X2[col] += d
        resp = np.mean(base - b5.predict(X2))
        print(f"baseline model {col} +{d}: predicted drop {resp:.2f}, generator drop {rate*d:.2f}, ratio {rate*d/max(resp,1e-9):.2f}x")

# 2. knn_per100 ablation on the current dish models
REC = ROOT / "experiments" / "2026-10-24-audit1-fixes" / "models"
gq = lgb.Booster(model_file=str(REC / "guests_q0.5.txt"))
s["att_pred"] = gq.predict(s[GUEST_FEATURES]) * s["present"]
static, vecs = dish_static(cat, "tfidf")
f = build_dish_frame(te["servings"], te["services"], te["hourly"], cat, static, embeddings.nearest(vecs, 5)).merge(
    s[["date", "meal", "att_pred"]], on=["date", "meal"])
ev = f[f.date < "2026-10-11"].copy()
dq = {q: lgb.Booster(model_file=str(REC / f"dishes_q{q}.txt")) for q in DISH_QUANTILES}
units = ev.dish_id.map({d["id"]: d["kg_per_unit"] for d in cat["dishes"]}).to_numpy()
y = ev.true_demand_kg.to_numpy()

def run(X):
    q = np.sort(np.column_stack([dq[k].predict(X) for k in DISH_QUANTILES]), axis=1) * ev.att_pred.to_numpy()[:, None] / 100
    return q

qa = run(ev[DISH_FEATURES])
Xn = ev[DISH_FEATURES].copy(); Xn["knn_per100"] = np.nan
qb = run(Xn)
newids = [d["id"] for d in cat["dishes"] if d["freq"] == "new"]
nm = ev.dish_id.isin(newids).to_numpy()
print("knn->NaN: mean abs change of median, all rows %.2f%%, new-dish rows %.2f%%" % (
    np.mean(np.abs(qb[:, 1] / qa[:, 1] - 1)) * 100, np.mean(np.abs(qb[nm, 1] / qa[nm, 1] - 1)) * 100))
first = nm & ev.hist_per100.isna().to_numpy()
print("first servings of new dishes (no own history):", first.sum(), "median change %.2f%%" % (np.mean(np.abs(qb[first, 1] / qa[first, 1] - 1)) * 100))
print("first servings: abs err with knn %.3f kg, without %.3f kg" % (np.mean(np.abs(y[first]-qa[first,1])), np.mean(np.abs(y[first]-qb[first,1]))))
for name, q in [("with knn", qa), ("knn NaN", qb)]:
    rec = round_up_half_unit(interpolate_quantile(q, DISH_QUANTILES, 0.9), units)
    sim = simulate(rec, y, units, 0.12)
    print(name, "10%% risk waste %.2f t stockout %.2f%%" % (sim["waste"].sum() / 1000, sim["stockout"].mean() * 100))

# 3. breakfast rate vs temperature in test season (generator has no such effect)
for nm_, ss in [("test", s)]:
    b = ss[ss.meal == "breakfast"]
    print(nm_, "breakfast corr(rate,temp)", round(b.rate.corr(b.temperature), 3))
