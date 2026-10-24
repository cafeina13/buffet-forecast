"""Audit 2 part 3: recompute nasil.html / README numbers from data/ and saved boosters (no training)."""
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
from olculu.features import (DISH_FEATURES, GUEST_FEATURES, build_dish_frame, dish_static,  # noqa: E402
                             prepare_services)
from olculu.models import DISH_QUANTILES, interpolate_quantile  # noqa: E402
from olculu.newsvendor import round_up_half_unit, simulate  # noqa: E402

REC = ROOT / "experiments" / "2026-10-24-audit1-fixes" / "models"
cfg, cat = load_synthetic(), load_dishes()
dishes = {d["id"]: d for d in cat["dishes"]}
unit_kg = {k: d["kg_per_unit"] for k, d in dishes.items()}

def load(name):
    return {t: pd.read_csv(ROOT / "data" / name / f"{t}.csv") for t in ["services", "servings", "hourly"]}

tr, te = load("train"), load("test")
s_tr, s_te = prepare_services(tr["services"]), prepare_services(te["services"])
cutoff = pd.Timestamp("2026-10-11")
open_te = s_te["date"] < cutoff
print("test season dates", s_te["date"].min().date(), s_te["date"].max().date(), "open last", s_te.loc[open_te, "date"].max().date())
print("open services", open_te.sum(), "days", s_te.loc[open_te, "date"].nunique())

# generator checks: present == eligible -> rate
print("events train:", tr["services"].groupby(["meal", "event"]).size().to_dict())
print("events test open:", s_te[open_te].groupby(["meal", "event"]).size().to_dict())

# test shift visible?
for nm, s in [("train", s_tr), ("test", s_te)]:
    print(nm, "occ max", round(s.occupancy.max(), 3), "child max", round(s.child_share.max(), 3),
          "temp max", round(s.temperature.max(), 1), "exc share mean", round((s.excursions / s.in_house).mean(), 4))

gq = {q: lgb.Booster(model_file=str(REC / f"guests_q{q}.txt")) for q in [0.1, 0.5, 0.9]}
rate = np.sort(np.column_stack([gq[q].predict(s_te[GUEST_FEATURES]) for q in [0.1, 0.5, 0.9]]), axis=1)
g = rate * s_te["present"].to_numpy()[:, None]
att = s_te["attendance"]
o = open_te.to_numpy()
print("MAE full", np.mean(np.abs(g[o, 1] - att[o])))
print("cov 10-90", np.mean((att[o] >= g[o, 0]) & (att[o] <= g[o, 2])))
mr = (s_tr.attendance / s_tr.in_house).groupby(s_tr.meal).mean()
pr = s_tr.groupby("meal")["rate"].mean()
ote = s_te[open_te]
print("MAE inhouse*rate", (ote.in_house * ote.meal.map(mr) - ote.attendance).abs().mean())
print("MAE present*rate", (ote.present * ote.meal.map(pr) - ote.attendance).abs().mean())
print("crossing guest rows open/all", int((np.abs(np.sort(np.column_stack([gq[q].predict(s_te[GUEST_FEATURES]) for q in [0.1,0.5,0.9]]),axis=1)-np.column_stack([gq[q].predict(s_te[GUEST_FEATURES]) for q in [0.1,0.5,0.9]]))>1e-9).any(axis=1).sum()))

# event effects learned by the median rate model: on minus off, over dinners / lunches of the open test season
for ev_name, meal in [("match", "dinner"), ("themed", "dinner"), ("conference", "lunch")]:
    X = s_te.loc[open_te & (s_te.meal == meal), GUEST_FEATURES].copy()
    for e in ["match", "themed", "conference"]:
        X[f"event_{e}"] = 0
    off = gq[0.5].predict(X)
    X[f"event_{ev_name}"] = 1
    on = gq[0.5].predict(X)
    print(f"learned {ev_name} ({meal}) effect on median rate: {np.mean(on/off-1)*100:.2f}% (ratio of sums {(on.sum()/off.sum()-1)*100:.2f}%)")
    Xt = s_tr.loc[s_tr.meal == meal, GUEST_FEATURES].copy()
    for e in ["match", "themed", "conference"]:
        Xt[f"event_{e}"] = 0
    off = gq[0.5].predict(Xt); Xt[f"event_{ev_name}"] = 1; on = gq[0.5].predict(Xt)
    print(f"   same on train-season rows: {np.mean(on/off-1)*100:.2f}%")

# rate dependence on weather/occupancy in data (should be none)
for meal in ["breakfast", "lunch", "dinner"]:
    sub = s_tr[(s_tr.meal == meal) & (s_tr.event == "none")]
    print(meal, "corr(rate,temp)", round(sub.rate.corr(sub.temperature), 3), "corr(rate,occ)", round(sub.rate.corr(sub.occupancy), 3),
          "corr(rate,nat_RU)", round(sub.rate.corr(sub.nat_RU), 3), "mean rate", round(sub.rate.mean(), 4))

# dish evaluation
static, vecs = dish_static(cat, "tfidf")
nb5 = embeddings.nearest(vecs, 5)
s_te["att_pred"] = g[:, 1]
f_te = build_dish_frame(te["servings"], te["services"], te["hourly"], cat, static, nb5).merge(
    s_te[["date", "meal", "att_pred"]], on=["date", "meal"])
ev = f_te[f_te["date"] < cutoff].copy()
dq = {q: lgb.Booster(model_file=str(REC / f"dishes_q{q}.txt")) for q in DISH_QUANTILES}
raw = np.column_stack([dq[q].predict(ev[DISH_FEATURES]) for q in DISH_QUANTILES])
qr = np.sort(raw, axis=1)
print("dish crossing rows", int((np.abs(qr - raw) > 1e-9).any(axis=1).sum()), "rows", len(ev))
qkg = qr * ev["att_pred"].to_numpy()[:, None] / 100
y = ev["true_demand_kg"].to_numpy()
for i, q in enumerate(DISH_QUANTILES):
    print("cov", q, np.mean(y <= qkg[:, i]))
units = ev["dish_id"].map(unit_kg).to_numpy()
keys = [ev["date"].to_numpy(), ev["meal"].to_numpy()]
nserv = pd.Series(1, index=pd.MultiIndex.from_arrays(keys)).groupby(level=[0, 1]).size().shape[0]
print("services in ev", nserv)

def swo(st):
    return pd.Series(st).groupby(keys).any().mean()

chef = simulate(ev.plan_kg.to_numpy(), y, units, 0.12)
print("chef waste via simulate", chef["waste"].sum(), "via data production-observed", (ev.production_kg - ev.observed_kg).sum(),
      "via leftover_kg", ev.leftover_kg.sum())
print("chef stockout", chef["stockout"].mean(), "data stockout_time notna", ev.stockout_time.notna().mean(),
      "swo", swo(chef["stockout"]), "plan short", np.mean(y > ev.plan_kg))
for r in [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]:
    rec = round_up_half_unit(interpolate_quantile(qkg, DISH_QUANTILES, 1 - r), units)
    sim = simulate(rec, y, units, 0.12)
    print(f"risk {r}: waste {sim['waste'].sum()/1000:.2f} t, stockout {sim['stockout'].mean()*100:.2f}%, "
          f"swo {swo(sim['stockout'])*100:.2f}%, plan_short {np.mean(y>rec)*100:.2f}%, "
          f"production {sim['production'].sum()/1000:.1f} t, refill used share {np.mean(sim['production']>rec)*100:.1f}%")
print("chef production t", chef["production"].sum() / 1000, "true demand t", y.sum() / 1000)

# censoring factor on training season
srv = tr["servings"]
sold = srv.stockout_time.notna()
from olculu.features import censoring_corrected  # noqa: E402
corr = censoring_corrected(srv, tr["hourly"])
fac = corr[sold] / srv.loc[sold, "observed_kg"]
need = srv.loc[sold, "true_demand_kg"] / srv.loc[sold, "observed_kg"]
print("train sold-out servings", sold.sum(), "median factor applied", fac.median(), "median needed", need.median(),
      "share capped", (fac >= 2 - 1e-9).mean(), "corrected<true share", (corr[sold] < srv.loc[sold, 'true_demand_kg']).mean())

# feature importance of knn_per100 in dish models
for q, b in dq.items():
    imp = dict(zip(b.feature_name(), b.feature_importance("gain")))
    tot = sum(imp.values())
    rank = sorted(imp, key=imp.get, reverse=True)
    print("q", q, "knn gain share", round(imp["knn_per100"] / tot * 100, 2), "% rank", rank.index("knn_per100") + 1,
          "top5", rank[:5])

# knn sensitivity on new dishes in eval rows
newids = [d["id"] for d in cat["dishes"] if d["freq"] == "new"]
nm = ev.dish_id.isin(newids)
X = ev.loc[nm, DISH_FEATURES].copy()
base = dq[0.5].predict(X)
for f in [0.5, 1.5]:
    X2 = X.copy(); X2["knn_per100"] = X2["knn_per100"] * f
    print("knn x", f, "median change on new-dish rows (mean abs %)", np.nanmean(np.abs(dq[0.5].predict(X2) / base - 1)) * 100)
print("new dish rows", nm.sum(), "with hist NaN", ev.loc[nm, "hist_per100"].isna().sum())

# neighbours (3) recompute
near3 = embeddings.nearest(vecs, 3)
ids = [d["id"] for d in cat["dishes"]]
for i, d in enumerate(cat["dishes"]):
    if d["freq"] == "new":
        print(d["name"], [dishes[ids[j]]["name"] for j in near3[i]])
