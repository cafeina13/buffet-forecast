"""Audit 3 part 2: recompute the monthly report and related metrics from data/ with the saved boosters.

Independent re-implementation of the evaluation (no training). Run:
  PYTHONIOENCODING=utf-8 uv run python audit/scripts/audit3/part2/recompute_report.py
"""
import json
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from olculu.config import load_dishes, load_synthetic
from olculu.features import DISH_FEATURES, GUEST_FEATURES, build_dish_frame, dish_static, prepare_services
from olculu import embeddings

ROOT = Path(__file__).resolve().parents[4]
MD = ROOT / "experiments" / "2026-10-24-audit2-fixes" / "models"
cfg, cat = load_synthetic(), load_dishes()
unit_kg = {d["id"]: d["kg_per_unit"] for d in cat["dishes"]}
refill = cfg["service"]["max_refill_share"]
end = date.fromisoformat(cfg["season"]["end"])
cutoff = pd.Timestamp(end - timedelta(weeks=cfg["season"]["blind_test_weeks"]) + timedelta(days=1))

def load(name):
    d = ROOT / "data" / name
    return {t: pd.read_csv(d / f"{t}.csv") for t in ["services", "servings", "hourly"]}

te = load("test")
s = prepare_services(te["services"])
gq = np.sort(np.column_stack([lgb.Booster(model_file=str(MD / f"guests_q{q}.txt")).predict(s[GUEST_FEATURES])
                              for q in [0.1, 0.5, 0.9]]), axis=1) * s["present"].to_numpy()[:, None]
s["att_pred"] = gq[:, 1]
opn = (s["date"] < cutoff).to_numpy()
a = s["attendance"].to_numpy()
print("guest coverage q10-q90 open:", np.mean((a[opn] >= gq[opn, 0]) & (a[opn] <= gq[opn, 2])))
for m in ["breakfast", "lunch", "dinner"]:
    k = opn & (s["meal"] == m).to_numpy()
    print("  ", m, "cov", round(np.mean((a[k] >= gq[k, 0]) & (a[k] <= gq[k, 2])), 3), "MAE", round(np.mean(np.abs(a[k]-gq[k,1])),2))
jul_s = opn & (s["date"].dt.month == 7).to_numpy()
print("guest coverage July:", np.mean((a[jul_s] >= gq[jul_s, 0]) & (a[jul_s] <= gq[jul_s, 2])))
# share of guests below / above
print("guest share attendance < q10:", np.mean(a[opn] < gq[opn, 0]), "> q90:", np.mean(a[opn] > gq[opn, 2]))

static, vecs = dish_static(cat, "tfidf")
nb = embeddings.nearest(vecs, 5)
f = build_dish_frame(te["servings"], te["services"], te["hourly"], cat, static, nb)
f = f.merge(s[["date", "meal", "att_pred"]], on=["date", "meal"])
ev = f[f["date"] < cutoff].copy()
Q = [0.1, 0.5, 0.8, 0.9, 0.95]
qr = np.sort(np.column_stack([lgb.Booster(model_file=str(MD / f"dishes_q{q}.txt")).predict(ev[DISH_FEATURES]) for q in Q]), axis=1)
qkg = qr * ev["att_pred"].to_numpy()[:, None] / 100
y = ev["true_demand_kg"].to_numpy()
print("rows", len(ev), "coverage", {q: round(float(np.mean(y <= qkg[:, i])), 4) for i, q in enumerate(Q)})
print("q10-q90 coverage", np.mean((y >= qkg[:, 0]) & (y <= qkg[:, 3])))
units = ev["dish_id"].map(unit_kg).to_numpy()

def sim(plan, y, u):
    cap = plan * (1 + refill)
    prod = np.where(y <= plan, plan, np.minimum(cap, plan + np.ceil(np.maximum(y - plan, 0) / u) * u))
    return prod, prod - np.minimum(y, prod), y > prod

jul = (ev["date"].dt.month == 7).to_numpy()
keys = ev.loc[jul, ["date", "meal"]].astype(str).agg("|".join, axis=1).to_numpy()
price = ev["group"].map(cfg["costs"]["try_per_kg"]).to_numpy()
cp, cw, cs = sim(ev["plan_kg"].to_numpy()[jul], y[jul], units[jul])
def svc(so):
    return pd.Series(so).groupby(keys).any().mean()
print("July n dish-servings", jul.sum(), "n services", len(set(keys)))
print("chef waste", cw.sum(), "stockout", cs.mean(), "svc", svc(cs), "plan_short", np.mean(y[jul] > ev["plan_kg"].to_numpy()[jul]))
# whole-unit refill: how much of chef/model waste comes from rounding up refill to whole units
for r in [0.2, 0.1, 0.05]:
    lvl = 1 - r
    rec_kg = np.array([np.interp(lvl, Q, row) for row in qkg[jul]])
    rec = np.maximum(units[jul] * 0.5, np.ceil(rec_kg / units[jul] * 2) / 2 * units[jul])
    p, w, so = sim(rec, y[jul], units[jul])
    refill_waste = np.where(y[jul] > rec, w, 0).sum()
    print(f"risk {r}: waste {w.sum():.1f} avoided {cw.sum()-w.sum():.1f} try {(cw*price[jul]).sum()-(w*price[jul]).sum():.0f}"
          f" stockout {so.mean():.4f} svc {svc(so):.4f} plan_short {np.mean(y[jul] > rec):.4f}"
          f" plan_short_unrounded {np.mean(y[jul] > rec_kg):.4f} waste_from_refill_rounding {refill_waste:.1f}")
    # stockout rate as share of dishes (dish ids) vs dish-servings
    print("   share of distinct dishes that ran out at least once:", ev.loc[jul].assign(so=so).groupby("dish_id")["so"].any().mean())
# hist_per100 definition check: does history start at test-season start?
print("first test date", s["date"].min(), "train first", pd.read_csv(ROOT/'data/train/services.csv')['date'].min())
# censoring effect on dish-servings: how often stock-outs in July (observed)
print("chef observed stockouts July (data)", ev.loc[jul, "stockout_time"].notna().mean())

# Per-month comparison: is July typical?
print("\nper month (open test): month, chef waste, then per risk: waste change %, model plan_short, model svc_stockout vs chef")
months = ev["date"].dt.month.to_numpy()
for mo in sorted(set(months)):
    k = months == mo
    kk = ev.loc[k, ["date", "meal"]].astype(str).agg("|".join, axis=1).to_numpy()
    _, cwm, csm = sim(ev["plan_kg"].to_numpy()[k], y[k], units[k])
    line = [f"{mo:2d} chef {cwm.sum():8.0f} svc {pd.Series(csm).groupby(kk).any().mean():.2f}"]
    for r in [0.2, 0.1, 0.05]:
        rec_kg = np.array([np.interp(1 - r, Q, row) for row in qkg[k]])
        rec = np.maximum(units[k] * 0.5, np.ceil(rec_kg / units[k] * 2) / 2 * units[k])
        _, w, so = sim(rec, y[k], units[k])
        line.append(f"r{r}: waste {100*(w.sum()/cwm.sum()-1):+5.1f}% short {np.mean(y[k] > rec):.3f} svc {pd.Series(so).groupby(kk).any().mean():.2f}")
    print(" | ".join(line))
