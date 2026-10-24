"""Audit 3, part 3: recompute nasil.html / README numbers from data/ and the saved boosters (no training).

Run: PYTHONIOENCODING=utf-8 uv run python audit/scripts/audit3/part3/recompute.py
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
from olculu.features import (DISH_FEATURES, GUEST_FEATURES, build_dish_frame, dish_static,  # noqa: E402
                             prepare_services)
from olculu.synth import apply_test_shift, generate  # noqa: E402

REC = ROOT / "experiments" / "2026-10-24-audit2-fixes"
MD = REC / "models"
out = {}

cfg, cat = load_synthetic(), load_dishes()
dishes = {d["id"]: d for d in cat["dishes"]}

# --- 0. data/ equals a fresh generator run (seed 42 / 43 shifted) ---
for name, c, seed in [("train", cfg, 42), ("test", apply_test_shift(cfg), 43)]:
    g = generate(c, cat, seed)
    disk = pd.read_csv(ROOT / "data" / name / "services.csv")
    gs = g["services"].copy()
    gs["date"] = gs["date"].astype(str)
    same = np.allclose(disk["attendance"], gs["attendance"]) and len(disk) == len(gs)
    disk_v = pd.read_csv(ROOT / "data" / name / "servings.csv")
    same_v = len(disk_v) == len(g["servings"]) and np.allclose(disk_v["true_demand_kg"], g["servings"]["true_demand_kg"])
    out[f"regen_{name}_matches"] = bool(same and same_v)

tr = {t: pd.read_csv(ROOT / "data/train" / f"{t}.csv") for t in ["services", "servings", "hourly"]}
te = {t: pd.read_csv(ROOT / "data/test" / f"{t}.csv") for t in ["services", "servings", "hourly"]}
s_tr, s_te = prepare_services(tr["services"]), prepare_services(te["services"])
end = pd.Timestamp(cfg["season"]["end"])
cutoff = end - pd.Timedelta(weeks=cfg["season"]["blind_test_weeks"]) + pd.Timedelta(days=1)
out["cutoff"] = str(cutoff.date())
open_te = s_te["date"] < cutoff
out["open_services"] = int(open_te.sum())
out["open_first_last"] = [str(s_te.loc[open_te, "date"].min().date()), str(s_te.loc[open_te, "date"].max().date())]

# --- 1. guest model from saved boosters ---
gb = {q: lgb.Booster(model_file=str(MD / f"guests_q{q}.txt")) for q in [0.1, 0.5, 0.9]}
rates = np.sort(np.column_stack([gb[q].predict(s_te[GUEST_FEATURES]) for q in [0.1, 0.5, 0.9]]), axis=1)
gq = rates * s_te["present"].to_numpy()[:, None]
s_te["att_pred"] = gq[:, 1]
o = s_te[open_te]
out["guest_mae_full"] = float((o["att_pred"] - o["attendance"]).abs().mean())
out["guest_cov_10_90"] = float(((o["attendance"] >= gq[open_te, 0]) & (o["attendance"] <= gq[open_te, 2])).mean())
meal_rate = (s_tr["attendance"] / s_tr["in_house"]).groupby(s_tr["meal"]).mean()
present_rate = s_tr.groupby("meal")["rate"].mean()
out["mae_in_house_x_rate"] = float((o["in_house"] * o["meal"].map(meal_rate) - o["attendance"]).abs().mean())
out["mae_present_x_rate"] = float((o["present"] * o["meal"].map(present_rate) - o["attendance"]).abs().mean())
out["guest_mae_by_meal"] = {m: float((g["att_pred"] - g["attendance"]).abs().mean()) for m, g in o.groupby("meal")}

# Learned event effects on the rate (median model), on open test dinners and on all dinners.
def effect(frame, flag):
    on, off = frame[GUEST_FEATURES].copy(), frame[GUEST_FEATURES].copy()
    for e in ["event_match", "event_themed", "event_conference"]:
        on[e] = 0
        off[e] = 0
    on[flag] = 1
    a, b = gb[0.5].predict(on), gb[0.5].predict(off)
    return {"mean_ratio": float(np.mean(a / b) - 1), "ratio_of_means": float(a.mean() / b.mean() - 1)}

for meal, flag in [("dinner", "event_match"), ("dinner", "event_themed"), ("lunch", "event_conference")]:
    out[f"effect_{flag}_{meal}_open_test"] = effect(o[o["meal"] == meal], flag)
    out[f"effect_{flag}_{meal}_train"] = effect(s_tr[s_tr["meal"] == meal], flag)
out["train_event_counts"] = s_tr.groupby(["meal", "event"]).size().to_dict()
out["train_event_counts"] = {f"{k[0]}:{k[1]}": int(v) for k, v in out["train_event_counts"].items()}
# Raw rate ratio in training data (event vs none) for reference
for meal, ev in [("dinner", "match"), ("dinner", "themed"), ("lunch", "conference")]:
    m = s_tr[s_tr["meal"] == meal]
    out[f"train_raw_rate_ratio_{ev}"] = float(m[m["event"] == ev]["rate"].mean() / m[m["event"] == "none"]["rate"].mean() - 1)

# --- 2. dish model ---
static, vecs = dish_static(cat, "tfidf")
nb5 = embeddings.nearest(vecs, 5)
f_te = build_dish_frame(te["servings"], te["services"], te["hourly"], cat, static, nb5)
f_te = f_te.merge(s_te[["date", "meal", "att_pred"]], on=["date", "meal"])
ev = f_te[f_te["date"] < cutoff].copy()
DQ = [0.1, 0.5, 0.8, 0.9, 0.95]
db = {q: lgb.Booster(model_file=str(MD / f"dishes_q{q}.txt")) for q in DQ}
raw = np.column_stack([db[q].predict(ev[DISH_FEATURES]) for q in DQ])
qr = np.sort(raw, axis=1)
out["dish_crossing_rows"] = int((np.abs(qr - raw) > 1e-9).any(axis=1).sum())
qkg = qr * ev["att_pred"].to_numpy()[:, None] / 100
y = ev["true_demand_kg"].to_numpy()
out["dish_rows"] = len(ev)
out["coverage"] = {q: float(np.mean(y <= qkg[:, i])) for i, q in enumerate(DQ)}

units = ev["dish_id"].map(lambda k: dishes[k]["kg_per_unit"]).to_numpy()
refill = cfg["service"]["max_refill_share"]


def sim(plan):
    cap = plan * (1 + refill)
    refilled = np.minimum(cap, plan + np.ceil(np.maximum(y - plan, 0) / units) * units)
    prod = np.where(y <= plan, plan, refilled)
    return prod, prod - np.minimum(y, prod), y > prod


def rec_for(risk):
    lvl = 1 - risk
    r = np.array([np.interp(lvl, DQ, row) for row in qkg])
    return np.maximum(units * 0.5, np.ceil(r / units * 2) / 2 * units)


keys = pd.MultiIndex.from_arrays([ev["date"].to_numpy(), ev["meal"].to_numpy()])
curve = {}
for name, plan in [("chef", ev["plan_kg"].to_numpy())] + [(f"risk{r}", rec_for(r)) for r in [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]]:
    prod, waste, so = sim(plan)
    curve[name] = {"waste_t": round(waste.sum() / 1000, 3), "stockout": round(so.mean(), 5),
                   "meals_any": round(pd.Series(so, index=keys).groupby(level=[0, 1]).any().mean(), 5),
                   "plan_short": round(np.mean(y > plan), 5),
                   "n_meals": int(pd.Series(so, index=keys).groupby(level=[0, 1]).any().size),
                   # share of dish-servings where the refill cap bound (demand > plan*(1.12))
                   "beyond_cap": round(np.mean(y > plan * (1 + refill)), 5),
                   "refill_overshoot_t": round(float((prod - plan)[(y > plan)].sum() - (np.minimum(y, prod) - plan)[(y > plan)].sum()) / 1000, 3)}
out["curve"] = curve

# --- 3. censoring factor on training season ---
f_tr = build_dish_frame(tr["servings"], tr["services"], tr["hourly"], cat, static, nb5)
sold = f_tr["stockout_time"].notna()
fac = f_tr.loc[sold, "target_kg"] / f_tr.loc[sold, "observed_kg"]
need = f_tr.loc[sold, "true_demand_kg"] / f_tr.loc[sold, "observed_kg"]
out["censor_train_rows"] = int(sold.sum())
out["censor_median_factor"] = float(fac.median())
out["censor_median_needed"] = float(need.median())
out["censor_capped_rows"] = int((fac >= 2 - 1e-9).sum())
out["censor_target_below_true_share"] = float((f_tr.loc[sold, "target_kg"] < f_tr.loc[sold, "true_demand_kg"]).mean())
sold_prone = ev.groupby("dish_id")["stockout_time"].transform(lambda s: s.notna().mean()) > 0.15
out["soldout_prone_rows"] = int(sold_prone.sum())
out["cov_q90_soldout_prone_aware"] = float(np.mean(y[sold_prone] <= qkg[sold_prone, 3]))
out["soldout_prone_dishes"] = sorted(ev.loc[sold_prone, "dish_id"].unique().tolist())

# --- 4. new dishes, neighbours, knn usage ---
ids = [d["id"] for d in cat["dishes"]]
nb3 = embeddings.nearest(vecs, 3)
new = [d for d in cat["dishes"] if d["freq"] == "new"]
out["n_dishes"] = len(cat["dishes"])
out["new_dishes"] = {d["name"]: {"from": d["new_from"], "nb3": [dishes[ids[j]]["name"] for j in nb3[ids.index(d["id"])]],
                                 "nb5": [dishes[ids[j]]["name"] for j in nb5[ids.index(d["id"])]]} for d in new}
first_served = te["servings"].groupby("dish_id")["date"].min()
out["new_first_served_test"] = {d["id"]: first_served.get(d["id"]) for d in new}
imp = {}
for q in DQ:
    gain = db[q].feature_importance("gain")
    imp[q] = round(float(gain[DISH_FEATURES.index("knn_per100")] / gain.sum()), 4)
out["knn_gain_share"] = imp
out["hist_gain_share_q0.5"] = round(float(db[0.5].feature_importance("gain")[DISH_FEATURES.index("hist_per100")]
                                    / db[0.5].feature_importance("gain").sum()), 4)
# Perturb knn_per100 by +-50% on all open rows; median change
X = ev[DISH_FEATURES].copy()
base = db[0.5].predict(X)
for f in [0.5, 1.5]:
    Xp = X.copy()
    Xp["knn_per100"] = Xp["knn_per100"] * f
    ch = db[0.5].predict(Xp) / base - 1
    out[f"knn_x{f}_median_abs_change"] = float(np.nanmedian(np.abs(ch)))
    newmask = ev["hist_per100"].isna().to_numpy()
    out[f"knn_x{f}_median_abs_change_nohist_rows"] = float(np.nanmedian(np.abs(ch[newmask]))) if newmask.any() else None
out["rows_without_hist"] = int(ev["hist_per100"].isna().sum())

# --- 5. demo checks ---
demo = json.loads((ROOT / "web/data/demo.json").read_text(encoding="utf-8"))
out["demo_dish_rows"] = sum(len(s["dishes"]) for s in demo["services"])
out["demo_dates"] = [demo["services"][0]["date"], demo["services"][-1]["date"]]
out["demo_child_share"] = sorted({round(s["guest_x"][GUEST_FEATURES.index("child_share")], 3) for s in demo["services"]})
out["support"] = demo["support"]
out["demo_new_dish_neighbours"] = {v["name"]: v["neighbours"] for v in demo["dishes"].values() if v["new"]}

print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
