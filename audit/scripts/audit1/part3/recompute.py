"""Part 3 audit: recompute nasil.html / README numbers from data + saved models (no training).

Run: PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1 uv run python audit/scripts/audit1/part3/recompute.py
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from olculu.config import load_dishes, load_synthetic, DATA_DIR  # noqa: E402
from olculu.features import (DISH_FEATURES, GUEST_FEATURES, prepare_services, build_dish_frame,  # noqa: E402
                             dish_static)
from olculu import embeddings  # noqa: E402
from olculu.synth import generate, apply_test_shift  # noqa: E402

REC = ROOT / "experiments" / "2026-10-24-guest-rule" / "models"
out = {}

cfg, cat = load_synthetic(), load_dishes()
dishes = {d["id"]: d for d in cat["dishes"]}

# 0. Data reproducibility: regenerate both seasons and compare with CSVs.
for name, c, seed in [("train", cfg, 42), ("test", apply_test_shift(cfg), 43)]:
    g = generate(c, cat, seed)
    for t in ["services", "servings"]:
        csv = pd.read_csv(DATA_DIR / name / f"{t}.csv")
        gen = g[t].copy()
        gen["date"] = gen["date"].astype(str)
        num = [col for col in csv.columns if csv[col].dtype.kind == "f"]
        same = len(csv) == len(gen) and np.allclose(csv[num].to_numpy(), gen[num].astype(float).to_numpy(),
                                                    equal_nan=True)
        out[f"regen_{name}_{t}_same"] = bool(same)

train = {t: pd.read_csv(DATA_DIR / "train" / f"{t}.csv") for t in ["services", "servings", "hourly"]}
test = {t: pd.read_csv(DATA_DIR / "test" / f"{t}.csv") for t in ["services", "servings", "hourly"]}
s_te = prepare_services(test["services"])
cutoff = pd.Timestamp("2026-10-11")
open_te = s_te["date"] < cutoff
out["season"] = [str(s_te["date"].min().date()), str(s_te["date"].max().date())]
out["n_open_days"] = int(s_te.loc[open_te, "date"].nunique())
out["n_open_services"] = int(open_te.sum())
out["rooms_x_gpr"] = cfg["hotel"]["rooms"] * cfg["hotel"]["guests_per_room"]
out["n_dishes"] = len(cat["dishes"])
out["new_dishes"] = {d["id"]: (d["name"], d["new_from"]) for d in cat["dishes"] if d["freq"] == "new"}

# 1. Guest model from saved boosters.
gb = {q: lgb.Booster(model_file=str(REC / f"guests_q{q}.txt")) for q in [0.1, 0.5, 0.9]}
rate = np.sort(np.column_stack([gb[q].predict(s_te[GUEST_FEATURES]) for q in [0.1, 0.5, 0.9]]), axis=1)
g = rate * s_te["present"].to_numpy()[:, None]
att = s_te["attendance"].to_numpy()
o = open_te.to_numpy()
out["guest_mae_full"] = float(np.mean(np.abs(g[o, 1] - att[o])))
out["guest_cov_10_90"] = float(np.mean((att[o] >= g[o, 0]) & (att[o] <= g[o, 2])))
# Simple non-ML baselines for context.
present = s_te["present"].to_numpy()
mean_rate_by_meal = s_te[o].groupby("meal")["rate"].mean()
out["mae_rule_x_const_rate_from_test"] = float(np.mean(np.abs(present[o] * s_te.loc[o, "meal"].map(mean_rate_by_meal) - att[o])))
s_tr = prepare_services(train["services"])
tr_rate = s_tr.groupby("meal")["rate"].mean()
out["mae_rule_x_train_meal_rate"] = float(np.mean(np.abs(present[o] * s_te.loc[o, "meal"].map(tr_rate) - att[o])))
tr_r2 = s_tr.groupby("meal").apply(lambda d: (d["attendance"] / d["in_house"]).mean())
out["mae_inhouse_x_train_meal_rate"] = float(np.mean(np.abs(s_te.loc[o, "in_house"] * s_te.loc[o, "meal"].map(tr_r2) - att[o])))

# 2. Dish frame + saved dish boosters -> risk curve.
static, vecs = dish_static(cat, "tfidf")
nb = embeddings.nearest(vecs, 5)
s_te["att_pred"] = g[:, 1]
f_te = build_dish_frame(test["servings"], test["services"], test["hourly"], cat, static, nb)
f_te = f_te.merge(s_te[["date", "meal", "att_pred"]], on=["date", "meal"])
ev = f_te[f_te["date"] < cutoff].copy()
QS = [0.1, 0.5, 0.8, 0.9, 0.95]
db = {q: lgb.Booster(model_file=str(REC / f"dishes_q{q}.txt")) for q in QS}
raw = np.column_stack([db[q].predict(ev[DISH_FEATURES]) for q in QS])
qr = np.sort(raw, axis=1)
out["dish_crossing_rows"] = int((np.abs(qr - raw) > 1e-9).any(axis=1).sum())
qkg = qr * ev["att_pred"].to_numpy()[:, None] / 100
y = ev["true_demand_kg"].to_numpy()
out["dish_rows"] = len(ev)
out["dish_cov"] = {q: float(np.mean(y <= qkg[:, i])) for i, q in enumerate(QS)}
units = ev["dish_id"].map(lambda k: dishes[k]["kg_per_unit"]).to_numpy()
refill = cfg["service"]["max_refill_share"]


def simulate(plan, true, unit):
    cap = plan * (1 + refill)
    refilled = np.minimum(cap, plan + np.ceil(np.maximum(true - plan, 0) / unit) * unit)
    prod = np.where(true <= plan, plan, refilled)
    cons = np.minimum(true, prod)
    return prod - cons, true > prod, true > plan


def rup(kg, u):
    return np.maximum(u * 0.5, np.ceil(kg / u * 2) / 2 * u)


svc_key = ev["date"].astype(str) + "|" + ev["meal"]
cw, cs, cplan = simulate(ev["plan_kg"].to_numpy(), y, units)
out["chef"] = {"waste_t": cw.sum() / 1000, "stockout_rows": float(cs.mean()),
               "share_services_with_any_stockout": float(pd.Series(cs).groupby(svc_key.values).any().mean()),
               "exceed_initial_plan": float(cplan.mean())}
# chef waste vs generator's own leftover (no scale noise)
gen_waste = (ev["production_kg"] - ev["observed_kg"]).sum() / 1000
out["chef_waste_from_generator_t"] = float(gen_waste)
out["chef_stockout_from_generator"] = float(ev["stockout_time"].notna().mean())
curve = []
for r in [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]:
    lvl = np.clip(1 - r, 0.1, 0.95)
    plan = rup(np.array([np.interp(lvl, QS, row) for row in qkg]), units)
    w, s, ex = simulate(plan, y, units)
    curve.append({"risk": r, "waste_t": round(w.sum() / 1000, 3), "stockout_rows": round(float(s.mean()), 5),
                  "share_services_any_stockout": round(float(pd.Series(s).groupby(svc_key.values).any().mean()), 4),
                  "exceed_initial_plan": round(float(ex.mean()), 4)})
out["curve"] = curve
out["saving_at_0.2"] = 1 - curve[3]["waste_t"] / out["chef"]["waste_t"]
out["mean_dishes_per_service"] = float(ev.groupby(["date", "meal"]).size().mean())

# 3. Map: 2-D PCA explained variance and neighbour preservation.
from sklearn.decomposition import PCA  # noqa: E402
p = PCA(2, random_state=0).fit(vecs)
out["pca2_explained_var"] = float(p.explained_variance_ratio_.sum())
xy = p.transform(vecs)
demo = json.load(open(ROOT / "web" / "data" / "demo.json", encoding="utf-8"))
mxy = np.array([[m["x"], m["y"]] for m in demo["map"]])
out["map_matches_pca"] = bool(np.allclose(np.abs(mxy), np.abs(xy), atol=1e-4))
ids = [d["id"] for d in cat["dishes"]]
n3 = embeddings.nearest(vecs, 3)
D2 = np.linalg.norm(xy[:, None] - xy[None], axis=2)
np.fill_diagonal(D2, np.inf)
ranks = []
new_info = {}
for i, d in enumerate(cat["dishes"]):
    order = np.argsort(D2[i])
    r = [int(np.where(order == j)[0][0]) + 1 for j in n3[i]]
    ranks.extend(r)
    if d["freq"] == "new":
        new_info[d["name"]] = {"neighbours_hd": [dishes[ids[j]]["name"] for j in n3[i]],
                               "rank_in_2d_of_those": r,
                               "nearest_in_2d": [dishes[ids[j]]["name"] for j in order[:3]]}
out["neighbour_2d_rank_median"] = float(np.median(ranks))
out["share_hd_top3_within_2d_top5"] = float(np.mean(np.array(ranks) <= 5))
out["new_dish_neighbours"] = new_info
# model uses 5 neighbours; which are they for new dishes, and do they share the new dish's meals?
out["new_dish_knn5"] = {dishes[ids[i]]["name"]: [(dishes[ids[j]]["name"], dishes[ids[j]]["meals"]) for j in nb[i]]
                        for i, d in enumerate(cat["dishes"]) if d["freq"] == "new"}
# new dishes: first-day knn feature present?
for k in out["new_dishes"]:
    rows = ev[ev["dish_id"] == k].sort_values("date")
    if len(rows):
        out.setdefault("new_first_rows", {})[k] = {
            "first_date": str(rows["date"].iloc[0].date()),
            "hist_nan_first": bool(np.isnan(rows["hist_per100"].iloc[0])),
            "knn_first": None if np.isnan(rows["knn_per100"].iloc[0]) else float(rows["knn_per100"].iloc[0])}

# 4. Censoring: how often, how big
sold = ev["stockout_time"].notna()
out["eval_rows_soldout_share"] = float(sold.mean())
tr_f = build_dish_frame(train["servings"], train["services"], train["hourly"], cat, static, nb)
ts = tr_f["stockout_time"].notna()
fac = tr_f.loc[ts, "target_kg"] / tr_f.loc[ts, "observed_kg"]
tru = tr_f.loc[ts, "true_demand_kg"] / tr_f.loc[ts, "observed_kg"]
out["train_soldout_share"] = float(ts.mean())
out["train_corr_factor_median"] = float(fac.median())
out["train_true_over_obs_median"] = float(tru.median())
out["train_corrected_le_true_share"] = float((tr_f.loc[ts, "target_kg"] <= tr_f.loc[ts, "true_demand_kg"] + 1e-9).mean())
out["train_factor_capped_share"] = float((fac >= 2 - 1e-9).mean())

print(json.dumps(out, indent=1, ensure_ascii=False, default=float))
