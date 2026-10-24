"""Independent recomputation for the chef's screen audit (part 2).

Uses the saved LightGBM boosters of the published record (no training), the project's
feature builder (to get identical feature rows), and my own simulation/aggregation.
Run: PYTHONIOENCODING=utf-8 uv run python audit/scripts/audit1/part2/recompute.py
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

REC = ROOT / "experiments/2026-10-24-guest-rule/models"
OUT = Path(__file__).parent
cfg, cat = load_synthetic(), load_dishes()
dishes = {d["id"]: d for d in cat["dishes"]}

G = {q: lgb.Booster(model_file=str(REC / f"guests_q{q}.txt")) for q in [0.1, 0.5, 0.9]}
D = {q: lgb.Booster(model_file=str(REC / f"dishes_q{q}.txt")) for q in [0.1, 0.5, 0.8, 0.9, 0.95]}

# ---- 1. parity of the web JSON with the boosters on the actual demo rows ----
demo = json.loads((ROOT / "web/data/demo.json").read_text(encoding="utf-8"))
grows = np.array([s["guest_x"] for s in demo["services"]], dtype=float)
drows = np.array([[np.nan if v is None else v for v in d["x"]] for s in demo["services"] for d in s["dishes"]], dtype=float)
parity = {"guest": {str(q): G[q].predict(grows).tolist() for q in G},
          "dish": {str(q): D[q].predict(drows).tolist() for q in D},
          "guest_contrib": G[0.5].predict(grows, pred_contrib=True).tolist(),
          "dish_contrib": D[0.5].predict(drows, pred_contrib=True).tolist()}
(OUT / "parity_expected.json").write_text(json.dumps(parity).replace("NaN", "null"))

# ---- 2. demo rows vs the generated data tables ----
te = {t: pd.read_csv(ROOT / f"data/test/{t}.csv") for t in ["services", "servings", "hourly"]}
s_te = prepare_services(te["services"])
bad = 0
for s in demo["services"]:
    row = s_te[(s_te["date"] == s["date"]) & (s_te["meal"] == s["meal"])].iloc[0]
    exp = [row[c] for c in GUEST_FEATURES]
    if not np.allclose(np.array(exp, float), np.array(s["guest_x"], float), atol=1e-4):
        bad += 1
    sv = te["servings"][(te["servings"]["date"] == s["date"]) & (te["servings"]["meal"] == s["meal"])]
    if sorted(sv["dish_id"]) != sorted(d["id"] for d in s["dishes"]):
        bad += 1
    for d in s["dishes"]:
        pk = sv[sv["dish_id"] == d["id"]]["plan_kg"].iloc[0]
        if abs(pk - d["chef_plan_kg"]) > 1e-3:
            bad += 1
print("demo rows inconsistent with data/test:", bad)

# ---- 3. July report, recomputed ----
cutoff = pd.Timestamp("2026-10-11")
s_te["att_pred"] = G[0.5].predict(s_te[GUEST_FEATURES]) * s_te["present"]  # rate * present
static, vecs = dish_static(cat, "tfidf")
nb = embeddings.nearest(vecs, 5)
f = build_dish_frame(te["servings"], te["services"], te["hourly"], cat, static, nb)
f = f.merge(s_te[["date", "meal", "att_pred"]], on=["date", "meal"])
ev = f[f["date"] < cutoff].copy()
qs = [0.1, 0.5, 0.8, 0.9, 0.95]
raw = np.column_stack([D[q].predict(ev[DISH_FEATURES]) for q in qs])
qkg = np.sort(raw, axis=1) * ev["att_pred"].to_numpy()[:, None] / 100
y = ev["true_demand_kg"].to_numpy()
u = ev["dish_id"].map(lambda i: dishes[i]["kg_per_unit"]).to_numpy()
price = ev["group"].map(cfg["costs"]["try_per_kg"]).to_numpy()
refill = cfg["service"]["max_refill_share"]


def sim(plan):
    cap = plan * (1 + refill)
    extra = np.ceil(np.maximum(y - plan, 0) / u) * u
    prod = np.where(y <= plan, plan, np.minimum(cap, plan + extra))
    return prod, prod - np.minimum(y, prod), y > prod


print("coverage q:", {q: round(float(np.mean(y <= qkg[:, i])), 4) for i, q in enumerate(qs)})
jul = (ev["date"].dt.month == 7).to_numpy()
key = ev["date"].astype(str) + ev["meal"]
cp, cw, cs = sim(ev["plan_kg"].to_numpy())
print(f"July rows {jul.sum()}, services {key[jul].nunique()}")


def summary(name, prod, waste, so):
    m = jul
    svc_any = pd.Series(so[m]).groupby(key[m].to_numpy()).any().mean()
    print(f"{name}: prod {prod[m].sum():.0f} kg, waste {waste[m].sum():.1f} kg, waste TRY {(waste[m]*price[m]).sum():.0f}, "
          f"dish-service stockout {so[m].mean():.4f}, services with >=1 stockout {svc_any:.4f}, "
          f"pre-refill shortfall rows {np.mean(y[m] > plan_of[name][m]):.4f}")


plan_of = {"chef": ev["plan_kg"].to_numpy()}
summary("chef", cp, cw, cs)
for r in [0.2, 0.1, 0.05]:
    lvl = 1 - r
    rec = np.array([np.interp(lvl, qs, row) for row in qkg])
    rec = np.maximum(u * 0.5, np.ceil(rec / u * 2 - 1e-9) / 2 * u)
    plan_of[f"model {r}"] = rec
    p, w, s = sim(rec)
    summary(f"model {r}", p, w, s)
    print(f"   avoided kg {cw[jul].sum()-w[jul].sum():.1f}, avoided TRY {(cw[jul]*price[jul]).sum()-(w[jul]*price[jul]).sum():.0f},"
          f" extra production vs chef {p[jul].sum()-cp[jul].sum():.1f} kg")

# Whole open test season too
print("season chef stockout", cs.mean().round(4), "waste", cw.sum().round(1))

# ---- 4. Generator truths used to judge the explanation sentences ----
tr = pd.read_csv(ROOT / "data/train/services.csv")
trs = prepare_services(tr)
print("train rate by meal/event:\n", trs.groupby(["meal", "event"])["rate"].agg(["mean", "count"]).round(4))
print("corr(rate, child_share | meal):", trs.groupby("meal").apply(lambda g: g["rate"].corr(g["child_share"])).round(3).to_dict())
print("corr(rate, temperature | meal):", trs.groupby("meal").apply(lambda g: g["rate"].corr(g["temperature"])).round(3).to_dict())
print("corr(rate, occupancy | meal):", trs.groupby("meal").apply(lambda g: g["rate"].corr(g["occupancy"])).round(3).to_dict())
print("train mean temperature", trs["temperature"].mean().round(2), "mean child", trs["child_share"].mean().round(3))
print("mean first_meal_share by meal:", trs.groupby("meal")["first_meal_share"].mean().round(4).to_dict(),
      "overall", trs["first_meal_share"].mean().round(4))
