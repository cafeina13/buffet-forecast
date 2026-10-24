"""Audit 3 part 3: how much of the model-vs-chef result rests on the (circular) presence rule.
Scores the saved dish boosters with guest estimates that do not use the rule (in-house x meal rate). No training."""
import sys
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from olculu import embeddings
from olculu.config import load_dishes, load_synthetic
from olculu.features import DISH_FEATURES, GUEST_FEATURES, build_dish_frame, dish_static, prepare_services
cfg, cat = load_synthetic(), load_dishes(); dishes = {d["id"]: d for d in cat["dishes"]}
tr = {t: pd.read_csv(ROOT / "data/train" / f"{t}.csv") for t in ["services"]}
te = {t: pd.read_csv(ROOT / "data/test" / f"{t}.csv") for t in ["services", "servings", "hourly"]}
s_tr, s_te = prepare_services(tr["services"]), prepare_services(te["services"])
MD = ROOT / "experiments/2026-10-24-audit2-fixes/models"
g = lgb.Booster(model_file=str(MD / "guests_q0.5.txt"))
meal_rate = (s_tr["attendance"] / s_tr["in_house"]).groupby(s_tr["meal"]).mean()
s_te["g_full"] = g.predict(s_te[GUEST_FEATURES]) * s_te["present"]
s_te["g_norule"] = s_te["in_house"] * s_te["meal"].map(meal_rate)
static, vecs = dish_static(cat, "tfidf")
f = build_dish_frame(te["servings"], te["services"], te["hourly"], cat, static, embeddings.nearest(vecs, 5))
f = f.merge(s_te[["date", "meal", "g_full", "g_norule"]], on=["date", "meal"])
ev = f[f["date"] < pd.Timestamp("2026-10-11")]
DQ = [0.1, 0.5, 0.8, 0.9, 0.95]
qr = np.sort(np.column_stack([lgb.Booster(model_file=str(MD / f"dishes_q{q}.txt")).predict(ev[DISH_FEATURES]) for q in DQ]), axis=1)
y = ev["true_demand_kg"].to_numpy(); u = ev["dish_id"].map(lambda k: dishes[k]["kg_per_unit"]).to_numpy()
keys = pd.MultiIndex.from_arrays([ev["date"].to_numpy(), ev["meal"].to_numpy()])
def run(plan):
    prod = np.where(y <= plan, plan, np.minimum(plan * 1.12, plan + np.ceil(np.maximum(y - plan, 0) / u) * u))
    so = y > prod
    return round((prod - np.minimum(y, prod)).sum() / 1000, 1), round(so.mean() * 100, 1), round(pd.Series(so, index=keys).groupby(level=[0, 1]).any().mean() * 100, 1)
print("chef", run(ev["plan_kg"].to_numpy()))
for col in ["g_full", "g_norule"]:
    qkg = qr * ev[col].to_numpy()[:, None] / 100
    for r in [0.2, 0.1]:
        raw = np.array([np.interp(1 - r, DQ, row) for row in qkg])
        print(col, r, run(np.maximum(u * 0.5, np.ceil(raw / u * 2) / 2 * u)))
