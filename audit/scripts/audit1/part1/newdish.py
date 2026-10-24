# Median-model error on first-time (no own history) dish rows vs other rows, open test season. No training.
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, lightgbm as lgb
from olculu import embeddings
from olculu.config import load_dishes
from olculu.features import build_dish_frame, dish_static, prepare_services, GUEST_FEATURES, DISH_FEATURES
R = ROOT / "experiments/2026-10-24-guest-rule/models"
cat = load_dishes()
static, vecs = dish_static(cat, "tfidf")
nb = embeddings.nearest(vecs, 5)
d = {t: pd.read_csv(ROOT / f"data/test/{t}.csv") for t in ["services", "servings", "hourly"]}
s = prepare_services(d["services"])
s["att_pred"] = lgb.Booster(model_file=str(R / "guests_q0.5.txt")).predict(s[GUEST_FEATURES]) * s["present"]
f = build_dish_frame(d["servings"], d["services"], d["hourly"], cat, static, nb).merge(s[["date","meal","att_pred"]], on=["date","meal"])
f = f[f["date"] < pd.Timestamp("2026-10-11")]
pred = lgb.Booster(model_file=str(R / "dishes_q0.5.txt")).predict(f[DISH_FEATURES]) * f["att_pred"] / 100
ape = np.abs(pred - f["true_demand_kg"]) / f["true_demand_kg"]
new = f["hist_per100"].isna()
print("rows", len(f), "no-history rows", int(new.sum()))
print("median APE no-history %.3f | with history %.3f" % (ape[new].median(), ape[~new].median()))
print(f.loc[new, "dish_id"].value_counts().to_dict())
