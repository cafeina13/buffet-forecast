"""Claude's check of audit 3 part 2 F1/F2: realised shortfall and waste vs chef, month by month (read-only)."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from olculu import embeddings
from olculu.config import load_dishes, load_synthetic
from olculu.features import DISH_FEATURES, build_dish_frame, dish_static, prepare_services
from olculu.models import DISH_QUANTILES, interpolate_quantile, predict_guests
from olculu.newsvendor import round_up_half_unit, simulate
from olculu.train import blind_cutoff, load_run

rec = "experiments/2026-10-24-audit2-fixes/models"
cfg, cat = load_synthetic(), load_dishes()
unit = {d["id"]: d["kg_per_unit"] for d in cat["dishes"]}
test = load_run("test")
s = prepare_services(test["services"])


class W:
    def __init__(self, b): self.b = b
    def predict(self, X): return self.b.predict(X)


gm = {q: W(lgb.Booster(model_file=f"{rec}/guests_q{q}.txt")) for q in [0.1, 0.5, 0.9]}
s["att_pred"] = predict_guests(gm, s)[0][:, 1]
static, vecs = dish_static(cat, "tfidf")
f = build_dish_frame(test["servings"], test["services"], test["hourly"], cat, static, embeddings.nearest(vecs, 5))
f = f.merge(s[["date", "meal", "att_pred"]], on=["date", "meal"])
ev = f[f["date"] < blind_cutoff(cfg)].copy()
dm = [lgb.Booster(model_file=f"{rec}/dishes_q{q}.txt") for q in DISH_QUANTILES]
q = np.sort(np.column_stack([b.predict(ev[DISH_FEATURES]) for b in dm]), axis=1) * ev["att_pred"].to_numpy()[:, None] / 100
y, u = ev["true_demand_kg"].to_numpy(), ev["dish_id"].map(unit).to_numpy()
refill = cfg["service"]["max_refill_share"]
month = ev["date"].dt.month.to_numpy()
chef = simulate(ev["plan_kg"].to_numpy(), y, u, refill)
for r in [0.2, 0.1]:
    recq = round_up_half_unit(interpolate_quantile(q, DISH_QUANTILES, 1 - r), u)
    sim = simulate(recq, y, u, refill)
    rows = []
    for m in sorted(set(month)):
        k = month == m
        rows.append(f"{m}: waste {100 * (sim['waste'][k].sum() / chef['waste'][k].sum() - 1):+.1f}% short {100 * np.mean(y[k] > recq[k]):.1f}%")
    print(f"risk {r}: " + " | ".join(rows))
