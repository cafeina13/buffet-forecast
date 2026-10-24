"""Claude's verification of audit 1 part 3 finding F1: stock-outs per dish vs per service (read-only)."""
import lightgbm as lgb
import numpy as np
import pandas as pd

from olculu import embeddings
from olculu.config import load_dishes, load_synthetic
from olculu.features import DISH_FEATURES, build_dish_frame, dish_static, prepare_services
from olculu.models import DISH_QUANTILES, interpolate_quantile, predict_guests
from olculu.newsvendor import round_up_half_unit, simulate
from olculu.train import blind_cutoff, load_run

rec = "experiments/2026-10-24-guest-rule/models"
cfg, cat = load_synthetic(), load_dishes()
unit = {d["id"]: d["kg_per_unit"] for d in cat["dishes"]}
test = load_run("test")
s = prepare_services(test["services"])
gm = {q: lgb.Booster(model_file=f"{rec}/guests_q{q}.txt") for q in [0.1, 0.5, 0.9]}


class W:  # minimal wrapper so predict_guests can call .predict
    def __init__(self, b): self.b = b
    def predict(self, X): return self.b.predict(X)


g, _ = predict_guests({q: W(b) for q, b in gm.items()}, s)
s["att_pred"] = g[:, 1]
static, vecs = dish_static(cat, "tfidf")
f = build_dish_frame(test["servings"], test["services"], test["hourly"], cat, static, embeddings.nearest(vecs, 5))
f = f.merge(s[["date", "meal", "att_pred"]], on=["date", "meal"])
ev = f[f["date"] < blind_cutoff(cfg)].copy()
dm = [lgb.Booster(model_file=f"{rec}/dishes_q{q}.txt") for q in DISH_QUANTILES]
q = np.sort(np.column_stack([b.predict(ev[DISH_FEATURES]) for b in dm]), axis=1) * ev["att_pred"].to_numpy()[:, None] / 100
y, u = ev["true_demand_kg"].to_numpy(), ev["dish_id"].map(unit).to_numpy()
refill = cfg["service"]["max_refill_share"]

def report(name, sim):
    ev["so"] = sim["stockout"]
    per_service = ev.groupby(["date", "meal"])["so"].any().mean()
    print(f"{name:10s} waste {sim['waste'].sum()/1000:6.1f} t | dishes run out {sim['stockout'].mean():.1%} | services with >=1 dish out {per_service:.1%}")

report("chef", simulate(ev["plan_kg"].to_numpy(), y, u, refill))
for r in [0.2, 0.1, 0.05]:
    report(f"model {int(r*100)}%", simulate(round_up_half_unit(interpolate_quantile(q, DISH_QUANTILES, 1 - r), u), y, u, refill))
print("dishes per service (median):", ev.groupby(["date", "meal"]).size().median())
