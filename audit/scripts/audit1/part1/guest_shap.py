# Mean |SHAP| of the published guest median model on the open test season (read-only).
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
import lightgbm as lgb, numpy as np, pandas as pd
from olculu.features import prepare_services, GUEST_FEATURES
s = prepare_services(pd.read_csv(ROOT / "data/test/services.csv"))
s = s[s["date"] < pd.Timestamp("2026-10-11")]
rec = ROOT / "experiments/2026-10-24-guest-rule/models"
b = lgb.Booster(model_file=str(rec / "guests_q0.5.txt"))
c = b.predict(s[GUEST_FEATURES], pred_contrib=True)
people = np.abs(c[:, :-1]) * s["present"].to_numpy()[:, None]
print("mean |SHAP| in guests (rate*present):")
for f, v in sorted(zip(GUEST_FEATURES, people.mean(0)), key=lambda t: -t[1]):
    print(f"  {f:18s} {v:7.2f}")
# Does the generator make attendance rate depend on temperature? correlation of rate with temp within meal & no event
for meal in ["breakfast","lunch","dinner"]:
    m = s[(s.meal==meal)&(s.event=="none")]
    print(meal, "corr(rate,temp)=%.3f corr(rate,occ)=%.3f corr(rate,child)=%.3f" % (m.rate.corr(m.temperature), m.rate.corr(m.occupancy), m.rate.corr(m.child_share)))
d = s[s.meal=="dinner"]
print("dinner mean rate by event:", d.groupby("event").rate.mean().round(3).to_dict())
# web json model equals experiment model?
g = json.load(open(ROOT/"web/data/model-guests.json", encoding="utf-8"))
print("web trees", [len(m["trees"]) for m in g["models"]], "exp trees", b.num_trees())
