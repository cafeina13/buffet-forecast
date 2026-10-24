"""Claude's verification of audit 1 part 1 findings (read-only)."""
import json

import numpy as np
import pandas as pd
import yaml

from olculu.features import DISH_FEATURES, censoring_corrected, prepare_services

s = prepare_services(pd.read_csv("data/train/services.csv"))
v = pd.read_csv("data/train/servings.csv")
h = pd.read_csv("data/train/hourly.csv")

# F1: which embedding backend produced the published record?
m = json.load(open("experiments/2026-10-24-guest-rule/metrics.json", encoding="utf-8"))
print("F1 embeddings backend:", m["embeddings"])

# F2: is occupancy a dish-model feature?
print("F2 occupancy in DISH_FEATURES:", "occupancy" in DISH_FEATURES)

# F3: parameter types in the config, and realised kg per guest by meal.
raw = yaml.safe_load(open("config/synthetic.yaml", encoding="utf-8"))
types = []
def walk(n):
    if isinstance(n, dict) and "value" in n and "source" in n:
        types.append(n.get("type"))
    elif isinstance(n, dict):
        for x in n.values():
            walk(x)
walk(raw)
print("F3 parameter types:", pd.Series(types).value_counts().to_dict())
kg = v.groupby(["date", "meal"])["true_demand_kg"].sum()
att = pd.read_csv("data/train/services.csv").set_index(["date", "meal"])["attendance"]
print("F3 realised kg per guest by meal:", (kg / att).groupby(level=1).mean().round(3).to_dict())

# F5: does the attendance rate depend on temperature (no events)?
d = s[(s["event"] == "none")]
for meal in ["breakfast", "lunch", "dinner"]:
    x = d[d["meal"] == meal]
    print(f"F5 corr(rate, temperature) {meal}: {np.corrcoef(x['rate'], x['temperature'])[0,1]:+.3f}")

# F6: censoring correction vs true demand on sold-out rows.
corr = censoring_corrected(v, h)
sold = v["stockout_time"].notna()
print("F6 sold-out rows:", int(sold.sum()), f"({sold.mean():.1%})")
print("F6 median factor corrected/observed:", round(float((corr[sold] / v.loc[sold, 'observed_kg']).median()), 3),
      "true/observed:", round(float((v.loc[sold, 'true_demand_kg'] / v.loc[sold, 'observed_kg']).median()), 3))
print("F6 corrected > true on any sold-out row:", bool((corr[sold] > v.loc[sold, "true_demand_kg"] + 1e-9).any()))
print("F6 censoring metrics:", m["censoring"])

# F8: nominal risk vs simulated stock-out rate.
print("F8 risk curve:", [(c["risk"], round(c["stockout_rate"], 3)) for c in m["risk_curve"] if c["risk"] in (0.2, 0.1, 0.05)])
print("Report July:", [(l["risk"], round(l["stockout_rate"], 3)) for l in m["monthly_report"]["levels"]], "chef", round(m["monthly_report"]["chef"]["stockout_rate"], 3))
