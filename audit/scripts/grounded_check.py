"""Sanity check of the semi-synthetic seasons against the fully synthetic ones (read-only)."""
import json

import pandas as pd

for run in ["train", "test"]:
    s = pd.read_csv(f"data/{run}/services.csv", parse_dates=["date"])
    v = pd.read_csv(f"data/{run}/servings.csv")
    meta = json.load(open(f"data/{run}/meta.json", encoding="utf-8"))
    d = s[s["meal"] == "dinner"]
    print(f"== {run} {meta['start']}..{meta['end']}, scale {meta.get('scale', 1):.2f}, match nights {len(meta.get('match_nights', []))}")
    by_m = d.groupby(d["date"].dt.month).agg(occ=("occupancy", "mean"), in_house=("in_house", "mean"),
                                             child=("child_share", "mean"), temp=("temperature", "mean"),
                                             arr=("arrivals_day", "mean"), match=("event", lambda e: (e == "match").sum()))
    print(by_m.round(2).to_string())
    v["so"] = v["stockout_time"].notna()
    print("stock-out rate", round(v["so"].mean(), 3), "waste share", round(v["leftover_kg"].sum() / v["production_kg"].sum(), 3))
    print("arrivals/in_house mean", round(((s["arrivals_day"] + s["arrivals_night"]) / s["in_house"]).mean(), 3))
