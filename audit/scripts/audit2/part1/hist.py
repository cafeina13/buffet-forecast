"""Recompute mercimek (s01) dinner history before 2026-07-15 from data/test, observed vs corrected."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from olculu.features import censoring_corrected  # noqa: E402

d = ROOT / "data" / "test"
sv = pd.read_csv(d / "servings.csv")
se = pd.read_csv(d / "services.csv")
hr = pd.read_csv(d / "hourly.csv")
sv["corr"] = censoring_corrected(sv, hr).to_numpy()
m = sv.merge(se[["date", "meal", "attendance", "temperature", "event"]], on=["date", "meal"])
x = m[(m.dish_id == "s01") & (m.meal == "dinner")].copy()
x["date"] = pd.to_datetime(x["date"])
before = x[x.date < "2026-07-15"]
print("season start", se.date.min(), "n prior servings", len(before))
print("mean corrected per100", (before["corr"] / before.attendance * 100).mean())
print("mean observed per100", (before["observed_kg"] / before.attendance * 100).mean())
print("mean true per100", (before["true_demand_kg"] / before.attendance * 100).mean())
print("stockouts among prior", before.stockout_time.notna().sum())
row = x[x.date == "2026-07-15"].iloc[0]
print("07-15 attendance", row.attendance, "temp", row.temperature, "event", row.event,
      "true kg", row.true_demand_kg, "plan", row.plan_kg)
# whole-test stats
print("all stockout share dish-servings", sv.stockout_time.notna().mean())
