"""s01 (lentil soup) at dinner, test run, before 2026-07-15: mean kg per 100 guests by definition."""
import pandas as pd
from pathlib import Path
root = Path(__file__).resolve().parents[4]
sv = pd.read_csv(root / "data/test/servings.csv"); se = pd.read_csv(root / "data/test/services.csv")
m = sv.merge(se[["date", "meal", "attendance"]], on=["date", "meal"])
m = m[(m.dish_id == "s01") & (m.meal == "dinner") & (m.date < "2026-07-15")]
for c in ["observed_kg", "true_demand_kg", "production_kg"]:
    print(c, round((m[c] / m.attendance * 100).mean(), 4))
print("n", len(m), "sold out", m.stockout_time.notna().sum())
