# How the stock-out correction behaves on the synthetic training season (read-only).
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd
from olculu.features import censoring_corrected
for run in ["train", "test"]:
    sv = pd.read_csv(ROOT / f"data/{run}/servings.csv"); h = pd.read_csv(ROOT / f"data/{run}/hourly.csv")
    corr = censoring_corrected(sv, h)
    sold = sv["stockout_time"].notna()
    t, o, c = sv.loc[sold, "true_demand_kg"], sv.loc[sold, "observed_kg"], corr[sold]
    print(run, "rows", len(sv), "stockout rows", int(sold.sum()), f"({sold.mean():.3%})",
          "| corrected/observed median %.3f max %.3f" % ((c/o).median(), (c/o).max()),
          "| true/observed median %.3f" % (t/o).median(),
          "| corrected > true share %.2f" % (c > t).mean())
    s = pd.read_csv(ROOT / f"data/{run}/services.csv")
    print("   services", len(s), "dates", s.date.min(), s.date.max())
