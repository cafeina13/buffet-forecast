"""Compare browser predictions (parity_js.json) with the saved LightGBM boosters of the record."""
import json
from pathlib import Path
import lightgbm as lgb
import numpy as np
H = Path(__file__).resolve().parent
ROOT = H.parents[3]
M = ROOT / "experiments" / "2026-10-24-audit1-fixes" / "models"
demo = json.loads((ROOT / "web/data/demo.json").read_text(encoding="utf-8"))
js = json.loads((H / "parity_js.json").read_text())
g = [lgb.Booster(model_file=str(M / f"guests_q{q}.txt")) for q in [0.1, 0.5, 0.9]]
d = [lgb.Booster(model_file=str(M / f"dishes_q{q}.txt")) for q in [0.1, 0.5, 0.8, 0.9, 0.95]]
mg = md = 0
for s, o in zip(demo["services"], js):
    x = np.array([[np.nan if v is None else v for v in s["guest_x"]]], dtype=float)
    mg = max(mg, np.abs(np.sort([b.predict(x)[0] for b in g]) - o["rate"]).max())
    X = np.array([[np.nan if v is None else v for v in dd["x"]] for dd in s["dishes"]], dtype=float)
    p = np.sort(np.column_stack([b.predict(X) for b in d]), axis=1)
    md = max(md, np.abs(p - np.array([dd["q"] for dd in o["dishes"]])).max())
print("max abs diff guest rate:", mg, " dish ratio:", md)
