"""Audit 3 part 3: README claims about demo controls (nationality, child share, temperature), on demo.json rows."""
import json, sys
from pathlib import Path
import lightgbm as lgb, numpy as np
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from olculu.features import DISH_FEATURES
demo = json.loads((ROOT / "web/data/demo.json").read_text(encoding="utf-8"))
b = lgb.Booster(model_file=str(ROOT / "experiments/2026-10-24-audit2-fixes/models/dishes_q0.5.txt"))
I = {f: i for i, f in enumerate(DISH_FEATURES)}
rows, ids, meals = [], [], []
for s in demo["services"]:
    for d in s["dishes"]:
        rows.append([np.nan if v is None else v for v in d["x"]]); ids.append(d["id"]); meals.append(s["meal"])
X = np.array(rows, dtype=float); ids = np.array(ids); meals = np.array(meals)
base = b.predict(X)
def nat(mix):
    Z = X.copy()
    for k, v in mix.items(): Z[:, I["nat_" + k]] = v
    return b.predict(Z) / base - 1
ru = nat({"DE": 0.1, "RU": 0.6, "UK": 0.1, "TR": 0.1, "OTHER": 0.1})
uk = nat({"DE": 0.1, "RU": 0.1, "UK": 0.6, "TR": 0.1, "OTHER": 0.1})
cat = {d["id"]: d for d in __import__("yaml").safe_load(open(ROOT / "config/dishes.yaml", encoding="utf-8"))["dishes"]}
for did in ["b08", "b09", "b11", "s04"]:
    m = ids == did
    if m.any(): print(did, cat[did]["name"], cat[did].get("affinity"), "RU-heavy %+.1f%%" % (100 * ru[m].mean()), "UK-heavy %+.1f%%" % (100 * uk[m].mean()))
# child share 0.15 -> 0.26
lo, hi = X.copy(), X.copy(); lo[:, I["child_share"]] = 0.15; hi[:, I["child_share"]] = 0.26
ch = b.predict(hi) / b.predict(lo) - 1
print("child 15->26%%: mean %+.2f%%, share of rows up %.2f, generator expects +%.1f%%" % (100 * ch.mean(), (ch > 0).mean(), 100 * (0.1 * 0.26 / 0.2 - 0.1 * 0.15 / 0.2) / (1 + 0.1 * 0.15 / 0.2)))
