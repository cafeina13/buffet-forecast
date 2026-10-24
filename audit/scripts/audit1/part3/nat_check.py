"""Does the saved median dish model raise English breakfast items under a Russian-heavy mix? (README claim)"""
import json
from pathlib import Path
import lightgbm as lgb, numpy as np
ROOT = Path(__file__).resolve().parents[4]
b = lgb.Booster(model_file=str(ROOT / "experiments/2026-10-24-guest-rule/models/dishes_q0.5.txt"))
F = b.feature_name()
d = json.load(open(ROOT / "web/data/demo.json", encoding="utf-8"))
nat = [F.index(f"nat_{n}") for n in ["DE", "RU", "UK", "TR", "OTHER"]]
res = {}
for s in d["services"]:
    if s["meal"] != "breakfast":
        continue
    for dish in s["dishes"]:
        x = np.array([np.nan if v is None else v for v in dish["x"]], dtype=float)
        x2 = x.copy()
        sh = x[nat].copy(); sh[1] += 0.3; sh /= sh.sum(); x2[nat] = sh
        p0, p1 = b.predict(x[None])[0], b.predict(x2[None])[0]
        res.setdefault(d["dishes"][dish["id"]]["name"], []).append(p1 / p0 - 1)
for k, v in sorted(res.items(), key=lambda kv: -np.mean(kv[1])):
    print(f"{k:28s} {np.mean(v):+.3f}")
