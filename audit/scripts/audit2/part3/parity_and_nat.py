"""Audit 2 part 3: JS parity replica (writes only into this folder) and a nationality-mix probe."""
import json, subprocess, sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from olculu.features import DISH_FEATURES
M = ROOT / "experiments/2026-10-24-audit1-fixes/models"
cases = []
for prefix, rows, web in [("dishes", "dish_rows_sample.csv", "model-dishes.json"), ("guests", "guest_rows_sample.csv", "model-guests.json")]:
    X = pd.read_csv(M / rows)
    b = json.loads((ROOT / "web/data" / web).read_text(encoding="utf-8"))
    print(web, "features match sample:", list(X.columns) == b["features"], "quantiles", b["quantiles"])
    for i, q in enumerate(b["quantiles"]):
        bo = lgb.Booster(model_file=str(M / f"{prefix}_q{q}.txt"))
        cases.append({"web": web, "qi": i, "rows": X.to_numpy().tolist(), "pred": bo.predict(X).tolist()})
(HERE / "cases.json").write_text(json.dumps(cases).replace("NaN", "null"), encoding="utf-8")
js = r"""
const fs=require('fs');const [mj,cf,wd]=process.argv.slice(-3);const M=require(mj);
const cs=JSON.parse(fs.readFileSync(cf,'utf8'));let w=0,n=0;
for(const c of cs){const m=JSON.parse(fs.readFileSync(wd+'/'+c.web,'utf8')).models[c.qi];
c.rows.forEach((x,i)=>{w=Math.max(w,Math.abs(M.predict(m,x)-c.pred[i]));n++;});}
console.log('rows',n,'worst abs diff',w);"""
print(subprocess.run(["node", "-e", js, str(ROOT / "web/js/model.js"), str(HERE / "cases.json"), str(ROOT / "web/data")], capture_output=True, text=True).stdout)

# nationality probe on demo breakfasts
d = json.loads((ROOT / "web/data/demo.json").read_text(encoding="utf-8"))
b5 = lgb.Booster(model_file=str(M / "dishes_q0.5.txt"))
nat = [f"nat_{n}" for n in ["DE", "RU", "UK", "TR", "OTHER"]]
idx = [DISH_FEATURES.index(c) for c in nat]
res = {}
for s in d["services"]:
    if s["meal"] != "breakfast":
        continue
    for dish in s["dishes"]:
        x = np.array([np.nan if v is None else v for v in dish["x"]], dtype=float)
        x2 = x.copy(); x2[idx] = [0.15, 0.55, 0.07, 0.15, 0.08]  # Russian-heavy
        p, p2 = b5.predict(x[None])[0], b5.predict(x2[None])[0]
        res.setdefault(d["dishes"][dish["id"]]["name"], []).append(p2 / p - 1)
for k, v in res.items():
    print(f"{k:28s} RU-heavy change {np.mean(v)*100:+.1f}%")
