"""Check that the published JSON models (run in Node) match the saved LightGBM boosters on the demo rows."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import lightgbm as lgb

root = Path(__file__).resolve().parents[4]
demo = json.loads((root / "web/data/demo.json").read_text(encoding="utf-8"))
rec = root / "experiments" / demo["record"] / "models"
js = r"""
const fs=require('fs'),M=require(process.argv[1]+'/web/js/model.js');
const demo=JSON.parse(fs.readFileSync(process.argv[1]+'/web/data/demo.json','utf8'));
const out={};
for (const [k,f] of [['guests','model-guests.json'],['dishes','model-dishes.json']]) {
  const B=JSON.parse(fs.readFileSync(process.argv[1]+'/web/data/'+f,'utf8'));
  out[k]={q:B.quantiles,p:B.models.map(m=> k==='guests'? demo.services.map(s=>M.predict(m,s.guest_x)) : demo.services.flatMap(s=>s.dishes.map(d=>M.predict(m,d.x))))};
}
console.log(JSON.stringify(out));
"""
res = json.loads(subprocess.run(["node", "-e", js, str(root)], capture_output=True, text=True, check=True).stdout)
X = {"guests": np.array([[np.nan if v is None else v for v in s["guest_x"]] for s in demo["services"]], float),
     "dishes": np.array([[np.nan if v is None else v for v in d["x"]] for s in demo["services"] for d in s["dishes"]], float)}
for k in ["guests", "dishes"]:
    for q, p in zip(res[k]["q"], res[k]["p"]):
        b = lgb.Booster(model_file=str(rec / f"{k}_q{q}.txt"))
        diff = np.max(np.abs(b.predict(X[k]) - np.array(p)))
        print(k, q, "max abs diff", diff)
