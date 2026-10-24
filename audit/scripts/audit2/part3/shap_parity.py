"""Audit 2 part 3: model.js TreeSHAP vs LightGBM pred_contrib on 60 sample rows per model (writes only here)."""
import json, subprocess
from pathlib import Path
import lightgbm as lgb
import pandas as pd
ROOT = Path(__file__).resolve().parents[4]; HERE = Path(__file__).resolve().parent
M = ROOT / "experiments/2026-10-24-audit1-fixes/models"
cases = []
for prefix, rows, web in [("dishes", "dish_rows_sample.csv", "model-dishes.json"), ("guests", "guest_rows_sample.csv", "model-guests.json")]:
    X = pd.read_csv(M / rows).head(60)
    qs = json.loads((ROOT / "web/data" / web).read_text(encoding="utf-8"))["quantiles"]
    for i, q in enumerate(qs):
        bo = lgb.Booster(model_file=str(M / f"{prefix}_q{q}.txt"))
        cases.append({"web": web, "qi": i, "rows": X.to_numpy().tolist(), "contrib": bo.predict(X, pred_contrib=True).tolist()})
(HERE / "shap_cases.json").write_text(json.dumps(cases).replace("NaN", "null"), encoding="utf-8")
js = r"""
const fs=require('fs');const [mj,cf,wd]=process.argv.slice(-3);const M=require(mj);
const cs=JSON.parse(fs.readFileSync(cf,'utf8'));let w=0;
for(const c of cs){const m=JSON.parse(fs.readFileSync(wd+'/'+c.web,'utf8')).models[c.qi];
c.rows.forEach((x,i)=>{const s=M.shap(m,x,x.length);const e=c.contrib[i];
s.phi.forEach((v,k)=>{w=Math.max(w,Math.abs(v-e[k]));});w=Math.max(w,Math.abs(s.base-e[e.length-1]));});}
console.log('worst SHAP diff',w);"""
print(subprocess.run(["node", "-e", js, str(ROOT / "web/js/model.js"), str(HERE / "shap_cases.json"), str(ROOT / "web/data")], capture_output=True, text=True))
