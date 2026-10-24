"""Own parity check (writes only into this folder): LightGBM (saved guest-rule boosters) vs web/js/model.js."""
import json, subprocess, sys
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from olculu.features import prepare_services, GUEST_FEATURES
REC = ROOT / "experiments/2026-10-24-guest-rule/models"
cases = []
dish_rows = pd.read_csv(REC / "dish_rows_sample.csv")
s = prepare_services(pd.read_csv(ROOT / "data/test/services.csv"))
guest_rows = s[GUEST_FEATURES].iloc[::3].reset_index(drop=True)
for prefix, X, web, qs in [("dishes", dish_rows, "model-dishes.json", [0.1, 0.5, 0.8, 0.9, 0.95]),
                           ("guests", guest_rows, "model-guests.json", [0.1, 0.5, 0.9])]:
    for qi, q in enumerate(qs):
        b = lgb.Booster(model_file=str(REC / f"{prefix}_q{q}.txt"))
        c = {"web": web, "qi": qi, "rows": X.to_numpy().tolist(), "pred": b.predict(X).tolist()}
        if q == 0.5:
            c["contrib"] = b.predict(X, pred_contrib=True).tolist()
        cases.append(c)
(HERE / "parity_cases.json").write_text(json.dumps(cases).replace("NaN", "null"), encoding="utf-8")
js = r"""
const fs=require('fs');const [mj,cf,wd]=process.argv.slice(-3);const M=require(mj);
const cases=JSON.parse(fs.readFileSync(cf,'utf8'));const res=[];
for(const c of cases){const b=JSON.parse(fs.readFileSync(wd+'/'+c.web,'utf8'));const m=b.models[c.qi];
let wp=0,ws=0;c.rows.forEach((x,i)=>{wp=Math.max(wp,Math.abs(M.predict(m,x)-c.pred[i]));
if(c.contrib){const s=M.shap(m,x,x.length);s.phi.forEach((v,k)=>{ws=Math.max(ws,Math.abs(v-c.contrib[i][k]))});ws=Math.max(ws,Math.abs(s.base-c.contrib[i][c.contrib[i].length-1]));}});
res.push({web:c.web,qi:c.qi,quantile:b.quantiles[c.qi],n:c.rows.length,worstPred:wp,worstShap:c.contrib?ws:null});}
console.log(JSON.stringify(res));"""
o = subprocess.run(["node", "-e", js, str(ROOT / "web/js/model.js"), str(HERE / "parity_cases.json"), str(ROOT / "web/data")],
                   capture_output=True, text=True)
print(o.stdout, o.stderr)
