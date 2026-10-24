// Compares model.js (published site) with LightGBM boosters on every demo row, all quantiles, plus SHAP.
const fs=require('fs'),path=require('path');
const ROOT=path.resolve(__dirname,'../../../..');
const M=require(path.join(ROOT,'audit/exports/2026-10-24-audit1/site/js/model.js'));
const E=JSON.parse(fs.readFileSync(path.join(__dirname,'parity_expected.json'),'utf8'));
const demo=JSON.parse(fs.readFileSync(path.join(ROOT,'web/data/demo.json'),'utf8'));
const G=JSON.parse(fs.readFileSync(path.join(ROOT,'web/data/model-guests.json'),'utf8'));
const D=JSON.parse(fs.readFileSync(path.join(ROOT,'web/data/model-dishes.json'),'utf8'));
const gx=demo.services.map(s=>s.guest_x), dx=demo.services.flatMap(s=>s.dishes.map(d=>d.x));
let w={};
const upd=(k,v)=>{w[k]=Math.max(w[k]||0,v)};
G.quantiles.forEach((q,i)=>gx.forEach((x,j)=>upd('guest_pred',Math.abs(M.predict(G.models[i],x)-E.guest[String(q)][j]))));
D.quantiles.forEach((q,i)=>dx.forEach((x,j)=>upd('dish_pred',Math.abs(M.predict(D.models[i],x)-E.dish[String(q)][j]))));
gx.forEach((x,j)=>{const s=M.shap(G.models[1],x,x.length);s.phi.forEach((v,k)=>upd('guest_shap',Math.abs(v-E.guest_contrib[j][k])));});
dx.forEach((x,j)=>{const s=M.shap(D.models[1],x,x.length);s.phi.forEach((v,k)=>upd('dish_shap',Math.abs(v-E.dish_contrib[j][k])));});
console.log(w, 'quantiles', G.quantiles, D.quantiles, 'rows', gx.length, dx.length);
