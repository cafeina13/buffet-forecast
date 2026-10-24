// Split counts and mean |SHAP| per feature for the published median dish model, on the demo dish rows.
const fs = require("fs"), path = require("path");
const root = path.resolve(__dirname, "../../../..");
const M = require(path.join(root, "web/js/model.js"));
const demo = JSON.parse(fs.readFileSync(path.join(root, "web/data/demo.json"), "utf8"));
const D = JSON.parse(fs.readFileSync(path.join(root, "web/data/model-dishes.json"), "utf8"));
const med = D.models[D.quantiles.indexOf(0.5)];
const nF = D.features.length, splits = new Array(nF).fill(0), abs = new Array(nF).fill(0);
for (const t of med.trees) t.f.forEach((f) => { if (f >= 0) splits[f]++; });
let n = 0, totPred = 0;
for (const s of demo.services) for (const d of s.dishes) {
  const { phi } = M.shap(med, d.x, nF);
  phi.forEach((p, i) => (abs[i] += Math.abs(p)));
  totPred += M.predict(med, d.x); n++;
}
const tot = abs.reduce((a, b) => a + b, 0);
console.log("rows", n, "mean median prediction (kg/100 guests)", (totPred / n).toFixed(3));
D.features.map((f, i) => [f, splits[i], abs[i] / n, abs[i] / tot]).sort((a, b) => b[2] - a[2])
  .forEach(([f, s, a, sh]) => console.log(f.padEnd(18), String(s).padStart(5), a.toFixed(4), (100 * sh).toFixed(2) + "%"));
