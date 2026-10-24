// Audit 3 part 3: published model.js + web/data model JSON vs LightGBM on every open-test row.
// Usage: node parity_all.js <parity_all.json> <repo root>
const fs = require("fs");
const [casesFile, root] = process.argv.slice(2);
const M = require(root + "/audit/exports/2026-10-25-audit3/site/js/model.js");
const cases = JSON.parse(fs.readFileSync(casesFile, "utf8"));
const rows = {};
cases.filter((c) => c.rows).forEach((c) => (rows[c.web] = c.rows));
const out = {};
for (const c of cases.filter((c) => c.pred)) {
  const b = JSON.parse(fs.readFileSync(root + "/web/data/" + c.web, "utf8"));
  let worst = 0;
  rows[c.web].forEach((x, i) => { worst = Math.max(worst, Math.abs(M.predict(b.models[c.qi], x) - c.pred[i])); });
  out[c.web + " q" + b.quantiles[c.qi]] = { rows: rows[c.web].length, worst };
}
// SHAP additivity on a sample
const b = JSON.parse(fs.readFileSync(root + "/web/data/model-dishes.json", "utf8"));
let worstSum = 0;
rows["model-dishes.json"].slice(0, 500).forEach((x) => {
  const s = M.shap(b.models[1], x, x.length);
  worstSum = Math.max(worstSum, Math.abs(s.base + s.phi.reduce((a, v) => a + v, 0) - M.predict(b.models[1], x)));
});
out.shap_additivity_worst = worstSum;
console.log(JSON.stringify(out, null, 1));
