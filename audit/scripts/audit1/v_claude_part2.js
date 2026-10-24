// Claude's verification of audit 1 part 2 findings F2, F3, F6 (read-only).
// Usage: node audit/scripts/audit1/v_claude_part2.js
const fs = require("fs");
const path = require("path");
const root = path.resolve(__dirname, "../../..");
const M = require(path.join(root, "web/js/model.js"));
const demo = JSON.parse(fs.readFileSync(path.join(root, "web/data/demo.json"), "utf8"));
const G = JSON.parse(fs.readFileSync(path.join(root, "web/data/model-guests.json"), "utf8"));
const D = JSON.parse(fs.readFileSync(path.join(root, "web/data/model-dishes.json"), "utf8"));
const yaml = fs.readFileSync(path.join(root, "config/dishes.yaml"), "utf8");
const gi = Object.fromEntries(G.features.map((f, i) => [f, i]));
const di = Object.fromEntries(D.features.map((f, i) => [f, i]));

// Temperature sensitivity per dish: dish override, else its group's.
const groupSens = {};
for (const m of yaml.matchAll(/^  (\w+): \{temp_sensitivity: (-?[\d.]+)\}/gm)) groupSens[m[1]] = Number(m[2]);
const dishSens = {};
for (const line of yaml.split("\n")) {
  const id = line.match(/id: (\w+)/);
  if (!id) continue;
  const g = line.match(/group: (\w+)/)[1];
  const own = line.match(/temp_sensitivity: (-?[\d.]+)/);
  dishSens[id[1]] = own ? Number(own[1]) : groupSens[g];
}

// F2: does knn_per100 move a new dish's forecast?
let f2 = [];
for (const s of demo.services) for (const d of s.dishes) {
  if (d.x[di.hist_per100] !== null) continue;
  const base = M.predict(D.models[1], d.x);
  const lo = d.x.slice(); lo[di.knn_per100] = 1;
  const hi = d.x.slice(); hi[di.knn_per100] = 12;
  const sh = M.shap(D.models[1], d.x, d.x.length);
  f2.push(`${s.date} ${s.meal} ${demo.dishes[d.id].name}: pred ${base.toFixed(2)}, knn=${d.x[di.knn_per100]?.toFixed(2)}, shap_knn ${sh.phi[di.knn_per100].toFixed(3)}, pred(knn=1) ${M.predict(D.models[1], lo).toFixed(2)}, pred(knn=12) ${M.predict(D.models[1], hi).toFixed(2)}`);
}
console.log("F2 new-dish rows:\n " + f2.join("\n "));

// F3: temperature SHAP sign vs generator sensitivity (only where SHAP credits temperature).
let wrong = 0, none = 0, total = 0;
const examples = [];
for (const s of demo.services) for (const d of s.dishes) {
  const sh = M.shap(D.models[1], d.x, d.x.length);
  const phi = sh.phi[di.temperature];
  if (Math.abs(phi) < 1e-9) continue;
  total++;
  const t = d.x[di.temperature], sens = dishSens[d.id];
  const truthSign = Math.sign(sens * (t - 27));
  if (sens === 0) none++;
  else if (Math.sign(phi) !== truthSign) { wrong++; if (examples.length < 6) examples.push(`${demo.dishes[d.id].name} T=${t.toFixed(0)} sens=${sens} phi=${phi.toFixed(3)}`); }
}
console.log(`F3 rows with temperature SHAP: ${total}; wrong sign vs generator: ${wrong}; dish has no temperature effect: ${none}`);
console.log(" e.g. " + examples.join(" | "));

// F6: guest SHAP for event_match vs counterfactual on/off, 15 Jul dinner.
const s = demo.services.find((x) => x.date === "2026-07-15" && x.meal === "dinner");
const p = s.presence;
const present = p.in_house - p.arrivals_night - p.alacarte;
const on = s.guest_x.slice(); on[gi.event_match] = 1;
const off = s.guest_x.slice(); off[gi.event_match] = 0;
const sh = M.shap(G.models[1], on, on.length);
console.log(`F6 15 Jul dinner: present ${present.toFixed(1)}; SHAP match ${(sh.phi[gi.event_match] * present).toFixed(1)} guests; counterfactual on-off ${((M.predict(G.models[1], on) - M.predict(G.models[1], off)) * present).toFixed(1)} guests`);
