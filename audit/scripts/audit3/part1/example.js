// Reproduce the home-page rationale example (match evening, lentil soup) with the demo's logic.
const fs = require("fs");
const path = require("path");
const root = path.resolve(__dirname, "../../../..");
const M = require(path.join(root, "web/js/model.js"));
const demo = JSON.parse(fs.readFileSync(path.join(root, "web/data/demo.json"), "utf8"));
const G = JSON.parse(fs.readFileSync(path.join(root, "web/data/model-guests.json"), "utf8"));
const D = JSON.parse(fs.readFileSync(path.join(root, "web/data/model-dishes.json"), "utf8"));
const gIdx = Object.fromEntries(G.features.map((f, i) => [f, i]));
const dIdx = Object.fromEntries(D.features.map((f, i) => [f, i]));
function present(meal, p) {
  return Math.max(1, meal === "breakfast" ? p.in_house - p.departures_very_early
    : meal === "lunch" ? p.in_house - p.departures_early - p.excursions : p.in_house - p.arrivals_night - p.alacarte);
}
const ids = Object.entries(demo.dishes).filter(([k, d]) => /mercimek/i.test(d.name));
console.log("lentil dishes:", ids.map(([k, d]) => [k, d.name, d.heat, d.group]));
console.log("record", demo.record, "support", demo.support);
for (const s of demo.services) {
  const pres = present(s.meal, s.presence);
  const rates = M.predictQuantiles(G, s.guest_x);
  const guests = rates[1] * pres;
  let line = `${s.date} ${s.meal} ${s.event} present=${pres.toFixed(1)} rate=${rates[1].toFixed(4)} guests=${guests.toFixed(1)} temp=${s.guest_x[gIdx.temperature].toFixed(2)}`;
  for (const d of s.dishes) {
    if (!/mercimek/i.test(demo.dishes[d.id].name)) continue;
    const x = d.x;
    const t = x[dIdx.temperature];
    const mild = x.slice(); mild[dIdx.temperature] = 27;
    const now = M.predict(D.models[1], x), ref = M.predict(D.models[1], mild);
    line += ` | ${d.id} hist=${x[dIdx.hist_per100]} knn=${x[dIdx.knn_per100]} dtemp=${t} eff=${((now - ref) / ref * 100).toFixed(2)}%`;
  }
  console.log(line);
}
