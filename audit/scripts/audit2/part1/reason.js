// Re-implements demo.js compute()/explain() for default state on all services; prints reasons.
const fs = require("fs");
const path = require("path");
const root = path.resolve(__dirname, "../../../..");
const M = require(path.join(root, "audit/exports/2026-10-24-audit2/site/js/model.js"));
const demo = JSON.parse(fs.readFileSync(path.join(root, "web/data/demo.json"), "utf8"));
const G = JSON.parse(fs.readFileSync(path.join(root, "web/data/model-guests.json"), "utf8"));
const D = JSON.parse(fs.readFileSync(path.join(root, "web/data/model-dishes.json"), "utf8"));
const gIdx = Object.fromEntries(G.features.map((f, i) => [f, i]));
const dIdx = Object.fromEntries(D.features.map((f, i) => [f, i]));
const MILD = 27;
const filter = process.argv[2] || "";
function present(meal, p) {
  const v = meal === "breakfast" ? p.in_house - p.departures_very_early
    : meal === "lunch" ? p.in_house - p.departures_early - p.excursions
      : p.in_house - p.arrivals_night - p.alacarte;
  return Math.max(1, v);
}
for (const s of demo.services) {
  const p = { ...s.presence };
  p.in_house = Math.min(p.in_house, demo.support.capacity_guests);
  const pr = present(s.meal, p);
  const gx = s.guest_x.slice();
  if (s.meal === "dinner") { gx[gIdx.event_match] = s.event === "match" ? 1 : 0; gx[gIdx.event_themed] = s.event === "themed" ? 1 : 0; }
  const rates = M.predictQuantiles(G, gx);
  const guests = Math.max(0, rates[1] * pr);
  const line = `${s.date} ${s.meal} ${s.event} present=${pr.toFixed(1)} rate=${rates[1].toFixed(4)} guests=${guests.toFixed(1)} (${Math.round(guests)}) temp=${gx[gIdx.temperature].toFixed(2)}`;
  if (filter && !line.includes(filter)) continue;
  console.log(line);
  for (const d of s.dishes) {
    const info = demo.dishes[d.id];
    const x = d.x.slice();
    if (s.meal === "dinner") { x[dIdx.event_match] = s.event === "match" ? 1 : 0; x[dIdx.event_themed] = s.event === "themed" ? 1 : 0; }
    const t = x[dIdx.temperature];
    let temp = "";
    if (info.heat !== 0 && Math.abs(t - MILD) >= 3) {
      const mild = x.slice(); mild[dIdx.temperature] = MILD;
      const now = M.predict(D.models[1], x), ref = M.predict(D.models[1], mild);
      const eff = ref > 0 ? (now - ref) / ref : 0;
      const shown = Math.abs(eff) >= 0.05 && Math.sign(eff) === info.heat * Math.sign(t - MILD);
      temp = `heat=${info.heat} eff=${(eff * 100).toFixed(2)}% shown=${shown}`;
    }
    console.log(`   ${d.id} ${info.name} hist_per100=${x[dIdx.hist_per100]} n_similar=${x[dIdx.n_similar]} ${temp}`);
  }
}
