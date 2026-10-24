// Temperature reasons hidden by the filter; guest-model event effects; initial-batch shares.
const M = require("../../../exports/2026-10-24-audit2/site/js/model.js");
const fs = require("fs");
const j = (f) => JSON.parse(fs.readFileSync(__dirname + "/../../../../web/data/" + f, "utf8"));
const demo = j("demo.json"), G = j("model-guests.json"), D = j("model-dishes.json");
const gi = Object.fromEntries(G.features.map((f, i) => [f, i]));
const di = Object.fromEntries(D.features.map((f, i) => [f, i]));
let shown = 0, hiddenDir = 0, hiddenHeat0 = 0, small = 0, total = 0; const ex = [];
for (const s of demo.services) for (const d of s.dishes) {
  const info = demo.dishes[d.id]; const x = d.x.slice(); const t = x[di.temperature];
  if (Math.abs(t - 27) < 3) continue;
  const m = x.slice(); m[di.temperature] = 27;
  const now = M.predict(D.models[1], x), ref = M.predict(D.models[1], m);
  const e = (now - ref) / ref; total++;
  if (Math.abs(e) < 0.05) { small++; continue; }
  const exp = info.heat * Math.sign(t - 27);
  if (info.heat === 0) { hiddenHeat0++; ex.push([s.date, s.meal, info.name, "heat0", (e * 100).toFixed(1)]); }
  else if (Math.sign(e) !== exp) { hiddenDir++; ex.push([s.date, s.meal, info.name, "wrongdir", (e * 100).toFixed(1)]); }
  else shown++;
}
console.log({ total, small, shown, hiddenDir, hiddenHeat0 }); console.log(ex.slice(0, 15).map((r) => r.join(" ")).join("\n"));
// Guest-model event effects (median rate), each service with event flag toggled
for (const [flag, meal] of [["event_conference", "lunch"], ["event_match", "dinner"], ["event_themed", "dinner"]]) {
  const eff = demo.services.filter((s) => s.meal === meal).map((s) => {
    const on = s.guest_x.slice(); on[gi[flag]] = 1; for (const f of ["event_match", "event_themed", "event_conference"]) if (f !== flag) on[gi[f]] = 0;
    const off = on.slice(); off[gi[flag]] = 0;
    return M.predict(G.models[1], on) / M.predict(G.models[1], off) - 1;
  });
  console.log(flag, eff.map((e) => (e * 100).toFixed(1)).join(" "));
}
// Nationality: move 10 points from DE to RU, effect on guest rate and dish ratio
const s0 = demo.services[2];
const g2 = s0.guest_x.slice(); g2[gi.nat_DE] -= 0.1; g2[gi.nat_RU] += 0.1;
console.log("guest nat shift", (M.predict(G.models[1], g2) / M.predict(G.models[1], s0.guest_x) - 1) * 100);
// Initial-batch share in demo week (risk 10%) -- uses chef-plan-independent recUnits approximated from harness output
