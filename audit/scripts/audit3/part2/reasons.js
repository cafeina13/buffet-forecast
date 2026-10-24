// Audit 3 part 2: how well do the stated reasons (guests x own history) explain the median / recommendation,
// how often is a temperature reason suppressed, and does child share move predictions plausibly.
const path = require("path");
const ROOT = path.resolve(__dirname, "../../../..");
const M = require(path.join(ROOT, "audit/exports/2026-10-25-audit3/site/js/model.js"));
const demo = require(path.join(ROOT, "web/data/demo.json"));
const G = require(path.join(ROOT, "web/data/model-guests.json"));
const D = require(path.join(ROOT, "web/data/model-dishes.json"));
const gi = Object.fromEntries(G.features.map((f, i) => [f, i]));
const di = Object.fromEntries(D.features.map((f, i) => [f, i]));
const present = (meal, p) => Math.max(1, meal === "breakfast" ? p.in_house - p.departures_very_early
  : meal === "lunch" ? p.in_house - p.departures_early - p.excursions : p.in_house - p.arrivals_night - p.alacarte);
const ratios = [], tempStats = { shown: 0, suppressedSign: 0, suppressedSmall: 0, noHeat: 0, nearMild: 0 };
const childEff = [], gChild = [];
const knnVsHist = [];
const shapShare = {};
for (const s of demo.services) {
  const gq = M.predictQuantiles(G, s.guest_x);
  const guests = gq[1] * present(s.meal, s.presence);
  const gx2 = s.guest_x.slice(); gx2[gi.child_share] = 0.14;
  const gx3 = s.guest_x.slice(); gx3[gi.child_share] = 0.26;
  gChild.push(M.predict(G.models[1], gx3) / M.predict(G.models[1], gx2) - 1);
  for (const d of s.dishes) {
    const info = demo.dishes[d.id];
    const x = d.x;
    const med = M.predict(D.models[1], x) * guests / 100;
    if (x[di.hist_per100] !== null) ratios.push({ id: d.id, meal: s.meal, date: s.date, r: med / (guests * x[di.hist_per100] / 100) });
    const t = x[di.temperature];
    if (info.heat === 0) tempStats.noHeat++;
    else if (Math.abs(t - 27) < 3) tempStats.nearMild++;
    else {
      const m = x.slice(); m[di.temperature] = 27;
      const now = M.predict(D.models[1], x), ref = M.predict(D.models[1], m);
      const e = (now - ref) / ref;
      const exp = info.heat * Math.sign(t - 27);
      if (Math.abs(e) < 0.05) tempStats.suppressedSmall++;
      else if (Math.sign(e) !== exp) { tempStats.suppressedSign++; }
      else tempStats.shown++;
    }
    const a = x.slice(); a[di.child_share] = 0.14; const b = x.slice(); b[di.child_share] = 0.26;
    childEff.push(M.predict(D.models[1], b) / M.predict(D.models[1], a) - 1);
    // SHAP shares of the median ratio model
    const sh = M.shap(D.models[1], x, D.features.length);
    const tot = sh.phi.reduce((p, v) => p + Math.abs(v), 0);
    sh.phi.forEach((v, i) => { const k = D.features[i].startsWith("emb_") ? "emb_*" : D.features[i].startsWith("nat_") ? "nat_*" : D.features[i]; shapShare[k] = (shapShare[k] || 0) + Math.abs(v) / tot; });
  }
}
const q = (arr, p) => { const s = arr.slice().sort((a, b) => a - b); return s[Math.floor(p * (s.length - 1))]; };
const rr = ratios.map((o) => o.r);
console.log("median kg / (guests*hist/100): n", rr.length, "p10", q(rr, .1).toFixed(2), "p50", q(rr, .5).toFixed(2), "p90", q(rr, .9).toFixed(2),
  "share off by >15%:", (rr.filter((r) => Math.abs(r - 1) > 0.15).length / rr.length).toFixed(2));
console.log("extremes", ratios.sort((a, b) => a.r - b.r).slice(0, 3).map(o => `${o.id} ${o.meal} ${o.date} ${o.r.toFixed(2)}`), ratios.slice(-3).map(o => `${o.id} ${o.meal} ${o.date} ${o.r.toFixed(2)}`));
console.log("temperature reason:", tempStats);
console.log("dish child 14%->26% effect on median: p10", q(childEff, .1).toFixed(3), "p50", q(childEff, .5).toFixed(3), "p90", q(childEff, .9).toFixed(3), "neg share", (childEff.filter(v => v < 0).length / childEff.length).toFixed(2));
console.log("guest rate child 14->26 effect: min", Math.min(...gChild).toFixed(3), "max", Math.max(...gChild).toFixed(3));
const n = Object.values(shapShare).reduce((a, b) => a + b, 0);
console.log("mean |SHAP| share by feature (median dish model):", Object.entries(shapShare).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${k} ${(100 * v / n).toFixed(1)}%`).join(", "));
