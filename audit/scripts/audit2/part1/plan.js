// Demo default (match-night dinner, risk 10%): recommended units, first batch ("İlk") and refills per dish.
const fs = require("fs");
const path = require("path");
const root = path.resolve(__dirname, "../../../..");
const M = require(path.join(root, "audit/exports/2026-10-24-audit2/site/js/model.js"));
const demo = JSON.parse(fs.readFileSync(path.join(root, "web/data/demo.json"), "utf8"));
const G = JSON.parse(fs.readFileSync(path.join(root, "web/data/model-guests.json"), "utf8"));
const D = JSON.parse(fs.readFileSync(path.join(root, "web/data/model-dishes.json"), "utf8"));
const gi = Object.fromEntries(G.features.map((f, i) => [f, i]));
const s = demo.services.find((s) => s.event === "match" && s.meal === "dinner");
const p = s.presence;
const present = p.in_house - p.arrivals_night - p.alacarte;
const guests = M.predictQuantiles(G, s.guest_x)[1] * present;
function interp(levels, values, level) {
  if (level <= levels[0]) return values[0];
  for (let i = 1; i < levels.length; i++) if (level <= levels[i]) { const t = (level - levels[i - 1]) / (levels[i] - levels[i - 1]); return values[i - 1] + t * (values[i] - values[i - 1]); }
  return values[values.length - 1];
}
let shareInitial = [];
for (const d of s.dishes) {
  const info = demo.dishes[d.id];
  const q = M.predictQuantiles(D, d.x).map((r) => Math.max(0, r * guests / 100));
  const rec = interp(D.quantiles, q, 0.9);
  const units = Math.max(0.5, Math.ceil(rec / info.kg_per_unit * 2 - 1e-9) / 2);
  const initial = Math.min(units, Math.max(1, Math.ceil(units * 0.4)));
  shareInitial.push(initial / units);
  console.log(info.name, "rec kg", rec.toFixed(1), "units", units, "initial", initial, "median kg", q[1].toFixed(1));
}
console.log("guests", guests.toFixed(1), "mean initial/total", (shareInitial.reduce((a, b) => a + b) / shareInitial.length).toFixed(2));
