// Dump JS predictions for every demo service (guest and dish models, all quantiles) for comparison with LightGBM.
const path = require("path"); const fs = require("fs");
const ROOT = path.resolve(__dirname, "../../../..");
const M = require(path.join(ROOT, "audit/exports/2026-10-25-audit3/site/js/model.js"));
const demo = require(path.join(ROOT, "web/data/demo.json"));
const G = require(path.join(ROOT, "web/data/model-guests.json"));
const D = require(path.join(ROOT, "web/data/model-dishes.json"));
const out = { g: [], gx: [], d: [], dx: [] };
for (const s of demo.services) { out.gx.push(s.guest_x); out.g.push(G.models.map((m) => M.predict(m, s.guest_x)));
  for (const d of s.dishes) { out.dx.push(d.x); out.d.push(D.models.map((m) => M.predict(m, d.x))); } }
fs.writeFileSync(process.argv[2], JSON.stringify(out));
