// Loads the published demo.js with a stub DOM, exposes its internal functions, and
// computes every number the chef's screen shows for every demo service and risk level.
// Usage: node harness.js <outfile.json> [scenarioJSON]
const fs = require("fs");
const path = require("path");
const ROOT = path.resolve(__dirname, "../../../..");
const SITE = path.join(ROOT, "audit/exports/2026-10-24-audit1/site");
const M = require(path.join(SITE, "js/model.js"));
globalThis.window = { OlculuModel: M };
globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
globalThis.document = { getElementById: () => ({ addEventListener() {} }), querySelectorAll: () => [] };
globalThis.location = { search: "" };
globalThis.fetch = () => new Promise(() => {});
let src = fs.readFileSync(path.join(SITE, "js/demo.js"), "utf8");
src = src.replace(/\n  boot\(\);\n\}\)\(\);\s*$/, `
  globalThis.__T = { compute, state, explain, refillPlan, guestDrivers, presentGuests, roundHalfUp, interp,
    setData(a, b, c) { demo = a; G = b; D = c; gIdx = Object.fromEntries(G.features.map((f, i) => [f, i]));
      dIdx = Object.fromEntries(D.features.map((f, i) => [f, i]));
      demo.dates = [...new Set(demo.services.map((s) => s.date))].sort(); },
    get gIdx() { return gIdx; }, get dIdx() { return dIdx; } };
})();`);
if (!src.includes("__T")) throw new Error("patch failed");
eval(src);
const T = globalThis.__T;
const W = path.join(ROOT, "web/data");
const demo = JSON.parse(fs.readFileSync(path.join(W, "demo.json"), "utf8"));
const G = JSON.parse(fs.readFileSync(path.join(W, "model-guests.json"), "utf8"));
const D = JSON.parse(fs.readFileSync(path.join(W, "model-dishes.json"), "utf8"));
T.setData(demo, G, D);
const scen = process.argv[3] ? JSON.parse(process.argv[3]) : {};
const out = [];
for (let day = 0; day < demo.dates.length; day++) {
  for (const meal of ["breakfast", "lunch", "dinner"]) {
    for (const risk of [0.2, 0.1, 0.05]) {
      Object.assign(T.state, { day, meal, risk, occ: 0, temp: 0, match: null, themed: null, child: null, presence: {} }, scen);
      const service = demo.services.find((s) => s.date === demo.dates[day] && s.meal === meal);
      const r = T.compute(service);
      out.push({ date: service.date, meal, risk, event: service.event, guests: r.guests, gq: r.gq, present: r.present,
        rate: r.rate, drivers: r.drivers, match: r.sc.match, themed: r.sc.themed, child: r.sc.child,
        dishes: r.dishes.map((d) => ({ id: d.id, name: d.info.name, unit: d.info.unit, kgu: d.info.kg_per_unit, q: d.q,
          recKg: d.recKg, recUnits: d.recUnits, low: d.lowUnits, high: d.highUnits, chef: d.chefUnits,
          reason: d.reason, refill: d.refill, isNew: d.info.new })) });
    }
  }
}
fs.writeFileSync(process.argv[2], JSON.stringify(out, null, 1));
console.log("services x risks:", out.length);
