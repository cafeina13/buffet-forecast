// Runs the published demo.js logic headlessly on web/data/*.json (read-only).
const fs = require("fs");
const path = require("path");
const ROOT = path.resolve(__dirname, "../../../..");
globalThis.window = globalThis;
globalThis.OlculuModel = require(path.join(ROOT, "web/js/model.js"));
let src = fs.readFileSync(path.join(ROOT, "web/js/demo.js"), "utf8");
src = src.replace(/\n  boot\(\);\n\}\)\(\);\s*$/, `
  globalThis.__t = { compute, explain, state, currentService,
    setup(d, g, m) { demo = d; G = g; D = m;
      gIdx = Object.fromEntries(G.features.map((f, i) => [f, i]));
      dIdx = Object.fromEntries(D.features.map((f, i) => [f, i]));
      demo.dates = [...new Set(demo.services.map((s) => s.date))].sort(); },
    get demo() { return demo; } };
})();`);
globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
eval(src);
const rd = (f) => JSON.parse(fs.readFileSync(path.join(ROOT, "web/data", f), "utf8"));
const T = globalThis.__t;
T.setup(rd("demo.json"), rd("model-guests.json"), rd("model-dishes.json"));
module.exports = T;
if (require.main === module) {
  const demo = T.demo;
  const out = [];
  for (const risk of [0.2, 0.1, 0.05]) {
    T.state.risk = risk;
    demo.dates.forEach((dt, i) => {
      for (const meal of ["breakfast", "lunch", "dinner"]) {
        T.state.day = i; T.state.meal = meal;
        const svc = T.currentService();
        const r = T.compute(svc);
        r.dishes.forEach((d) => out.push({ risk, date: dt, meal, event: svc.event, dish: d.info.name, unit: d.info.unit,
          rec: d.recUnits, chef: d.chefUnits, reason: d.reason, drivers: r.drivers.map(x=>x.label+" "+Math.round(x.people)).join(" | "),
          guests: Math.round(r.guests), refill: JSON.stringify(d.refill) }));
      }
    });
  }
  fs.writeFileSync(path.join(__dirname, "demo_out.json"), JSON.stringify(out, null, 1));
  console.log("rows", out.length);
}
