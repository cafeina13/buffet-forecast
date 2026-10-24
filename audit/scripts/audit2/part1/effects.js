// Learned event effects of the guest model (median), averaged over demo services; split-gain-free
// count of splits per feature in each exported model.
const fs = require("fs");
const path = require("path");
const root = path.resolve(__dirname, "../../../..");
const M = require(path.join(root, "audit/exports/2026-10-24-audit2/site/js/model.js"));
const demo = JSON.parse(fs.readFileSync(path.join(root, "web/data/demo.json"), "utf8"));
const G = JSON.parse(fs.readFileSync(path.join(root, "web/data/model-guests.json"), "utf8"));
const D = JSON.parse(fs.readFileSync(path.join(root, "web/data/model-dishes.json"), "utf8"));
const gi = Object.fromEntries(G.features.map((f, i) => [f, i]));

function splits(bundle, qi) {
  const cnt = {};
  for (const t of bundle.models[qi].trees) for (const f of t.f) if (f !== -1) cnt[bundle.features[f]] = (cnt[bundle.features[f]] || 0) + 1;
  return Object.entries(cnt).sort((a, b) => b[1] - a[1]);
}
console.log("guest median splits", JSON.stringify(splits(G, 1)));
console.log("dish median splits", JSON.stringify(splits(D, 1)));

for (const meal of ["breakfast", "lunch", "dinner"]) {
  for (const ev of ["event_match", "event_themed", "event_conference"]) {
    const effs = [];
    for (const s of demo.services.filter((s) => s.meal === meal)) {
      const on = s.guest_x.slice(), off = s.guest_x.slice();
      for (const e of ["event_match", "event_themed", "event_conference"]) { on[gi[e]] = 0; off[gi[e]] = 0; }
      on[gi[ev]] = 1;
      effs.push(M.predict(G.models[1], on) / M.predict(G.models[1], off) - 1);
    }
    console.log(meal, ev, "mean rel effect", (effs.reduce((a, b) => a + b, 0) / effs.length * 100).toFixed(2) + "%");
  }
}
// sensitivity of guest median to temperature, child share, weekday, nationality over demo dinner
const s = demo.services.find((s) => s.meal === "dinner" && s.event === "none");
const base = M.predict(G.models[1], s.guest_x);
for (const [f, vals] of [["temperature", [20, 25, 30, 35, 38]], ["child_share", [0.14, 0.2, 0.26]], ["weekday", [0, 1, 2, 3, 4, 5, 6]], ["nat_TR", [0.1, 0.2, 0.3]], ["occupancy", [0.5, 0.7, 0.9]]]) {
  console.log(f, vals.map((v) => { const x = s.guest_x.slice(); x[gi[f]] = v; return (M.predict(G.models[1], x) / base - 1) * 100; }).map((v) => v.toFixed(2)).join(" "));
}
