// Recompute guest MAEs on the open test weeks (before 2026-10-11) from data/ and the published model.
const fs = require("fs"), path = require("path");
const root = path.resolve(__dirname, "../../../..");
const M = require(path.join(root, "web/js/model.js"));
const G = JSON.parse(fs.readFileSync(path.join(root, "web/data/model-guests.json"), "utf8"));
const MEALS = ["breakfast", "lunch", "dinner"];
const load = (run) => {
  const lines = fs.readFileSync(path.join(root, `data/${run}/services.csv`), "utf8").trim().split(/\r?\n/);
  const head = lines[0].split(",");
  return lines.slice(1).map((l) => Object.fromEntries(l.split(",").map((v, i) => [head[i], isNaN(+v) || v === "" ? v : +v])));
};
const present = (r) => Math.max(1, r.meal === "breakfast" ? r.in_house - r.departures_very_early
  : r.meal === "lunch" ? r.in_house - r.departures_early - r.excursions : r.in_house - r.arrivals_night - r.alacarte);
const tr = load("train"), te = load("test").filter((r) => r.date < "2026-10-11");
const mean = (a) => a.reduce((s, x) => s + x, 0) / a.length;
const mr = {}, pr = {};
for (const m of MEALS) {
  const rs = tr.filter((r) => r.meal === m);
  mr[m] = mean(rs.map((r) => r.attendance / r.in_house));
  pr[m] = mean(rs.map((r) => r.attendance / present(r)));
}
const vec = (r) => G.features.map((f) => f === "meal_code" ? MEALS.indexOf(r.meal) : f.startsWith("event_") ? (r.event === f.slice(6) ? 1 : 0) : r[f]);
const full = mean(te.map((r) => Math.abs(M.predictQuantiles(G, vec(r))[1] * present(r) - r.attendance)));
const ih = mean(te.map((r) => Math.abs(r.in_house * mr[r.meal] - r.attendance)));
const pp = mean(te.map((r) => Math.abs(present(r) * pr[r.meal] - r.attendance)));
const cov = mean(te.map((r) => { const q = M.predictQuantiles(G, vec(r)).map((x) => x * present(r)); return r.attendance >= q[0] && r.attendance <= q[2] ? 1 : 0; }));
console.log({ n: te.length, full, in_house_x_rate: ih, presence_x_rate: pp, cov, rule_share_of_gain: (ih - pp) / (ih - full) });
