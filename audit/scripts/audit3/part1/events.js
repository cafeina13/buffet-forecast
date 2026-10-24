// Event effects learned by the published guest model: median rate with each event flag on vs off,
// averaged over the services of the given meal in data/<run>/services.csv.
const fs = require("fs"), path = require("path");
const root = path.resolve(__dirname, "../../../..");
const M = require(path.join(root, "web/js/model.js"));
const G = JSON.parse(fs.readFileSync(path.join(root, "web/data/model-guests.json"), "utf8"));
const MEALS = ["breakfast", "lunch", "dinner"];
for (const run of ["train", "test"]) {
  const lines = fs.readFileSync(path.join(root, `data/${run}/services.csv`), "utf8").trim().split(/\r?\n/);
  const head = lines[0].split(",");
  const rows = lines.slice(1).map((l) => Object.fromEntries(l.split(",").map((v, i) => [head[i], v])));
  const counts = {};
  rows.forEach((r) => { counts[r.meal + ":" + r.event] = (counts[r.meal + ":" + r.event] || 0) + 1; });
  console.log(run, "event counts", counts);
  const vec = (r, ev) => G.features.map((f) => {
    if (f === "meal_code") return MEALS.indexOf(r.meal);
    if (f.startsWith("event_")) return ev === f.slice(6) ? 1 : 0;
    return Number(r[f]);
  });
  for (const [meal, ev, truth] of [["dinner", "match", -0.06], ["dinner", "themed", 0.05], ["lunch", "conference", 0.10]]) {
    const rs = rows.filter((r) => r.meal === meal);
    let s = 0;
    rs.forEach((r) => { s += M.predict(G.models[1], vec(r, ev)) / M.predict(G.models[1], vec(r, "none")) - 1; });
    console.log(` ${run} ${meal} ${ev}: mean learned effect ${(100 * s / rs.length).toFixed(2)}% (generator ${truth * 100}%)`);
  }
}
