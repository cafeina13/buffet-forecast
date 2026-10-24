// Prints browser-model predictions (model.js) for every demo service and dish, base scenario.
const M = require("../../../exports/2026-10-24-audit2/site/js/model.js");
const fs = require("fs");
const j = (f) => JSON.parse(fs.readFileSync(__dirname + "/../../../../web/data/" + f, "utf8"));
const demo = j("demo.json"), G = j("model-guests.json"), D = j("model-dishes.json");
const out = demo.services.map((s) => ({ date: s.date, meal: s.meal,
  rate: M.predictQuantiles(G, s.guest_x.map((v) => (v === null ? NaN : v))),
  dishes: s.dishes.map((d) => ({ id: d.id, q: M.predictQuantiles(D, d.x.map((v) => (v === null ? NaN : v))) })) }));
fs.writeFileSync(__dirname + "/parity_js.json", JSON.stringify(out));
