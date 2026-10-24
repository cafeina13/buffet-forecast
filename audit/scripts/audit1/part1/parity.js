const M = require("../../../../web/js/model.js");
const d = require("../../../../web/data/demo.json"), G = require("../../../../web/data/model-guests.json"), D = require("../../../../web/data/model-dishes.json");
const out = { g: [], d: [] };
d.services.forEach((s) => { out.g.push({ x: s.guest_x, p: G.models.map((m) => M.predict(m, s.guest_x)) });
  s.dishes.forEach((x) => out.d.push({ x: x.x, p: D.models.map((m) => M.predict(m, x.x)) })); });
require("fs").writeFileSync(__dirname + "/parity_js.json", JSON.stringify(out));
