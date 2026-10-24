// Approve a dish at risk 10 %, then switch the risk: does the shown quantity follow?
process.argv[2] = __dirname + "/../../../exports/2026-10-24-audit2/site";
process.argv[3] = __dirname + "/../../../../web/data";
process.env.NOOUT = "1";
const { byId } = require("./harness.js");
const card = () => byId["dish-grid"].children.find((c) => c.children[1].textContent === "Mercimek çorbası");
const txt = () => card().children.map((p) => p.textContent).join(" | ");
setTimeout(() => {
  console.log("10% before:", txt());
  card().children[7].children[0].fire("click");
  console.log("10% approved:", txt());
  byId["risk-seg"].children[0].fire("click");
  console.log("20% after approval:", txt());
  byId.occ.fire("input", { target: { value: "-25" } });
  console.log("20%, occ -25:", txt(), "\n guests", byId["guest-number"].textContent);
}, 300);
