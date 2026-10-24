// Drives scenario controls of the real demo.js: occupancy, temperature, child share, toggles.
process.argv[2] = __dirname + "/../../../exports/2026-10-24-audit2/site";
process.argv[3] = __dirname + "/../../../../web/data";
process.env.NOOUT = "1";
const { byId } = require("./harness.js");
const t = (id) => byId[id].textContent.replace(/\s+/g, " ").trim();
const card = (n) => byId["dish-grid"].children.find((c) => c.children[1].textContent === n).children.map((p) => p.textContent).join(" | ");
setTimeout(() => {
  const show = (lbl) => console.log(`--- ${lbl}\n guest ${t("guest-number")} | ${t("guest-range")}\n ${t("guest-breakdown")}\n drivers: ${t("guest-drivers")}\n presence: ${t("presence-controls")}\n occ ${t("occ-val")} temp ${t("temp-val")} child ${t("child-val")}\n ${card("Mercimek çorbası")}`);
  show("base 15 Jul dinner");
  byId.occ.fire("input", { target: { value: "25" } }); show("occ +25");
  byId.occ.fire("input", { target: { value: "-25" } }); show("occ -25");
  byId.reset.fire("click");
  byId.temp.fire("input", { target: { value: "8" } }); show("temp +8");
  byId.temp.fire("input", { target: { value: "-8" } }); show("temp -8");
  byId.reset.fire("click");
  byId.match.fire("change", { target: { checked: false } }); show("match off");
  byId.themed.fire("change", { target: { checked: true } }); show("themed on");
  byId.reset.fire("click");
  byId.child.fire("input", { target: { value: "15" } }); show("child 15");
  byId.child.fire("input", { target: { value: "26" } }); show("child 26");
}, 300);
