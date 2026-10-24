// Audit 3 part 2: run the published demo.js against a tiny DOM stub and dump the text it renders.
// Usage: node harness.js [meal] [dayIndex] [riskIndex] [extra-json]
// extra-json e.g. '{"occ":10,"temp":5,"child":30,"match":true,"tab":"live","clock":3}'
const fs = require("fs");
const path = require("path");
const ROOT = path.resolve(__dirname, "../../../..");
const SITE = path.join(ROOT, "audit/exports/2026-10-25-audit3/site");

class Node {
  constructor(tag) { this.tag = tag; this.children = []; this.attrs = {}; this._text = ""; this.listeners = {}; this.hidden = false; this.dataset = {}; this.className = ""; this.value = ""; this.checked = false; this.open = false; }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(""); }
  set innerHTML(v) { this.children = []; this._text = ""; this._html = v; }
  get innerHTML() { return this._html || ""; }
  appendChild(c) { this.children.push(c); return c; }
  setAttribute(k, v) { this.attrs[k] = v; if (k.startsWith("data-")) this.dataset[k.slice(5)] = v; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener(ev, fn) { (this.listeners[ev] = this.listeners[ev] || []).push(fn); }
  fire(ev, e) { (this.listeners[ev] || []).forEach((fn) => fn(e || { target: this })); }
}
class TextNode { constructor(t) { this.t = t; } get textContent() { return this.t; } }
const byId = {};
const document = {
  getElementById(id) { return byId[id] || (byId[id] = new Node("div")); },
  createElement(t) { return new Node(t); },
  createTextNode(t) { return new TextNode(t); },
  querySelectorAll(sel) {
    if (sel === "#presence-controls [data-val]") {
      const out = [];
      const walk = (n) => { if (n.dataset && n.dataset.val) out.push(n); (n.children || []).forEach(walk); };
      walk(byId["presence-controls"] || new Node("x"));
      return out;
    }
    return [];
  },
  body: { dataset: {} },
};
const args = process.argv.slice(2);
const meal = args[0] || "dinner";
const extra = args[3] ? JSON.parse(args[3]) : {};
const store = {};
global.window = global;
global.document = document;
global.localStorage = { getItem: (k) => store[k] || null, setItem: (k, v) => { store[k] = v; }, removeItem: (k) => { delete store[k]; } };
global.location = { search: `?meal=${meal}&tab=${extra.tab || "plan"}${extra.clock !== undefined ? "&clock=" + extra.clock : ""}` };
const data = {};
for (const f of ["demo.json", "model-guests.json", "model-dishes.json"]) data["data/" + f] = JSON.parse(fs.readFileSync(path.join(ROOT, "web/data", f), "utf8"));
global.fetch = async (u) => ({ ok: true, json: async () => data[u] });
eval(fs.readFileSync(path.join(SITE, "js/site-config.js"), "utf8"));
global.OlculuModel = require(path.join(SITE, "js/model.js"));
eval(fs.readFileSync(path.join(SITE, "js/demo.js"), "utf8"));

function dump(n, depth = 0) {
  const out = [];
  const walk = (x) => {
    if (x instanceof TextNode) return;
    if (x.tag === "article" || x.tag === "li" || (x.tag === "p" && x.className)) { out.push(x.textContent); return; }
    (x.children || []).forEach(walk);
  };
  walk(n);
  return out;
}
setTimeout(() => {
  if (args[1] !== undefined) byId["day-chips"].children[Number(args[1])].fire("click");
  if (args[2] !== undefined) byId["risk-seg"].children[Number(args[2])].fire("click");
  if (extra.occ !== undefined) { byId.occ.value = extra.occ; byId.occ.fire("input", { target: { value: extra.occ } }); }
  if (extra.temp !== undefined) byId.temp.fire("input", { target: { value: extra.temp } });
  if (extra.child !== undefined) byId.child.fire("input", { target: { value: extra.child } });
  if (extra.match !== undefined) byId.match.fire("change", { target: { checked: extra.match } });
  if (extra.themed !== undefined) byId.themed.fire("change", { target: { checked: extra.themed } });
  if (extra.presence) {
    const walk = (n) => { if (n.attrs && n.attrs["data-key"] in extra.presence) n.fire("input", { target: { value: extra.presence[n.attrs["data-key"]] } }); (n.children || []).forEach(walk); };
    walk(byId["presence-controls"]);
  }
  const sections = extra.only ? extra.only.split(",") : ["guest", "occ", "dish", "live", "report"];
  if (sections.includes("guest")) {
    console.log("GUEST", byId["guest-number"].textContent, "|", byId["guest-range"].textContent, "|", byId["event-tag"].hidden ? "" : byId["event-tag"].textContent);
    console.log(dump(byId["guest-breakdown"]).join(" / "));
    console.log("DRIVERS", byId["guest-drivers"].textContent);
  }
  if (sections.includes("occ")) console.log("OCC", byId["occ-val"].textContent, "TEMP", byId["temp-val"].textContent, "CHILD", byId["child-val"].textContent, byId["child-note"].textContent, "presence:", dump(byId["presence-controls"]).join(";"), JSON.stringify((byId["presence-controls"].children||[]).map(l=>l.textContent)));
  if (sections.includes("dish")) dump(byId["dish-grid"]).forEach((t) => console.log("DISH", t));
  if (sections.includes("live")) { console.log("CLOCK", byId["clock-val"].textContent); console.log("NOW", dump(byId["refill-now"]).join(" | ")); console.log("NEXT", dump(byId["refill-next"]).join(" | ")); }
  if (sections.includes("report")) console.log("REPORT", dump(byId.report).join("\n"));
}, 50);
