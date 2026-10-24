// Runs the published demo.js against a minimal DOM shim and dumps the rendered text.
// Usage: node harness.js <siteDir> <dataDir> [query] [actions.json]
// Prints JSON: one record per (day, meal, risk) with all visible text.
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const siteDir = process.argv[2];
const dataDir = process.argv[3];
const html = fs.readFileSync(path.join(siteDir, "demo.html"), "utf8");

class El {
  constructor(tag) {
    this.tagName = tag.toUpperCase(); this.children = []; this.attrs = {}; this.listeners = {};
    this._text = ""; this._html = ""; this.hidden = false; this.dataset = {}; this.className = "";
    this.value = ""; this.checked = false; this.open = false; this.min = ""; this.max = "";
  }
  set textContent(v) { this._text = String(v); this.children = []; this._html = ""; }
  get textContent() { return this._text + this._html.replace(/<[^>]*>/g, " ") + this.children.map((c) => c.textContent).join(" "); }
  set innerHTML(v) { this._html = String(v); this.children = []; this._text = ""; }
  get innerHTML() { return this._html; }
  appendChild(c) { this.children.push(c); return c; }
  setAttribute(k, v) {
    this.attrs[k] = String(v);
    if (k.startsWith("data-")) this.dataset[k.slice(5).replace(/-(\w)/g, (_, a) => a.toUpperCase())] = String(v);
    if (k === "value") this.value = String(v);
  }
  getAttribute(k) { return this.attrs[k]; }
  removeAttribute(k) { delete this.attrs[k]; }
  addEventListener(t, f) { (this.listeners[t] = this.listeners[t] || []).push(f); }
  fire(t, ev) { (this.listeners[t] || []).forEach((f) => f(ev || { target: this })); }
  all() { return [this, ...this.children.flatMap((c) => (c instanceof El ? c.all() : []))]; }
}
const byId = {};
for (const m of html.matchAll(/<(\w+)[^>]*id="([^"]+)"/g)) byId[m[2]] = new El(m[1]);
byId.app.hidden = true;
const document = {
  getElementById: (id) => byId[id],
  createElement: (t) => new El(t),
  createTextNode: (t) => { const e = new El("#text"); e._text = t; return e; },
  querySelectorAll: (sel) => {
    if (sel === "#presence-controls [data-val]") return byId["presence-controls"].all().filter((e) => e.dataset.val);
    return [];
  },
  body: { dataset: {} },
  title: "",
};
const query = process.argv[4] || "";
const store = {};
const ctx = {
  window: {}, document, console, Intl, Math, JSON, Number, String, Object, Array, Promise, Error, Set,
  URLSearchParams, location: { search: query },
  localStorage: { getItem: (k) => store[k] ?? null, setItem: (k, v) => { store[k] = v; }, removeItem: (k) => { delete store[k]; } },
  fetch: (u) => Promise.resolve({ ok: true, json: () => Promise.resolve(JSON.parse(fs.readFileSync(path.join(dataDir, path.basename(u)), "utf8"))) }),
};
ctx.window = ctx; ctx.globalThis = ctx;
vm.createContext(ctx);
for (const f of ["site-config.js", "site.js", "model.js", "demo.js"]) {
  vm.runInContext(fs.readFileSync(path.join(siteDir, "js", f), "utf8"), ctx, { filename: f });
}
module.exports = { byId, ctx, El };

function textOf(id) { return byId[id].textContent.replace(/\s+/g, " ").trim(); }
function dishCards() {
  return byId["dish-grid"].children.map((c) => c.children.map((p) => p.textContent.replace(/\s+/g, " ").trim()));
}
function click(elm) { elm.fire("click"); }

if (!process.env.NOOUT) setTimeout(() => {
  const out = [];
  const nDays = byId["day-chips"].children.length;
  const which = process.env.MODE || "all";
  for (let di = 0; di < nDays; di++) {
    for (let mi = 0; mi < 3; mi++) {
      click(byId["day-chips"].children[di]);
      click(byId["meal-seg"].children[mi]);
      for (let ri = 0; ri < 3; ri++) {
        click(byId["risk-seg"].children[ri]);
        out.push({
          day: textOf("day-chips").split(" ").slice(0), di, mi, ri,
          chip: byId["day-chips"].children[di].textContent.trim(),
          meal: byId["meal-seg"].children[mi].textContent.trim(),
          risk: byId["risk-seg"].children[ri].textContent.trim(),
          guest: textOf("guest-number"), range: textOf("guest-range"),
          breakdown: byId["guest-breakdown"].children.map((c) => c.textContent.trim()),
          drivers: textOf("guest-drivers"), eventTag: byId["event-tag"].hidden ? "" : textOf("event-tag"),
          presence: textOf("presence-controls"), child: textOf("child-val"), childMin: byId.child.min, childMax: byId.child.max, childVal: byId.child.value,
          dishes: which === "all" ? dishCards() : undefined,
          clock: textOf("clock-val"), now: textOf("refill-now"), next: textOf("refill-next"),
          report: textOf("report"),
          flow: byId["flow-chart"].innerHTML.match(/<title>[^<]*<\/title>/g),
        });
      }
    }
  }
  console.log(JSON.stringify({ record: textOf("record"), out }, null, 1));
}, 200);
