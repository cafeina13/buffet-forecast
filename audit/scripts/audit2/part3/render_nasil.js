// Run site/js/nasil.js against a stub DOM and print every text it writes (audit 2 part 3).
const fs = require("fs"), path = require("path");
const root = path.resolve(__dirname, "../../../..");
const site = path.join(root, "audit/exports/2026-10-24-audit2/site");
const els = {};
global.document = { getElementById: (id) => (els[id] = els[id] || { id, innerHTML: "", textContent: "" }) };
global.fetch = (u) => Promise.resolve({ json: () => JSON.parse(fs.readFileSync(path.join(root, "web", u), "utf8")) });
eval(fs.readFileSync(path.join(site, "js/nasil.js"), "utf8"));
setTimeout(() => {
  const strip = (h) => h.replace(/<title>(.*?)<\/title>/g, " [title: $1] ").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ");
  for (const [k, v] of Object.entries(els)) console.log("##", k, "\n", v.textContent || strip(v.innerHTML), "\n");
}, 200);
