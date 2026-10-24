// Audit 3 part 3: run the published nasil.js against web/data/demo.json with a minimal DOM stub; print text output.
const fs = require("fs");
const root = process.argv[2];
const els = {};
global.document = { getElementById: (id) => (els[id] = els[id] || { id, innerHTML: "", textContent: "" }) };
global.fetch = () => Promise.resolve({ json: () => JSON.parse(fs.readFileSync(root + "/web/data/demo.json", "utf8")) });
eval(fs.readFileSync(root + "/audit/exports/2026-10-25-audit3/site/js/nasil.js", "utf8"));
setTimeout(() => {
  const strip = (h) => h.replace(/<title>/g, " [title: ").replace(/<\/title>/g, "] ").replace(/<\/(td|th|li)>/g, " | ").replace(/<[^>]+>/g, " ").replace(/\s+/g, " ");
  for (const [k, v] of Object.entries(els)) console.log("==", k, "\n", v.textContent || strip(v.innerHTML));
}, 50);
