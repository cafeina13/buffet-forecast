"""Check that web/js/model.js reproduces LightGBM predictions and SHAP values for every quantile model.

Run after training: uv run python tests/js_parity.py experiments/<date>-<name>
Writes expected values with LightGBM into the record's models/ folder, runs the JS runtime in Node, compares.
"""

import json
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python tests/js_parity.py experiments/<date>-<name>")
    models_dir = ROOT / sys.argv[1] / "models"
    cases = []
    for prefix, rows_file, web_file in [("dishes", "dish_rows_sample.csv", "model-dishes.json"),
                                        ("guests", "guest_rows_sample.csv", "model-guests.json")]:
        X = pd.read_csv(models_dir / rows_file)
        bundle = json.loads((ROOT / "web/data" / web_file).read_text(encoding="utf-8"))
        for i, q in enumerate(bundle["quantiles"]):
            booster = lgb.Booster(model_file=str(models_dir / f"{prefix}_q{q}.txt"))
            cases.append({"web": web_file, "quantile_index": i,
                          "rows": X.to_numpy().tolist(), "pred": booster.predict(X).tolist(),
                          "contrib": booster.predict(X, pred_contrib=True).tolist()})

    tmp = models_dir / "parity_cases.json"
    tmp.write_text(json.dumps(cases).replace("NaN", "null"), encoding="utf-8")
    script = r"""
const fs = require('fs');
// With node -e there is no script path, so arguments start at argv[1].
const [modelJs, casesFile, webData] = process.argv.slice(-3);
const M = require(modelJs);
const cases = JSON.parse(fs.readFileSync(casesFile, 'utf8'));
let worstPred = 0, worstShap = 0, worstSum = 0, checked = 0;
for (const c of cases) {
  const bundle = JSON.parse(fs.readFileSync(webData + '/' + c.web, 'utf8'));
  const model = bundle.models[c.quantile_index];
  c.rows.forEach((x, i) => {
    const p = M.predict(model, x);
    worstPred = Math.max(worstPred, Math.abs(p - c.pred[i]));
    const s = M.shap(model, x, x.length);
    const exp = c.contrib[i];
    s.phi.forEach((v, k) => { worstShap = Math.max(worstShap, Math.abs(v - exp[k])); });
    worstShap = Math.max(worstShap, Math.abs(s.base - exp[exp.length - 1]));
    worstSum = Math.max(worstSum, Math.abs(s.base + s.phi.reduce((a, b) => a + b, 0) - p));
    checked++;
  });
}
console.log(JSON.stringify({ models: cases.length, rows: checked, worstPred, worstShap, worstSum }));
"""
    out = subprocess.run(["node", "-e", script, str(ROOT / "web/js/model.js"), str(tmp), str(ROOT / "web/data")],
                         capture_output=True, text=True)
    if out.returncode != 0:
        print(out.stderr)
        sys.exit(1)
    res = json.loads(out.stdout)
    print(res)
    # Exported leaf values are rounded to 6 decimals, so allow a small tolerance.
    ok = res["worstPred"] < 1e-3 and res["worstShap"] < 1e-3 and res["worstSum"] < 1e-6
    print("PARITY OK" if ok else "PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
