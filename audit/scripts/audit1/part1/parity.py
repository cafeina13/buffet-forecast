import json, numpy as np, lightgbm as lgb
from pathlib import Path
H = Path(__file__).parent; R = H.parents[3] / "experiments/2026-10-24-guest-rule/models"
o = json.load(open(H / "parity_js.json"))
for key, pre, qs in [("g", "guests", [0.1, 0.5, 0.9]), ("d", "dishes", [0.1, 0.5, 0.8, 0.9, 0.95])]:
    X = np.array([[np.nan if v is None else v for v in r["x"]] for r in o[key]], dtype=float)
    P = np.column_stack([lgb.Booster(model_file=str(R / f"{pre}_q{q}.txt")).predict(X) for q in qs])
    J = np.array([r["p"] for r in o[key]])
    print(key, X.shape, "max abs diff", np.abs(P - J).max())
