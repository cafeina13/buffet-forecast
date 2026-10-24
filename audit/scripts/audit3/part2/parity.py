"""Compare JS tree evaluation (parity.js output) with LightGBM on the saved boosters."""
import json, sys
from pathlib import Path
import lightgbm as lgb
import numpy as np
ROOT = Path(__file__).resolve().parents[4]
MD = ROOT / "experiments/2026-10-24-audit2-fixes/models"
o = json.load(open(sys.argv[1]))
for pre, X, P, qs in [("guests", o["gx"], o["g"], [0.1, 0.5, 0.9]), ("dishes", o["dx"], o["d"], [0.1, 0.5, 0.8, 0.9, 0.95])]:
    X = np.array([[np.nan if v is None else v for v in r] for r in X], dtype=float)
    P = np.array(P)
    for i, q in enumerate(qs):
        lg = lgb.Booster(model_file=str(MD / f"{pre}_q{q}.txt")).predict(X)
        print(pre, q, "max abs diff", float(np.max(np.abs(lg - P[:, i]))))
