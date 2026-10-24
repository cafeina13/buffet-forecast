"""Mean |SHAP| (in guests) of the saved guest-rule rate model on the open test season, and event effects vs generator."""
import sys
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from olculu.features import prepare_services, GUEST_FEATURES
b = lgb.Booster(model_file=str(ROOT / "experiments/2026-10-24-guest-rule/models/guests_q0.5.txt"))
s = prepare_services(pd.read_csv(ROOT / "data/test/services.csv"))
s = s[s["date"] < "2026-10-11"].reset_index(drop=True)
c = b.predict(s[GUEST_FEATURES], pred_contrib=True)[:, :-1] * s["present"].to_numpy()[:, None]
print(pd.Series(np.abs(c).mean(0), index=GUEST_FEATURES).sort_values(ascending=False).round(2).to_string())
for ev, meal, eff in [("event_match", "dinner", -0.06), ("event_themed", "dinner", 0.05), ("event_conference", "lunch", 0.10)]:
    m = (s[ev] == 1) & (s["meal"] == meal)
    X1 = s.loc[s["meal"] == meal, GUEST_FEATURES].copy(); X0 = X1.copy()
    X1[ev] = 1; X0[ev] = 0
    rel = np.mean(b.predict(X1) / b.predict(X0) - 1)
    print(ev, "n=", int(m.sum()), "model rel effect", round(rel, 3), "generator", eff)
for col, dlt in [("temperature", 5), ("occupancy", 0.1)]:
    X1 = s[GUEST_FEATURES].copy(); X0 = X1.copy(); X1[col] += dlt
    print(col, "+", dlt, "mean guests change", round(np.mean((b.predict(X1) - b.predict(X0)) * s["present"]), 2), "generator: 0")
