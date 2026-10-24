"""Why is the child effect not learned? Collinearity in the data, and the learned response (read-only)."""
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

from olculu import embeddings
from olculu.config import load_dishes
from olculu.features import DISH_FEATURES, build_dish_frame, dish_static, prepare_services
from olculu.train import load_run

record = sys.argv[1]
cat = load_dishes()
tr = prepare_services(load_run("train")["services"])
d = tr[tr["meal"] == "dinner"].copy()
d["doy"] = d["date"].dt.dayofyear
print("train child share: min %.3f max %.3f sd %.3f" % (d.child_share.min(), d.child_share.max(), d.child_share.std()))
print("corr(child, day-of-season bump proxy):", round(np.corrcoef(d.child_share, -(d.doy - d.doy.mean()) ** 2)[0, 1], 3),
      " corr(child, temperature):", round(np.corrcoef(d.child_share, d.temperature)[0, 1], 3))
# Day-to-day variation left after removing a 15-day rolling mean.
resid = d.child_share - d.child_share.rolling(15, center=True, min_periods=1).mean()
print("day-to-day child variation (sd of residual):", round(float(resid.std()), 4))

test = load_run("test")
static, vecs = dish_static(cat, "tfidf")
f = build_dish_frame(test["servings"], test["services"], test["hourly"], cat, static, embeddings.nearest(vecs, 5))
b = lgb.Booster(model_file=f"{record}/models/dishes_q0.5.txt")
X = f[DISH_FEATURES].copy()
lo, hi = d.child_share.quantile(0.05), d.child_share.quantile(0.95)
X_lo, X_hi = X.copy(), X.copy()
X_lo["child_share"], X_hi["child_share"] = lo, hi
eff = b.predict(X_hi) / b.predict(X_lo) - 1
gen = (1 + 0.10 * hi / 0.2) / (1 + 0.10 * lo / 0.2) - 1
print(f"child share {lo:.3f} -> {hi:.3f}: learned median effect {np.median(eff):+.1%} (10th {np.percentile(eff,10):+.1%}, 90th {np.percentile(eff,90):+.1%}), "
      f"wrong sign on {np.mean(eff < 0):.0%} of rows; generator effect {gen:+.1%}")
