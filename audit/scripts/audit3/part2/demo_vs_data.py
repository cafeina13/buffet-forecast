"""Check demo.json services against data/test (menus, chef plans, presence, guest features, hist_per100)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from olculu.config import load_dishes
from olculu.features import DISH_FEATURES, GUEST_FEATURES, build_dish_frame, dish_static, prepare_services
from olculu import embeddings
ROOT = Path(__file__).resolve().parents[4]
demo = json.load(open(ROOT / "web/data/demo.json", encoding="utf-8"))
t = {k: pd.read_csv(ROOT / f"data/test/{k}.csv") for k in ["services", "servings", "hourly"]}
s = prepare_services(t["services"])
cat = load_dishes(); st, v = dish_static(cat); f = build_dish_frame(t["servings"], t["services"], t["hourly"], cat, st, embeddings.nearest(v, 5))
bad = 0; nd = 0
for sv in demo["services"]:
    row = s[(s.date == sv["date"]) & (s.meal == sv["meal"])].iloc[0]
    gx = np.array([row[c] for c in GUEST_FEATURES], dtype=float)
    if np.max(np.abs(gx - np.array(sv["guest_x"], dtype=float))) > 1e-3: bad += 1; print("guest_x mismatch", sv["date"], sv["meal"])
    fr = f[(f.date == sv["date"]) & (f.meal == sv["meal"])].set_index("dish_id")
    if set(fr.index) != {d["id"] for d in sv["dishes"]}: print("menu mismatch", sv["date"], sv["meal"])
    for d in sv["dishes"]:
        nd += 1
        r = fr.loc[d["id"]]
        if abs(r["plan_kg"] - d["chef_plan_kg"]) > 1e-3: print("plan mismatch", d["id"])
        xv = np.array([np.nan if x is None else x for x in d["x"]], dtype=float)
        rv = r[DISH_FEATURES].to_numpy(dtype=float)
        ok = np.allclose(np.nan_to_num(xv, nan=-999), np.nan_to_num(rv, nan=-999), atol=1e-3)
        if not ok: print("x mismatch", d["id"], sv["date"], sv["meal"])
# actual outcomes in the demo week: how did the chef do (true demand vs plan) -- for context
wk = f[f.date.isin(pd.to_datetime(sorted({x['date'] for x in demo['services']})))]
print("demo week dish-servings", len(wk), "checked", nd, "chef plan_short", (wk.true_demand_kg > wk.plan_kg).mean(), "observed stockouts", wk.stockout_time.notna().mean())
print("child share demo week", s[s.date.isin(wk.date.unique())].child_share.unique()[:3], "train max child", pd.read_csv(ROOT/'data/train/services.csv').child_share.max())
