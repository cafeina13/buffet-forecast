"""Independent recomputation of the chef's-screen numbers (audit 2, part 2).

Loads the saved boosters of the published record (no training), rebuilds the test-season
features with the project's feature code, then recomputes with my own code:
  - July monthly report (waste, cost, CO2, stock-out rates) at risks 20/10/5 %
  - nominal vs realised "first cook exceeded" rates, and interval coverage
  - hist_per100 (censoring-corrected vs observed-only) for the demo week
  - no-history rows: is the interval wider?
  - demo.json services vs the CSV data (presence, chef plan, features)
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from olculu import embeddings  # noqa: E402
from olculu.config import load_dishes, load_synthetic  # noqa: E402
from olculu.features import (DISH_FEATURES, GUEST_FEATURES, build_dish_frame, dish_static,  # noqa: E402
                             prepare_services)

REC = ROOT / "experiments" / "2026-10-24-audit1-fixes" / "models"
cfg, cat = load_synthetic(), load_dishes()
dishes = {d["id"]: d for d in cat["dishes"]}
demo = json.loads((ROOT / "web" / "data" / "demo.json").read_text(encoding="utf-8"))


def boosters(prefix, qs):
    return [lgb.Booster(model_file=str(REC / f"{prefix}_q{q}.txt")) for q in qs]


GQ, DQ = [0.1, 0.5, 0.9], [0.1, 0.5, 0.8, 0.9, 0.95]
gb, db = boosters("guests", GQ), boosters("dishes", DQ)

d = ROOT / "data" / "test"
sv, sg, hr = (pd.read_csv(d / f"{t}.csv") for t in ["services", "servings", "hourly"])
s = prepare_services(sv)
rate = np.sort(np.column_stack([b.predict(s[GUEST_FEATURES]) for b in gb]), axis=1)
s["g_med"] = rate[:, 1] * s["present"]
s["g_lo"], s["g_hi"] = rate[:, 0] * s["present"], rate[:, 2] * s["present"]
cutoff = pd.Timestamp("2026-10-11")  # season end 2026-10-31 minus 3 weeks + 1 day
openm = s["date"] < cutoff
print("guest q10-q90 coverage (open test):",
      round(float(((s.attendance >= s.g_lo) & (s.attendance <= s.g_hi))[openm].mean()), 4))

static, vecs = dish_static(cat, "tfidf")
nb = embeddings.nearest(vecs, 5)
f = build_dish_frame(sg, sv, hr, cat, static, nb).merge(s[["date", "meal", "g_med"]], on=["date", "meal"])
ev = f[f["date"] < cutoff].copy().reset_index(drop=True)
q = np.sort(np.column_stack([b.predict(ev[DISH_FEATURES]) for b in db]), axis=1) * ev["g_med"].to_numpy()[:, None] / 100
y = ev["true_demand_kg"].to_numpy()
unit = ev["dish_id"].map(lambda k: dishes[k]["kg_per_unit"]).to_numpy()
print("rows", len(ev), "coverage", {qq: round(float((y <= q[:, i]).mean()), 4) for i, qq in enumerate(DQ)})
print("q10-q90 interval coverage:", round(float(((y >= q[:, 0]) & (y <= q[:, 3])).mean()), 4))


def level(qm, lv):
    return np.array([np.interp(lv, DQ, r) for r in qm])


def roundup(kg, u):
    return np.maximum(0.5 * u, np.ceil(kg / u * 2 - 1e-9) / 2 * u)


def sim(plan, true, u, share=0.12):
    """My own implementation of the refill rule described in the generator."""
    prod = np.where(true <= plan, plan, np.minimum(plan * (1 + share), plan + np.ceil((true - plan) / u) * u))
    return prod, prod - np.minimum(true, prod), true > prod


price = ev["group"].map(cfg["costs"]["try_per_kg"]).to_numpy()
jul = (ev["date"].dt.month == 7).to_numpy()
key = ev["date"].astype(str) + ev["meal"]
cp, cw, cs = sim(ev["plan_kg"].to_numpy(), y, unit)
print("\nJULY chef: waste", round(cw[jul].sum(), 1), "stockout_rate", round(cs[jul].mean(), 4),
      "svc_with_so", round(pd.Series(cs[jul]).groupby(key[jul].to_numpy()).any().mean(), 4),
      "n_services", key[jul].nunique(), "plan_short", round(float((y[jul] > ev.plan_kg.to_numpy()[jul]).mean()), 4))
for r in [0.2, 0.1, 0.05]:
    rec = roundup(level(q, 1 - r), unit)
    p, w, so = sim(rec, y, unit)
    print(f"risk {r}: waste {w[jul].sum():.1f} avoided {cw[jul].sum() - w[jul].sum():.1f} "
          f"TL {(cw[jul] * price[jul]).sum() - (w[jul] * price[jul]).sum():.0f} "
          f"so_rate {so[jul].mean():.4f} svc_so {pd.Series(so[jul]).groupby(key[jul].to_numpy()).any().mean():.4f} "
          f"first_cook_short {(y[jul] > rec[jul]).mean():.4f} (all open test {(y > rec).mean():.4f}) "
          f"so_all {so.mean():.4f} prod {p[jul].sum():.0f}")
    # If only the scheduled 40%+batches were "first cooked" (demo Live tab), how often is that exceeded?
    init = np.minimum(rec / unit, np.maximum(1, np.ceil(rec / unit * 0.4))) * unit
    print(f"    demand > demo 'Ilk' (initial batch) share: {(y > init).mean():.4f}")

# Interval width: rows without own history vs with history
nh = ev["hist_per100"].isna().to_numpy()
wid = (q[:, 3] - q[:, 0]) / np.maximum(q[:, 1], 1e-9)
print("\nrel. width q10-q90 / q50: no-history", round(float(np.median(wid[nh])), 3), "n", int(nh.sum()),
      "with history", round(float(np.median(wid[~nh])), 3))
print("coverage q10-q90 no-history", round(float(((y >= q[:, 0]) & (y <= q[:, 3]))[nh].mean()), 3))

# hist_per100 in the demo week: corrected vs observed-only (my own expanding mean)
sg2 = sg.copy()
sg2["date"] = pd.to_datetime(sg2["date"])
s_att = s.set_index(["date", "meal"])["attendance"]
sg2["att"] = s_att.reindex(pd.MultiIndex.from_frame(sg2[["date", "meal"]])).to_numpy()
sg2["obs100"] = sg2["observed_kg"] / sg2["att"] * 100
f2 = f.copy()
f2["tgt100"] = f2["target_kg"] / f2["attendance"] * 100
rows = []
for sv_ in demo["services"]:
    dt = pd.Timestamp(sv_["date"])
    for dd in sv_["dishes"]:
        past = sg2[(sg2.dish_id == dd["id"]) & (sg2.meal == sv_["meal"]) & (sg2.date < dt)]
        pastf = f2[(f2.dish_id == dd["id"]) & (f2.meal == sv_["meal"]) & (f2.date < dt)]
        hx = dd["x"][DISH_FEATURES.index("hist_per100")]
        rows.append(dict(date=sv_["date"], meal=sv_["meal"], dish=dd["id"], shown=hx,
                         obs=past.obs100.mean() if len(past) else None,
                         corrm=pastf.tgt100.mean() if len(pastf) else None, n=len(past),
                         n_so=int(past.stockout_time.notna().sum())))
h = pd.DataFrame(rows)
h["diff_vs_obs"] = (h.shown - h.obs) / h.obs
print("\nhist shown == my corrected mean:", bool(np.allclose(h.shown.dropna(), h.corrm.dropna(), atol=1e-3)))
print("shown vs observed-only: median rel diff", round(float(h.diff_vs_obs.median()), 4),
      "max", round(float(h.diff_vs_obs.max()), 4), "rows >5%:", int((h.diff_vs_obs > 0.05).sum()), "of", len(h))
print(h.sort_values("diff_vs_obs", ascending=False).head(6).to_string())
print("first season date in test data:", sv.date.min())

# demo services vs CSV
bad = 0
for sv_ in demo["services"]:
    row = sv[(sv.date == sv_["date"]) & (sv.meal == sv_["meal"])].iloc[0]
    for k, v in sv_["presence"].items():
        bad += abs(row[k] - v) > 1e-3
    for dd in sv_["dishes"]:
        pr = sg[(sg.date == sv_["date"]) & (sg.meal == sv_["meal"]) & (sg.dish_id == dd["id"])].iloc[0]
        bad += abs(pr.plan_kg - dd["chef_plan_kg"]) > 1e-3
print("\ndemo.json presence/chef-plan mismatches vs data/test:", bad)
dw = s[(s.date >= "2026-07-12") & (s.date <= "2026-07-18")]
print(dw[["date", "meal", "event", "child_share", "temperature", "occupancy", "in_house", "attendance", "present", "g_med"]]
      .round(3).to_string())
