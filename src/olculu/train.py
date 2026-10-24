"""Train both models on the training season, evaluate on the test season, export for the site.

Usage: uv run python -m olculu.train [--embeddings tfidf|berturk|minilm|category|st] [--name NAME]

The last weeks of the test season are the blind test. They are excluded here and
never used for the demo; they are opened once, at the very end, by a separate command.
"""

import argparse
import json
import shutil
from datetime import date, timedelta

import numpy as np
import pandas as pd

from olculu import embeddings
from olculu.config import CONFIG_DIR, DATA_DIR, MEALS, ROOT, WEB_DATA_DIR, load_dishes, load_synthetic
from olculu.export_web import quantile_models_json, write_json
from olculu.features import (DISH_FEATURES, GUEST_FEATURES, GUEST_FEATURES_OCC_ONLY, PRESENCE_COLS,
                             build_dish_frame, dish_static, prepare_services)
from olculu.models import (DISH_QUANTILES, fit_dish_models, fit_guest_models, interpolate_quantile,
                           oof_guest_median, pinball, predict_guests, predict_quantiles)
from olculu.newsvendor import round_up_half_unit, simulate
from olculu.synth.generator import arrival_profile, slot_times

RISKS = [0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05]
# Every setting the demo's risk slider can reach (5% to 30% in 1-point steps), for the season report.
SLIDER_RISKS = [round(r / 100, 2) for r in range(5, 31)]


def load_run(name):
    d = DATA_DIR / name
    return {t: pd.read_csv(d / f"{t}.csv") for t in ["services", "servings", "hourly"]}


def blind_cutoff(cfg, end=None):
    """First day of the blind weeks. `end` is the test season's last day (from data/test/meta.json)."""
    end = date.fromisoformat(str(end or cfg["season"]["end"]))
    return pd.Timestamp(end - timedelta(weeks=cfg["season"]["blind_test_weeks"]) + timedelta(days=1))


def run_meta(name):
    path = DATA_DIR / name / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"mode": "synthetic"}


def attach_guest_pred(frame, services, pred_col):
    return frame.merge(services[["date", "meal", pred_col]], on=["date", "meal"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings", choices=embeddings.BACKENDS, default="tfidf")
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()

    cfg, catalogue = load_synthetic(), load_dishes()
    dishes = {d["id"]: d for d in catalogue["dishes"]}
    unit_kg = {k: d["kg_per_unit"] for k, d in dishes.items()}
    refill = cfg["service"]["max_refill_share"]
    test_meta = run_meta("test")
    cutoff = blind_cutoff(cfg, test_meta.get("end"))

    train, test = load_run("train"), load_run("test")
    s_tr, s_te = prepare_services(train["services"]), prepare_services(test["services"])

    # Layer 1: guests per meal = presence rule x learned share who come to the buffet.
    guest_models = fit_guest_models(s_tr)
    s_tr["att_pred"] = oof_guest_median(s_tr)
    g_te, _ = predict_guests(guest_models, s_te)
    s_te["att_pred"] = g_te[:, 1]
    open_te = s_te["date"] < cutoff
    # Crossing is counted on the open weeks only; the blind weeks stay untouched in every metric.
    _, crossed_g = predict_guests(guest_models, s_te[open_te])

    occ_only = fit_guest_models(s_tr, GUEST_FEATURES_OCC_ONLY, target="attendance")
    occ_pred = occ_only[0.5].predict(s_te[GUEST_FEATURES_OCC_ONLY])
    guest_metrics = {
        "mae_full": float(np.mean(np.abs(s_te.loc[open_te, "att_pred"] - s_te.loc[open_te, "attendance"]))),
        "mae_occupancy_only": float(np.mean(np.abs(occ_pred[open_te] - s_te.loc[open_te, "attendance"]))),
        "coverage_q10_q90": float(np.mean((s_te.loc[open_te, "attendance"] >= g_te[open_te, 0])
                                          & (s_te.loc[open_te, "attendance"] <= g_te[open_te, 2]))),
    }
    # Simple baselines: how much of the gain is the presence rule, and how much the learned rate.
    meal_rate = (s_tr["attendance"] / s_tr["in_house"]).groupby(s_tr["meal"]).mean()
    present_rate = s_tr.groupby("meal")["rate"].mean()
    ote = s_te[open_te]
    guest_metrics["mae_in_house_times_meal_rate"] = float(
        (ote["in_house"] * ote["meal"].map(meal_rate) - ote["attendance"]).abs().mean())
    guest_metrics["mae_presence_rule_times_meal_rate"] = float(
        (ote["present"] * ote["meal"].map(present_rate) - ote["attendance"]).abs().mean())

    # Layer 2: per-dish consumption per 100 predicted guests.
    static, vecs = dish_static(catalogue, args.embeddings)
    neighbours = embeddings.nearest(vecs, 5)
    f_tr = attach_guest_pred(build_dish_frame(train["servings"], train["services"], train["hourly"],
                                              catalogue, static, neighbours), s_tr, "att_pred")
    f_te = attach_guest_pred(build_dish_frame(test["servings"], test["services"], test["hourly"],
                                              catalogue, static, neighbours), s_te, "att_pred")
    f_tr["ratio"] = f_tr["target_kg"] / f_tr["att_pred"] * 100
    dish_models = fit_dish_models(f_tr)

    ev = f_te[f_te["date"] < cutoff].copy()
    q_ratio, crossed_d = predict_quantiles(dish_models, ev[DISH_FEATURES])
    q_kg = q_ratio * ev["att_pred"].to_numpy()[:, None] / 100
    y = ev["true_demand_kg"].to_numpy()  # evaluation only

    coverage = {str(q): float(np.mean(y <= q_kg[:, i])) for i, q in enumerate(DISH_QUANTILES)}
    pinballs = {str(q): pinball(y, q_kg[:, i], q) for i, q in enumerate(DISH_QUANTILES)}
    mae_median = float(np.mean(np.abs(y - q_kg[:, 1])))

    # Dishes on days when they have no history yet in that meal: where the dish vector has to carry the forecast.
    no_hist = ev["hist_per100"].isna().to_numpy()
    groups_by_id = [d["group"] for d in catalogue["dishes"]]
    embedding_metrics = {
        "backend": args.embeddings,
        "no_history_rows": int(no_hist.sum()),
        "no_history_mae_median_kg": float(np.mean(np.abs(y[no_hist] - q_kg[no_hist, 1]))) if no_hist.any() else None,
        "no_history_mape_median": float(np.median(np.abs(y[no_hist] - q_kg[no_hist, 1]) / np.maximum(y[no_hist], 1e-9)))
        if no_hist.any() else None,
        "no_history_pinball_0.9": pinball(y[no_hist], q_kg[no_hist, 3], 0.9) if no_hist.any() else None,
        "with_history_mape_median": float(np.median(np.abs(y[~no_hist] - q_kg[~no_hist, 1]) / np.maximum(y[~no_hist], 1e-9))),
        # Share of each dish's 3 nearest neighbours that belong to its own dish group.
        "neighbours_same_group_top3": float(np.mean([[groups_by_id[j] == groups_by_id[i] for j in neighbours[i, :3]]
                                                     for i in range(len(groups_by_id))])),
    }

    units = ev["dish_id"].map(unit_kg).to_numpy()
    service_key = [ev["date"].to_numpy(), ev["meal"].to_numpy()]

    def services_with_stockout(stockout, mask=None):
        """Share of meal services in which at least one dish ran out."""
        so = pd.Series(stockout)
        keys = [k if mask is None else k[mask] for k in service_key]
        return float(so.groupby(keys).any().mean())

    chef = simulate(ev["plan_kg"].to_numpy(), y, units, refill)
    curve = []
    for r in RISKS:
        rec = round_up_half_unit(interpolate_quantile(q_kg, DISH_QUANTILES, 1 - r), units)
        sim = simulate(rec, y, units, refill)
        curve.append({"risk": r, "waste_kg": float(sim["waste"].sum()),
                      # Share of dish-servings that ran out after refills (not a share of meals).
                      "stockout_rate": float(sim["stockout"].mean()),
                      "services_with_stockout": services_with_stockout(sim["stockout"]),
                      # Share of dish-servings where demand exceeded the first cook (before refills).
                      "plan_short_rate": float(np.mean(y > rec))})
    chef_point = {"waste_kg": float(chef["waste"].sum()), "stockout_rate": float(chef["stockout"].mean()),
                  "services_with_stockout": services_with_stockout(chef["stockout"]),
                  "plan_short_rate": float(np.mean(y > ev["plan_kg"].to_numpy()))}

    # Naive variant (stock-outs ignored), for the censoring comparison.
    f_tr_naive = attach_guest_pred(build_dish_frame(train["servings"], train["services"], train["hourly"],
                                                    catalogue, static, neighbours, censoring=False), s_tr, "att_pred")
    f_tr_naive["ratio"] = f_tr_naive["target_kg"] / f_tr_naive["att_pred"] * 100
    naive_models = fit_dish_models(f_tr_naive)
    qn, _ = predict_quantiles(naive_models, ev[DISH_FEATURES])
    qn_kg = qn * ev["att_pred"].to_numpy()[:, None] / 100
    sold_prone = ev.groupby("dish_id")["stockout_time"].transform(lambda s: s.notna().mean()) > 0.15
    censoring_metrics = {
        "coverage_q90_aware_all": coverage["0.9"],
        "coverage_q90_naive_all": float(np.mean(y <= qn_kg[:, 3])),
        "coverage_q90_aware_soldout_prone": float(np.mean(y[sold_prone] <= q_kg[sold_prone, 3])),
        "coverage_q90_naive_soldout_prone": float(np.mean(y[sold_prone] <= qn_kg[sold_prone, 3])),
        "n_soldout_prone_rows": int(sold_prone.sum()),
    }

    # Season report: the whole open test season, model at three risk levels vs the chef's fixed plan,
    # plus a per-month breakdown so no single month stands in for the season.
    costs = cfg["costs"]["try_per_kg"]
    co2 = cfg["costs"]["co2_per_kg_waste"]
    price = np.array([costs[g] for g in ev["group"].to_numpy()])
    plan_chef = ev["plan_kg"].to_numpy()
    month = ev["date"].dt.month.to_numpy()

    def report_for(mask):
        chef_waste_try = float((chef["waste"][mask] * price[mask]).sum())
        out = {"chef": {
            "waste_kg": float(chef["waste"][mask].sum()), "stockout_rate": float(chef["stockout"][mask].mean()),
            "services_with_stockout": services_with_stockout(chef["stockout"][mask], mask),
            "plan_short_rate": float(np.mean(y[mask] > plan_chef[mask])),
            "production_kg": float(chef["production"][mask].sum())}, "levels": []}
        for r in SLIDER_RISKS:
            rec = round_up_half_unit(interpolate_quantile(q_kg[mask], DISH_QUANTILES, 1 - r), units[mask])
            sim = simulate(rec, y[mask], units[mask], refill)
            avoided = out["chef"]["waste_kg"] - float(sim["waste"].sum())
            out["levels"].append({
                "risk": r, "waste_kg": float(sim["waste"].sum()), "stockout_rate": float(sim["stockout"].mean()),
                "services_with_stockout": services_with_stockout(sim["stockout"], mask),
                "plan_short_rate": float(np.mean(y[mask] > rec)),
                "production_kg": float(sim["production"].sum()),
                # Cost of the avoided waste only; the cost of extra stock-outs is not netted out.
                "avoided_kg": avoided, "avoided_try": chef_waste_try - float((sim["waste"] * price[mask]).sum()),
                "avoided_co2_kg": avoided * co2,
            })
        return out

    all_rows = np.ones(len(ev), dtype=bool)
    report = {"period": f"{ev['date'].min().date()}..{ev['date'].max().date()}", "mode": test_meta["mode"],
              **report_for(all_rows),
              "months": [{"month": int(m), **report_for(month == m)} for m in sorted(set(month))]}

    # Method facts quoted on the site, computed here so every number has a record.
    sold_tr = f_tr["stockout_time"].notna()
    dinners = s_te[open_te & (s_te["meal"] == "dinner")]
    lunches = s_te[open_te & (s_te["meal"] == "lunch")]

    def event_effect(rows, flag):
        on, off = rows[GUEST_FEATURES].copy(), rows[GUEST_FEATURES].copy()
        on[flag], off[flag] = 1, 0
        return float(np.mean(guest_models[0.5].predict(on) / guest_models[0.5].predict(off) - 1))

    method_facts = {
        "censoring_factor_median": float((f_tr.loc[sold_tr, "target_kg"] / f_tr.loc[sold_tr, "observed_kg"]).median()),
        "true_over_observed_median": float((f_tr.loc[sold_tr, "true_demand_kg"] / f_tr.loc[sold_tr, "observed_kg"]).median()),
        "guest_effect_match": event_effect(dinners, "event_match"),
        "guest_effect_themed": event_effect(dinners, "event_themed"),
        "guest_effect_conference": event_effect(lunches, "event_conference"),
        "conference_lunches_in_training": int(((s_tr["meal"] == "lunch") & (s_tr["event"] == "conference")).sum()),
        # Share of dish-servings where demand exceeds the unrounded (1 - risk) quantile, before rounding up.
        "plan_short_unrounded": {str(r): float(np.mean(y > interpolate_quantile(q_kg, DISH_QUANTILES, 1 - r)))
                                 for r in SLIDER_RISKS},
    }

    metrics = {
        "note": "Synthetic data. Not evidence of accuracy.",
        "embeddings": args.embeddings,
        "guest": guest_metrics, "quantile_crossing_rows": {"guest": crossed_g, "dish": crossed_d},
        "dish": {"coverage": coverage, "pinball": pinballs, "mae_median_kg": mae_median, "rows": int(len(ev))},
        "embedding": embedding_metrics,
        "risk_curve": curve, "chef": chef_point, "censoring": censoring_metrics, "season_report": report,
        "method_facts": method_facts,
    }

    record = ROOT / "experiments" / f"{date.today().isoformat()}-{args.name}"
    record.mkdir(parents=True, exist_ok=True)
    shutil.copy(CONFIG_DIR / "synthetic.yaml", record / "synthetic.yaml")
    shutil.copy(CONFIG_DIR / "dishes.yaml", record / "dishes.yaml")
    (record / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    model_dir = record / "models"
    model_dir.mkdir(exist_ok=True)
    for prefix, models in [("guests", guest_models), ("dishes", dish_models)]:
        for q, m in models.items():
            m.booster_.save_model(str(model_dir / f"{prefix}_q{q}.txt"))
    f_te[f_te["date"] < cutoff].head(300)[DISH_FEATURES].to_csv(model_dir / "dish_rows_sample.csv", index=False)
    s_te.head(60)[GUEST_FEATURES].to_csv(model_dir / "guest_rows_sample.csv", index=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    # Ranges seen in training, so the demo's sliders stay where the models have data.
    support = {
        "temperature": [float(s_tr["temperature"].min()), float(s_tr["temperature"].max())],
        "child_share": [float(s_tr["child_share"].min()), float(s_tr["child_share"].max())],
        "capacity_guests": float(cfg["hotel"]["rooms"] * cfg["hotel"]["guests_per_room"]),
    }
    export(cfg, catalogue, guest_models, dish_models, s_te, f_te, cutoff, vecs, metrics, record.name, support)


def export(cfg, catalogue, guest_models, dish_models, s_te, f_te, cutoff, vecs, metrics, record_name, support):
    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_json(WEB_DATA_DIR / "model-guests.json", quantile_models_json(guest_models, GUEST_FEATURES), exact=True)
    write_json(WEB_DATA_DIR / "model-dishes.json", quantile_models_json(dish_models, DISH_FEATURES), exact=True)

    dishes = {d["id"]: d for d in catalogue["dishes"]}
    # Demo week: seven days of the open test season around a match night in July or August.
    open_s = s_te[(s_te["date"] < cutoff) & s_te["date"].dt.month.isin([7, 8])]
    matches = open_s[open_s["event"] == "match"]["date"].sort_values()
    anchor = matches.iloc[0] if len(matches) else open_s["date"].min()
    days = pd.date_range(anchor - pd.Timedelta(days=3), anchor + pd.Timedelta(days=3))

    services = []
    for d in days:
        for meal in MEALS:
            row = s_te[(s_te["date"] == d) & (s_te["meal"] == meal)].iloc[0]
            dishes_rows = f_te[(f_te["date"] == d) & (f_te["meal"] == meal)]
            services.append({
                "date": d.date().isoformat(), "meal": meal, "event": row["event"],
                "guest_x": [row[c] for c in GUEST_FEATURES],
                "presence": {c: row[c] for c in PRESENCE_COLS},
                "dishes": [{
                    "id": r["dish_id"], "x": [r[c] for c in DISH_FEATURES],
                    "chef_plan_kg": r["plan_kg"],
                } for _, r in dishes_rows.iterrows()],
            })

    mcfg = cfg["meals"]
    profiles = {}
    for meal in MEALS:
        normal = arrival_profile(mcfg[meal], mcfg["slot_minutes"])
        match = arrival_profile(mcfg[meal], mcfg["slot_minutes"], mcfg["match_dinner_shift_hours"],
                                mcfg["match_late_wave_share"])
        profiles[meal] = {"slots": list(slot_times(mcfg[meal], mcfg["slot_minutes"])),
                          "normal": list(normal), "match": list(match)}

    xy = embeddings.project_2d(vecs)
    ids = [d["id"] for d in catalogue["dishes"]]
    near = embeddings.nearest(vecs, 3)
    neighbour_names = {ids[i]: [dishes[ids[j]]["name"] for j in near[i]] for i in range(len(ids))}
    neighbour_ids = {ids[i]: [ids[j] for j in near[i]] for i in range(len(ids))}
    groups = catalogue["groups"]

    def heat_direction(d):
        """Kitchen expectation for hot days: +1 more demand, -1 less, 0 no clear direction.

        Taken from the catalogue's temperature sensitivity; used only to decide whether a
        temperature reason may be shown, never as a model input.
        """
        sens = d.get("temp_sensitivity", groups[d["group"]]["temp_sensitivity"])
        return 0 if abs(sens) < 0.5 else (1 if sens > 0 else -1)

    write_json(WEB_DATA_DIR / "demo.json", {
        "record": record_name,
        "hotel": "Kurgusal Demo Resort",
        "max_refill_share": cfg["service"]["max_refill_share"],
        "slot_minutes": mcfg["slot_minutes"],
        "co2_per_kg": cfg["costs"]["co2_per_kg_waste"],
        "support": support,
        "dishes": {k: {"name": d["name"], "group": d["group"], "unit": d["unit"],
                       "kg_per_unit": d["kg_per_unit"], "ingredients": d["ingredients"],
                       "new": d["freq"] == "new", "heat": heat_direction(d),
                       "neighbours": neighbour_names[k], "neighbour_ids": neighbour_ids[k]}
                   for k, d in dishes.items()},
        "map": [{"id": d["id"], "x": float(xy[i, 0]), "y": float(xy[i, 1])} for i, d in enumerate(catalogue["dishes"])],
        "profiles": profiles,
        "services": services,
        "metrics": {k: metrics[k] for k in ["risk_curve", "chef", "season_report", "guest", "censoring", "dish", "method_facts"]},
    })


if __name__ == "__main__":
    main()
