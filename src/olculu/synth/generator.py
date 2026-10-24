"""Synthetic season generator for one fictional resort.

Produces three tables:
  services: one row per (date, meal) with guest counts, events and temperature
  servings: one row per (date, meal, dish) with plan, production, observed
            consumption, true demand (evaluation only), stock-out time, leftovers
  hourly:   arrival share per 15-minute slot for each service
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd

from olculu.config import MEALS, NATIONALITIES


def _bump(t, centre, width):
    return np.exp(-0.5 * ((t - centre) / width) ** 2)


def _seasonal_01(t, centre, width):
    """Bell curve rescaled so that day 0 maps to 0 and the centre to 1."""
    b0 = _bump(0, centre, width)
    return (_bump(t, centre, width) - b0) / (1 - b0)


def apply_test_shift(cfg: dict) -> dict:
    """Return a copy of the config with the test-run parameter shifts applied."""
    import copy

    c = copy.deepcopy(cfg)
    s = c["test_shift"]
    c["occupancy"]["peak"] = s["occupancy_peak"]
    c["guest_mix"]["child_share_summer"] = s["child_share_summer"]
    c["temperature"]["august_mean"] = s["temperature_august_mean"]
    c["activities"]["excursion_share"] = s["excursion_share"]
    return c


def slot_times(meal_cfg: dict, slot_minutes: int) -> np.ndarray:
    step = slot_minutes / 60
    return np.arange(meal_cfg["start"], meal_cfg["end"] - 1e-9, step)


def arrival_profile(meal_cfg, slot_minutes, shift=0.0, late_wave=0.0):
    """Share of the service's guests arriving in each slot."""
    times = slot_times(meal_cfg, slot_minutes) + slot_minutes / 120
    w = _bump(times, meal_cfg["peak"] + shift, meal_cfg["spread"])
    w = w / w.sum()
    if late_wave > 0:
        late = _bump(times, meal_cfg["end"] - 0.4, 0.25)
        w = (1 - late_wave) * w + late_wave * late / late.sum()
    return w


def _menu_for(dishes, meal, d: date, rng):
    menu = []
    for dish in dishes:
        if meal not in dish["meals"]:
            continue
        freq = dish["freq"]
        if freq == "daily":
            on = True
        elif freq == "new":
            on = d >= date.fromisoformat(dish["new_from"]) and rng.random() < dish["new_prob"]
        elif "weekly" in freq:
            on = d.weekday() in freq["weekly"]
        else:
            on = rng.random() < freq["sparse"]
        if on:
            menu.append(dish)
    return menu


def generate(cfg: dict, catalogue: dict, seed: int, grounding: dict | None = None) -> dict[str, pd.DataFrame]:
    """grounding: real per-day inputs from synth.grounding.season_inputs (semi-synthetic mode), or None."""
    rng = np.random.default_rng(seed)
    dishes = catalogue["dishes"]
    groups = catalogue["groups"]

    if grounding:
        start, end = grounding["start"], grounding["end"]
    else:
        start = date.fromisoformat(cfg["season"]["start"])
        end = date.fromisoformat(cfg["season"]["end"])
    n_days = (end - start).days + 1

    hotel, occ, mix = cfg["hotel"], cfg["occupancy"], cfg["guest_mix"]
    arr, act, att, ev = cfg["arrivals"], cfg["activities"], cfg["attendance"], cfg["events"]
    tcfg, mcfg, dem, chef, svc = cfg["temperature"], cfg["meals"], cfg["demand"], cfg["chef"], cfg["service"]
    slot_min = mcfg["slot_minutes"]

    services, servings, hourly = [], [], []
    # Chef memory: observed kg per attending guest, per (dish, meal).
    history: dict[tuple[str, str], list[float]] = {}
    last_attendance: dict[str, float] = {}

    for t in range(n_days):
        d = start + timedelta(days=t)
        occupancy = occ["low"] + (occ["peak"] - occ["low"]) * _bump(t, occ["peak_day_of_season"], occ["width_days"])
        occupancy = float(np.clip(occupancy + rng.normal(0, occ["daily_noise_sd"]), 0.25, 1.0))
        in_house = int(round(hotel["rooms"] * occupancy * hotel["guests_per_room"]))

        summer = _bump(t, 110, 30)
        shares = np.array(mix["base_shares"], dtype=float)
        shares[3] += mix["tr_summer_boost"] * summer
        shares = np.clip(shares + rng.normal(0, mix["share_noise_sd"], len(shares)), 0.01, None)
        shares /= shares.sum()
        child_share = mix["child_share_base"] + (mix["child_share_summer"] - mix["child_share_base"]) * summer

        arrivals = in_house / hotel["avg_stay_nights"] * rng.lognormal(0, 0.15)
        departures = in_house / hotel["avg_stay_nights"] * rng.lognormal(0, 0.15)
        real = grounding["days"].get(d) if grounding else None
        if real:
            # Real resort pattern (scaled) replaces the synthetic occupancy, arrivals, departures and children.
            in_house = int(round(real["in_house"]))
            occupancy = min(1.0, in_house / (hotel["rooms"] * hotel["guests_per_room"]))
            arrivals, departures = real["arrivals"], real["departures"]
            child_share = real["child_share"]
        night_arrivals = arrivals * arr["night_arrival_share"] * rng.lognormal(0, 0.2)
        day_arrivals = arrivals - night_arrivals
        early_dep = departures * arr["early_departure_share"]
        very_early_dep = departures * arr["very_early_departure_share"]
        excursions = in_house * max(0.0, rng.normal(act["excursion_share"], act["excursion_share_sd"]))
        alacarte = in_house * max(0.0, rng.normal(act["alacarte_share"], act["alacarte_share_sd"]))

        r = rng.random()
        match = r < ev["match_prob"]
        themed = (not match) and r < ev["match_prob"] + ev["themed_night_prob"]
        conference = rng.random() < ev["conference_prob"]

        temp = tcfg["april_mean"] + (tcfg["august_mean"] - tcfg["april_mean"]) * _seasonal_01(t, 122, 90)
        temp = float(temp + rng.normal(0, tcfg["noise_sd"]))
        if real:
            # Real match nights and real temperature; themed nights stay random on the other evenings.
            match = real["match"]
            themed = (not match) and rng.random() < ev["themed_night_prob"]
            temp = real["temperature"]

        for meal in MEALS:
            if meal == "breakfast":
                eligible = in_house - very_early_dep
                first_meal = night_arrivals * 0.9
                event = "none"
                effect = 0.0
            elif meal == "lunch":
                eligible = in_house - early_dep - excursions
                first_meal = day_arrivals * 0.25
                event = "conference" if conference else "none"
                effect = ev["conference_lunch_attendance"] if conference else 0.0
            else:
                eligible = in_house - night_arrivals - alacarte
                first_meal = day_arrivals * 0.7
                event = "match" if match else ("themed" if themed else "none")
                effect = {"match": ev["match_dinner_attendance"], "themed": ev["themed_dinner_attendance"]}.get(event, 0.0)

            attendance = eligible * att[meal] * (1 + effect) * (1 + rng.normal(0, att["noise_sd"]))
            attendance = int(max(20, round(attendance)))
            first_share = float(min(0.5, first_meal / attendance))

            shift = mcfg["match_dinner_shift_hours"] if event == "match" else 0.0
            wave = mcfg["match_late_wave_share"] if event == "match" else 0.0
            profile = arrival_profile(mcfg[meal], slot_min, shift, wave)
            profile = profile * rng.lognormal(0, 0.08, len(profile))
            profile /= profile.sum()
            times = slot_times(mcfg[meal], slot_min)
            for st, share in zip(times, profile):
                hourly.append({"date": d, "meal": meal, "slot": float(st), "share": float(share)})
            cum = np.cumsum(profile)

            services.append({
                "date": d, "meal": meal, "weekday": d.weekday(),
                "occupancy": occupancy, "in_house": in_house, "attendance": attendance,
                "child_share": child_share, "first_meal_share": first_share,
                **{f"nat_{n}": float(s) for n, s in zip(NATIONALITIES, shares)},
                "arrivals_night": night_arrivals, "arrivals_day": day_arrivals,
                "departures_early": early_dep, "departures_very_early": very_early_dep,
                "excursions": excursions, "alacarte": alacarte,
                "event": event, "temperature": temp,
            })

            menu = _menu_for(dishes, meal, d, rng)
            group_counts = pd.Series([m["group"] for m in menu]).value_counts().to_dict()
            chef_attendance = last_attendance.get(meal, attendance) * rng.lognormal(0, 0.05)

            for dish in menu:
                aff = sum(s * dish.get("affinity", {}).get(n, 1.0) for n, s in zip(NATIONALITIES, shares))
                aff = aff ** dem["nationality_affinity_exponent"]
                child = 1 + dem["child_effect"] * child_share / 0.2
                first = 1 + dem["first_meal_effect"] * first_share / 0.1
                sens = dish.get("temp_sensitivity", groups[dish["group"]]["temp_sensitivity"])
                temp_f = np.exp(sens * dem["temperature_effect_per_deg"] * (temp - dem["temperature_ref"]))
                n_similar = group_counts.get(dish["group"], 1) - 1
                comp = 1 / (1 + dem["competition_strength"] * n_similar)
                portion_kg = dish["portion_g"] / 1000
                mean_kg = attendance * dish["take"] * portion_kg * dem["meal_scale"][meal] * aff * child * first * temp_f * comp
                mean_portions = max(mean_kg / portion_kg, 0.5)
                size = dem["nb_dispersion"]
                portions = rng.negative_binomial(size, size / (size + mean_portions))
                true_kg = portions * portion_kg

                key = (dish["id"], meal)
                past = history.get(key, [])[-chef["history_services"]:]
                if past:
                    per_guest = float(np.mean(past))
                else:
                    per_guest = dish["take"] * portion_kg * rng.lognormal(0, 0.2)
                buffer = rng.uniform(chef["buffer_low"], chef["buffer_high"])
                plan = per_guest * chef_attendance * buffer * rng.lognormal(0, chef["plan_noise_sd"])
                unit = dish["kg_per_unit"]
                plan = max(unit * 0.5, round(plan / unit * 2) / 2 * unit)

                cap = plan * (1 + svc["max_refill_share"])
                if true_kg <= plan:
                    production = plan
                else:
                    production = min(cap, plan + np.ceil((true_kg - plan) / unit) * unit)
                observed = min(true_kg, production)
                stockout_time = None
                if true_kg > production:
                    frac = production / true_kg
                    idx = int(np.searchsorted(cum, frac))
                    idx = min(idx, len(times) - 1)
                    stockout_time = float(times[idx] + slot_min / 60)
                leftover = (production - observed) * (1 + rng.uniform(-svc["scale_noise"], svc["scale_noise"]))

                history.setdefault(key, []).append(observed / attendance)
                servings.append({
                    "date": d, "meal": meal, "dish_id": dish["id"], "group": dish["group"],
                    "plan_kg": plan, "production_kg": production, "observed_kg": observed,
                    "true_demand_kg": true_kg, "stockout_time": stockout_time,
                    "leftover_kg": max(0.0, leftover),
                })
            last_attendance[meal] = attendance

    return {
        "services": pd.DataFrame(services),
        "servings": pd.DataFrame(servings),
        "hourly": pd.DataFrame(hourly),
    }
