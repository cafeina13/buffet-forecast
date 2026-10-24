"""From a chosen risk to a quantity, and service simulation for evaluation."""

import numpy as np


def risk_to_level(risk: float) -> float:
    """Risk that this dish runs out -> quantile of demand to cook for."""
    return 1.0 - risk


def critical_ratio(cu: float, co: float) -> float:
    """Newsvendor optimal service level Cu / (Cu + Co)."""
    return cu / (cu + co)


def round_up_half_unit(kg, unit_kg):
    return np.maximum(unit_kg * 0.5, np.ceil(np.asarray(kg) / unit_kg * 2) / 2 * unit_kg)


def simulate(plan_kg, true_kg, unit_kg, max_refill_share):
    """Same refill rule as the synthetic chef: refills in whole units, capped at a share of the plan."""
    plan_kg, true_kg, unit_kg = map(np.asarray, (plan_kg, true_kg, unit_kg))
    cap = plan_kg * (1 + max_refill_share)
    refilled = np.minimum(cap, plan_kg + np.ceil(np.maximum(true_kg - plan_kg, 0) / unit_kg) * unit_kg)
    production = np.where(true_kg <= plan_kg, plan_kg, refilled)
    consumed = np.minimum(true_kg, production)
    return {
        "production": production,
        "waste": production - consumed,
        "stockout": true_kg > production,
    }
