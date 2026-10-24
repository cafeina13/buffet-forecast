"""Export trained LightGBM models and demo services as JSON for the static site.

The browser evaluates the trees itself (web/js/model.js), so visitors run the real
model. Tree format per tree, flattened in pre-order:
  f: split feature index (-1 for a leaf), t: threshold, l/r: child indices,
  v: leaf value, c: training-sample count (used by TreeSHAP), d: 1 if missing goes left,
  m: missing type (0 none, 1 zero, 2 NaN)
"""

import json

import numpy as np

MISSING = {"None": 0, "Zero": 1, "NaN": 2}


def _flatten(node, out):
    i = len(out["f"])
    for k in out:
        out[k].append(0)
    if "leaf_value" in node:
        out["f"][i], out["v"][i], out["c"][i] = -1, float(node["leaf_value"]), int(node.get("leaf_count", 0))
        out["l"][i] = out["r"][i] = -1
        return i
    if node.get("decision_type", "<=") != "<=":
        raise ValueError("only numerical splits are supported")
    out["f"][i] = int(node["split_feature"])
    out["t"][i] = float(node["threshold"])
    out["c"][i] = int(node["internal_count"])
    out["d"][i] = 1 if node["default_left"] else 0
    out["m"][i] = MISSING[node["missing_type"]]
    out["l"][i] = _flatten(node["left_child"], out)
    out["r"][i] = _flatten(node["right_child"], out)
    return i


def booster_json(model) -> dict:
    dump = model.booster_.dump_model()
    trees = []
    for info in dump["tree_info"]:
        out = {k: [] for k in ["f", "t", "l", "r", "v", "c", "d", "m"]}
        _flatten(info["tree_structure"], out)
        # Thresholds keep full precision so no row can land on the other side of a split.
        out["v"] = [round(x, 6) for x in out["v"]]
        trees.append(out)
    return {"trees": trees}


def quantile_models_json(models: dict, features: list[str]) -> dict:
    return {"features": features, "quantiles": sorted(models),
            "models": [booster_json(models[q]) for q in sorted(models)]}


def clean(x):
    """Make numpy / pandas values JSON-safe; NaN becomes null."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if np.isnan(x) else round(float(x), 5)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def write_json(path, obj, exact=False):
    """exact=True skips rounding (used for model files)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj if exact else clean(obj), f, ensure_ascii=False, separators=(",", ":"))
