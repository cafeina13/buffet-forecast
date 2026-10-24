"""Loading of the YAML configs. Every parameter is stored as {value, source, type}."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
WEB_DATA_DIR = ROOT / "web" / "data"

MEALS = ["breakfast", "lunch", "dinner"]
NATIONALITIES = ["DE", "RU", "UK", "TR", "OTHER"]


def _strip(node):
    """Replace {value, source, type} leaves with their value."""
    if isinstance(node, dict):
        if "value" in node and "source" in node:
            return node["value"]
        return {k: _strip(v) for k, v in node.items()}
    return node


def load_synthetic(path: Path = CONFIG_DIR / "synthetic.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return _strip(yaml.safe_load(f))


def load_synthetic_raw(path: Path = CONFIG_DIR / "synthetic.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dishes(path: Path = CONFIG_DIR / "dishes.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
