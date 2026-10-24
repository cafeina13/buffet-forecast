"""The test season is a separate run and the blind weeks are held out."""

import pandas as pd

from olculu.config import load_dishes, load_synthetic
from olculu.synth import apply_test_shift, generate
from olculu.train import blind_cutoff


def test_test_run_uses_shifted_parameters():
    cfg = load_synthetic()
    shifted = apply_test_shift(cfg)
    assert shifted["occupancy"]["peak"] != cfg["occupancy"]["peak"]
    assert shifted["temperature"]["august_mean"] != cfg["temperature"]["august_mean"]


def test_runs_differ_with_seed():
    cfg, catalogue = load_synthetic(), load_dishes()
    cfg["season"]["end"] = "2026-04-10"
    a = generate(cfg, catalogue, seed=1)["services"]["attendance"]
    b = generate(apply_test_shift(cfg), catalogue, seed=2)["services"]["attendance"]
    assert not a.equals(b)


def test_blind_cutoff_is_three_weeks_before_season_end():
    cfg = load_synthetic()
    cutoff = blind_cutoff(cfg)
    end = pd.Timestamp(cfg["season"]["end"])
    assert (end - cutoff).days + 1 == 7 * cfg["season"]["blind_test_weeks"]


def test_every_parameter_has_a_source():
    from olculu.config import load_synthetic_raw

    def leaves(node, path=""):
        if isinstance(node, dict) and "value" in node:
            yield path, node
        elif isinstance(node, dict):
            for k, v in node.items():
                yield from leaves(v, f"{path}.{k}")

    for path, leaf in leaves(load_synthetic_raw()):
        assert leaf.get("source"), f"{path} has no source"
