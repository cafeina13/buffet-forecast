"""The real-world inputs are read as documented (skipped when data/raw/ has not been fetched)."""

from datetime import date

import pytest

from olculu.config import DATA_DIR, load_grounding, load_synthetic
from olculu.synth.grounding import match_nights_json, match_nights_txt, season_inputs

RAW = DATA_DIR / "raw"
pytestmark = pytest.mark.skipif(not (RAW / "hotels.csv").exists(), reason="run python -m olculu.synth.fetch first")


def test_euro_2016_evening_matches_of_turkey_germany_england():
    nights = match_nights_txt(RAW / "euro2016.txt", 2016, 18)
    # Kick-off 18:00 CEST or later; the 15:00 matches (Turkey-Croatia, England-Wales) are not dinner-time.
    assert date(2016, 6, 12) in nights      # Germany-Ukraine 21:00 (Turkey-Croatia 15:00 the same day)
    assert date(2016, 6, 17) in nights      # Spain-Turkey 21:00
    assert date(2016, 6, 21) in nights      # Czech Republic-Turkey, time inherited from 21:00
    assert date(2016, 6, 16) in nights      # Germany-Poland 21:00 (England-Wales 15:00 the same day)
    assert len(nights) == 10


def test_euro_2024_turkey_quarter_final_is_a_match_night():
    nights = match_nights_json(RAW / "euro2024.json", 18)
    assert date(2024, 7, 6) in nights       # Netherlands-Turkey 21:00 CEST
    assert date(2024, 7, 2) in nights       # Austria-Turkey 21:00 CEST


def test_bookings_scaled_to_the_configured_peak():
    cfg = {**load_synthetic(), **load_grounding()}
    g = season_inputs(cfg, RAW, "train")
    capacity = cfg["hotel"]["rooms"] * cfg["hotel"]["guests_per_room"]
    peak = max(v["in_house"] for v in g["days"].values())
    assert abs(peak / capacity - cfg["grounding"]["peak_occupancy"]) < 1e-9
    assert all(0 <= v["child_share"] <= 1 for v in g["days"].values())
