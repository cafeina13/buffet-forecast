"""Real-world inputs for a semi-synthetic season.

What is real, per day of the season:
  - guests staying, arriving and departing, and the share of children: from the resort hotel (H1) of the
    Hotel Booking Demand dataset (Antonio, Almeida & Nunes 2019, Data in Brief 22:41-49, CC BY 4.0),
    non-cancelled bookings, scaled to our resort's size;
  - the day's maximum temperature in Antalya: Open-Meteo historical archive (ERA5-based);
  - match nights: Euro 2016 / Euro 2024 matches of Türkiye, Germany or England kicking off in the evening
    (openfootball fixture lists).
Everything else (dish demand, tours, à la carte, nationality mix, themed nights, conferences) stays synthetic.

Downloaded files live in data/raw/ (not in git); `python -m olculu.synth.fetch` downloads them.
"""

import json
import re
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
     "November", "December"], start=1)}
TEAMS = ("Turkey", "Germany", "England")


def booking_days(hotels_csv: Path, bookings_year: int, calendar_year: int, start: date, end: date) -> pd.DataFrame:
    """Daily guests staying, arriving and departing at the resort hotel, placed on the target calendar.

    A booking from `bookings_year` is moved to the same month and day of `calendar_year`.
    """
    h = pd.read_csv(hotels_csv)
    h = h[(h["hotel"] == "Resort Hotel") & (h["is_canceled"] == 0)].copy()
    h["children"] = h["children"].fillna(0)
    h["arrival"] = pd.to_datetime(dict(
        year=h["arrival_date_year"], month=h["arrival_date_month"].map(MONTHS), day=h["arrival_date_day_of_month"]))
    h["nights"] = h["stays_in_weekend_nights"] + h["stays_in_week_nights"]
    h = h[h["nights"] > 0]
    h["guests"] = h["adults"] + h["children"] + h["babies"]
    h["kids"] = h["children"] + h["babies"]

    days = pd.date_range(pd.Timestamp(bookings_year, start.month, start.day), pd.Timestamp(bookings_year, end.month, end.day))
    rows = []
    for d in days:
        staying = h[(h["arrival"] <= d) & (h["arrival"] + pd.to_timedelta(h["nights"], unit="D") > d)]
        arriving = h[h["arrival"] == d]
        leaving = h[h["arrival"] + pd.to_timedelta(h["nights"], unit="D") == d]
        rows.append({
            "date": date(calendar_year, d.month, d.day),
            "in_house_raw": float(staying["guests"].sum()),
            "arrivals_raw": float(arriving["guests"].sum()),
            "departures_raw": float(leaving["guests"].sum()),
            "child_share": float(staying["kids"].sum() / max(staying["guests"].sum(), 1)),
        })
    return pd.DataFrame(rows)


def daily_max_temperature(weather_json: Path) -> dict:
    w = json.loads(Path(weather_json).read_text(encoding="utf-8"))["daily"]
    return {date.fromisoformat(t): float(v) for t, v in zip(w["time"], w["temperature_2m_max"]) if v is not None}


def _evening(kickoff_cest: str | None, evening_from: int) -> bool:
    return kickoff_cest is not None and int(kickoff_cest.split(":")[0]) >= evening_from


def match_nights_json(euro_json: Path, evening_from: int) -> set:
    """openfootball euro.json (2020+): one record per match with date and CEST kick-off."""
    ms = json.loads(Path(euro_json).read_text(encoding="utf-8"))["matches"]
    return {date.fromisoformat(m["date"]) for m in ms
            if (m["team1"] in TEAMS or m["team2"] in TEAMS) and _evening(m.get("time"), evening_from)}


def match_nights_txt(euro_txt: Path, year: int, evening_from: int) -> set:
    """openfootball euro.txt (older tournaments): date lines, then matches; a missing time repeats the last one."""
    out, day, kickoff = set(), None, None
    for line in Path(euro_txt).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(Jun|Jul)\s+(\d{1,2})\s*$", line.strip())
        if m:
            day = date(year, 6 if m.group(1) == "Jun" else 7, int(m.group(2)))
            kickoff = None
            continue
        if day is None or "@" not in line:
            continue
        t = re.match(r"^\s*(\d{2}:\d{2})\s", line)
        if t:
            kickoff = t.group(1)
        if any(team in line for team in TEAMS) and _evening(kickoff, evening_from):
            out.add(day)
    return out


def season_inputs(cfg: dict, raw_dir: Path, which: str) -> dict:
    """Per-date real inputs for the 'train' or 'test' season, keyed by date."""
    g = cfg["grounding"][which]
    start, end = date.fromisoformat(g["start"]), date.fromisoformat(g["end"])
    b = booking_days(raw_dir / "hotels.csv", g["bookings_year"], g["calendar_year"], start, end)
    # Scale the hotel to our resort: its busiest night of the season becomes the configured peak occupancy.
    capacity = cfg["hotel"]["rooms"] * cfg["hotel"]["guests_per_room"]
    scale = capacity * cfg["grounding"]["peak_occupancy"] / b["in_house_raw"].max()
    temps = daily_max_temperature(raw_dir / g["weather_file"])
    evening = cfg["grounding"]["evening_kickoff_from_cest"]
    if g["fixtures_file"].endswith(".json"):
        matches = match_nights_json(raw_dir / g["fixtures_file"], evening)
    else:
        matches = match_nights_txt(raw_dir / g["fixtures_file"], g["calendar_year"], evening)
    out = {}
    for _, r in b.iterrows():
        d = r["date"]
        out[d] = {
            "in_house": r["in_house_raw"] * scale,
            "arrivals": r["arrivals_raw"] * scale,
            "departures": r["departures_raw"] * scale,
            "child_share": r["child_share"],
            "temperature": temps[d],
            "match": d in matches,
        }
    return {"start": start, "end": end, "days": out, "scale": scale, "match_nights": sorted(matches)}
