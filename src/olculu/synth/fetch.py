"""Download the real-world inputs for the semi-synthetic seasons into data/raw/.

Usage: uv run python -m olculu.synth.fetch
The files are data, never executed. Sources and licences are listed in config/grounding.yaml.
"""

import urllib.request

from olculu.config import DATA_DIR

FILES = {
    "hotels.csv": "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-02-11/hotels.csv",
    "antalya_weather_2016.json": "https://archive-api.open-meteo.com/v1/archive?latitude=36.8969&longitude=30.7133"
                                 "&start_date=2016-04-01&end_date=2016-10-31&daily=temperature_2m_max&timezone=Europe%2FIstanbul",
    "antalya_weather_2024.json": "https://archive-api.open-meteo.com/v1/archive?latitude=36.8969&longitude=30.7133"
                                 "&start_date=2024-04-01&end_date=2024-10-31&daily=temperature_2m_max&timezone=Europe%2FIstanbul",
    "euro2016.txt": "https://raw.githubusercontent.com/openfootball/euro/master/2016--france/euro.txt",
    "euro2024.json": "https://raw.githubusercontent.com/openfootball/euro.json/master/2024/euro.json",
}


def main():
    out = DATA_DIR / "raw"
    out.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            (out / name).write_bytes(r.read())
        print("downloaded", name)


if __name__ == "__main__":
    main()
