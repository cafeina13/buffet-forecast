"""Generate the training season (given seed) and the test season (seed + 1, shifted parameters).

Usage: uv run python -m olculu.synth --seed 42 [--grounded]

--grounded builds semi-synthetic seasons: real resort bookings, Antalya weather and Euro match nights
(config/grounding.yaml; run `python -m olculu.synth.fetch` first). Training season 2016, test season 2024.
"""

import argparse
import json

from olculu.config import DATA_DIR, load_dishes, load_grounding, load_synthetic
from olculu.synth import apply_test_shift, generate
from olculu.synth.grounding import season_inputs


def write_run(name, tables, meta):
    out = DATA_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    for table, df in tables.items():
        df.to_csv(out / f"{table}.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    print(f"{name}: " + ", ".join(f"{k}={len(v)} rows" for k, v in tables.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grounded", action="store_true")
    args = parser.parse_args()

    cfg, catalogue = load_synthetic(), load_dishes()
    for name, seed, run_cfg in [("train", args.seed, cfg), ("test", args.seed + 1, apply_test_shift(cfg))]:
        grounding = None
        meta = {"mode": "synthetic", "start": cfg["season"]["start"], "end": cfg["season"]["end"]}
        if args.grounded:
            run_cfg = {**run_cfg, **load_grounding()}
            grounding = season_inputs(run_cfg, DATA_DIR / "raw", name)
            meta = {"mode": "grounded", "start": grounding["start"], "end": grounding["end"],
                    "scale": grounding["scale"], "match_nights": grounding["match_nights"]}
        write_run(name, generate(run_cfg, catalogue, seed=seed, grounding=grounding), meta)


if __name__ == "__main__":
    main()
