"""Generate the training season (given seed) and the test season (seed + 1, shifted parameters).

Usage: uv run python -m olculu.synth --seed 42
"""

import argparse

from olculu.config import DATA_DIR, load_dishes, load_synthetic
from olculu.synth import apply_test_shift, generate


def write_run(name, tables):
    out = DATA_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    for table, df in tables.items():
        df.to_csv(out / f"{table}.csv", index=False)
    print(f"{name}: " + ", ".join(f"{k}={len(v)} rows" for k, v in tables.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg, catalogue = load_synthetic(), load_dishes()
    write_run("train", generate(cfg, catalogue, seed=args.seed))
    write_run("test", generate(apply_test_shift(cfg), catalogue, seed=args.seed + 1))


if __name__ == "__main__":
    main()
