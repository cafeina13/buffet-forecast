"""Audit 2 part 3: regenerate both seasons in memory (seed 42 / 43 + test_shift) and compare with data/."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from olculu.config import load_dishes, load_synthetic
from olculu.synth import apply_test_shift, generate
cfg, cat = load_synthetic(), load_dishes()
for name, c, seed in [("train", cfg, 42), ("test", apply_test_shift(cfg), 43)]:
    t = generate(c, cat, seed)
    for k in ["services", "servings"]:
        a = pd.read_csv(ROOT / "data" / name / f"{k}.csv")
        b = t[k].copy(); b["date"] = b["date"].astype(str)
        num = [col for col in a.select_dtypes("number").columns]
        same = len(a) == len(b) and np.allclose(a[num].to_numpy(float), b[num].to_numpy(float), equal_nan=True)
        print(name, k, len(a), len(b), "identical numerics:", same)
