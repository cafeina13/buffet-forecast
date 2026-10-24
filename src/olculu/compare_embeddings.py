"""Compare dish-vector backends on neighbour quality (no training).

Usage: uv run python -m olculu.compare_embeddings
For each backend: the share of every dish's 3 nearest neighbours that are in its own dish group, and the
neighbours of the dishes that enter the menu mid-season. Writes experiments/<date>-embedding-neighbours/.
The forecast side (error on dishes without history) comes from `olculu.train --embeddings <backend>`.
"""

import json
from datetime import date

import numpy as np

from olculu import embeddings
from olculu.config import ROOT, load_dishes

BACKENDS = ["tfidf", "berturk", "minilm", "category"]


def main():
    dishes = load_dishes()["dishes"]
    groups = [d["group"] for d in dishes]
    names = [d["name"] for d in dishes]
    new = [i for i, d in enumerate(dishes) if d["freq"] == "new"]
    out = {}
    for backend in BACKENDS:
        near = embeddings.nearest(embeddings.embed(dishes, backend), 3)
        same = float(np.mean([[groups[j] == groups[i] for j in near[i]] for i in range(len(dishes))]))
        out[backend] = {"neighbours_same_group_top3": same,
                        "new_dishes": {names[i]: [names[j] for j in near[i]] for i in new}}
        print(f"{backend:9s} same group in top-3: {same:.1%}")
        for i in new:
            print(f"    {names[i]:22s} -> {', '.join(names[j] for j in near[i])}")
    record = ROOT / "experiments" / f"{date.today().isoformat()}-embedding-neighbours"
    record.mkdir(parents=True, exist_ok=True)
    (record / "neighbours.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
