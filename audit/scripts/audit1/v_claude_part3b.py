"""Claude's verification of audit 1 part 3 findings F3 (map) and F10 (trivial guest baseline). Read-only."""
import numpy as np
import pandas as pd

from olculu import embeddings
from olculu.config import load_dishes, load_synthetic
from olculu.features import dish_static, prepare_services
from olculu.train import blind_cutoff, load_run

cat = load_dishes()
_, vecs = dish_static(cat, "tfidf")
near = embeddings.nearest(vecs, 3)
xy = embeddings.project_2d(vecs)
d2 = np.linalg.norm(xy[:, None] - xy[None], axis=2)
np.fill_diagonal(d2, np.inf)
rank2d = d2.argsort(axis=1).argsort(axis=1)  # 0 = closest on the map
ranks = np.array([[rank2d[i, j] + 1 for j in near[i]] for i in range(len(near))])
names = [d["name"] for d in cat["dishes"]]
print("F3 median 2-D rank of true top-3 neighbours:", np.median(ranks))
print("F3 share of true neighbours in 2-D top 5:", round(float((ranks <= 5).mean()), 2))
from sklearn.decomposition import PCA
print("F3 variance kept by 2-D PCA:", round(float(PCA(2).fit(vecs).explained_variance_ratio_.sum()), 3))
for i, d in enumerate(cat["dishes"]):
    if d["freq"] == "new":
        print(f"  {d['name']}: neighbours {[names[j] for j in near[i]]}, 2-D ranks {list(ranks[i])}")

# F10: trivial baseline = presence-free in_house x training per-meal attendance rate.
cfg = load_synthetic()
tr = prepare_services(load_run("train")["services"])
te = prepare_services(load_run("test")["services"])
te = te[te["date"] < blind_cutoff(cfg)]
rate = (tr["attendance"] / tr["in_house"]).groupby(tr["meal"]).mean()
pred = te["in_house"] * te["meal"].map(rate)
print("F10 trivial in_house x meal-rate MAE:", round(float((pred - te["attendance"]).abs().mean()), 2))
prate = tr.groupby("meal")["rate"].mean()
pred2 = te["present"] * te["meal"].map(prate)
print("F4 presence rule x constant meal rate MAE:", round(float((pred2 - te["attendance"]).abs().mean()), 2))
