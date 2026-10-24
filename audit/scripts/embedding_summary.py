"""Summarise the per-backend training records written by `olculu.train --embeddings <b> --name emb-<b>`."""
import glob
import json

for b in ["tfidf", "berturk", "minilm", "category"]:
    paths = sorted(glob.glob(f"experiments/*-emb-{b}/metrics.json"))
    if not paths:
        print(b, "no record")
        continue
    m = json.load(open(paths[-1], encoding="utf-8"))
    e = m["embedding"]
    r = [c for c in m["risk_curve"] if c["risk"] == 0.1][0]
    print(f"{b:9s} no-history rows {e['no_history_rows']}, MAPE {e['no_history_mape_median']:.1%}, "
          f"MAE {e['no_history_mae_median_kg']:.2f} kg, pinball90 {e['no_history_pinball_0.9']:.3f} | "
          f"with-history MAPE {e['with_history_mape_median']:.1%} | all MAE {m['dish']['mae_median_kg']:.3f} | "
          f"q95 cov {m['dish']['coverage']['0.95']:.3f} | 10%: waste {r['waste_kg'] / 1000:.1f} t, "
          f"out {r['stockout_rate']:.1%} | neighbours same group {e['neighbours_same_group_top3']:.1%}")
