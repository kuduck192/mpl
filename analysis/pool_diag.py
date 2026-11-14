#!/usr/bin/env python3
"""Student-pool diagnostic: is the pool diverse enough for min / mean / max (O / E / P) over
the pool to give different feedback?  Per run: trajectory of the pool statistics at a few
iterations, medians over the second half of training, and the end-of-run student finetune.

  python analysis/pool_diag.py results/fmnist-ssl-pool-diag

Reading guide
  pool_cos_pair       cosine between two members' feedback (same batch); compare with
  pool_cos_aug_ref    the same member on a second strong view (augmentation noise only)
  pool_cos_max_mean   P vs E; near 1 means pessimism over the pool changes nothing
  pool_norm_mean_rel  |mean of members| / mean |member|: 1 if members agree, 1/sqrt(k) if orthogonal
  pool_ll_spread      (max - min) / mean of the members' held-out labeled loss after their step
  pool_worst_switch   how often the worst member changes between logged steps
  pool_dis_val        pairwise disagreement of the members' (EMA) predictions on the val set
"""
import glob
import json
import os
import sys

import numpy as np

KEYS = ["pool_cos_pair", "pool_cos_aug_ref", "pool_cos_max_mean", "pool_cos_min_mean", "pool_cos_max_min",
        "pool_cos_cvar_mean", "pool_norm_mean_rel", "pool_ll_mean", "pool_ll_spread", "pool_worst_switch",
        "pool_dis_val", "pool_lab_loss_min", "pool_lab_loss_max", "test_ema", "stu_test_ema", "pool_ens_test",
        "pool_test_ema_min", "pool_test_ema_max"]


def main():
    files = sorted(f for d in sys.argv[1:] for f in glob.glob(os.path.join(d, "**", "*.json"), recursive=True))
    for f in files:
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if "hist" not in r or not r["hist"] or "pool_cos_pair" not in r["hist"][-1]:
            continue
        h = r["hist"]
        a = r["args"]
        print(f"\n## {a['name']}  (k={a['n_stu']}, agg={a['pool_agg']}, aug={a['pool_aug']}, bag={a['pool_bag']}, "
              f"{r['minutes']:.0f} min)\n")
        its = [x["iter"] for x in h]
        pick = [i for i in (1000, 2000, 5000, 10000, 15000, 20000) if i in its]
        print("| stat | " + " | ".join(f"{i // 1000}k" for i in pick) + " | median 2nd half |")
        print("|---|" + "---|" * (len(pick) + 1))
        late = [x for x in h if x["iter"] > its[-1] / 2]
        for k in KEYS:
            if not any(k in x for x in h):
                continue
            row = [h[its.index(i)].get(k, np.nan) for i in pick]
            med = np.nanmedian([x.get(k, np.nan) for x in late])
            fmt = (lambda v: f"{100 * v:.2f}") if ("test" in k) else (lambda v: f"{v:.3g}")
            print(f"| {k} | " + " | ".join(fmt(v) for v in row) + f" | {fmt(med)} |")
        worst = {j: np.mean([x.get(f"pool_worst_is_{j}", np.nan) for x in late]) for j in range(a["n_stu"])}
        print("\nshare of logged steps (2nd half) in which member j is the worst: "
              + ", ".join(f"{j}: {v:.2f}" for j, v in worst.items()))
        ft = r.get("stu_ft")
        if ft:
            print("\nstudent finetune on the labels (test %, members | ensemble):")
            for tag in ("before", "after"):
                t = ft[tag]
                print(f"  {tag:6s} " + " ".join(f"{100 * v:.2f}" for v in t["test"])
                      + f" | ens {100 * t['ens_test']:.2f}  (val ens {100 * t['ens_val']:.2f})")
            print(f"  teacher EMA (deployed): {100 * r['final']['test_ema']:.2f}")


if __name__ == "__main__":
    main()
