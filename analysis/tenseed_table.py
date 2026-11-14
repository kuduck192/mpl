#!/usr/bin/env python3
"""40-label comparison on the seeds 1-10: per-seed final accuracy, mean final / val-selected /
peak / late drop / cost, and paired differences against a reference method with the paired
t-test, the Wilcoxon signed-rank test and Holm-adjusted t-test p-values over the methods
compared (a family of tests against the same reference).  Methods with fewer than --min_n
common seeds are listed but not tested.

  python analysis/tenseed_table.py results/fmnist-ssl-* --ref fixmatch
  python analysis/tenseed_table.py results/fmnist-ssl-* --pairs pbpl:mpl,abl_self:abl_selfopt
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats

# the family compared on seeds 1-10 (the exact-scale one-network pair, seeds 1-5, is a separate
# theory test: --methods fixmatch,abl_selfopt,abl_self,abl_selfoptraw,abl_selfraw)
METHODS = ["fixmatch", "fmtuned", "mpl", "pbpl", "ebpl", "abl_samfb", "abl_selfopt", "abl_self",
           "abl_fibersam", "abl_labsam"]


def load(dirs, npc=4, seeds=range(1, 11)):
    R = defaultdict(dict)  # method -> seed -> dict
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            if "final" not in r or "args" not in r or not r.get("hist"):
                continue
            m = re.match(r"(.+)_(\d+)_s(\d+)$", r["args"]["name"])
            if not m or int(m.group(2)) != npc or int(m.group(3)) not in seeds:
                continue
            if r.get("stopped_early") or r["final"].get("iter", 0) < r["args"].get("iters", 0):
                continue
            h = [x for x in r["hist"] if "test_ema" in x]
            R[m.group(1)][int(m.group(3))] = {
                "final": 100 * r["final"]["test_ema"],
                "val": 100 * max(h, key=lambda x: x.get("val_ema", -1))["test_ema"],
                "peak": 100 * max(x["test_ema"] for x in h),
                "min": r.get("minutes", np.nan)}
    return R


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        adj[j] = run
    return adj


def paired(R, a, b, key):
    common = sorted(set(R.get(a, {})) & set(R.get(b, {})))
    x = np.array([R[a][s][key] for s in common])
    y = np.array([R[b][s][key] for s in common])
    return common, x - y, x, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ref", default="fixmatch")
    ap.add_argument("--min_n", type=int, default=5)
    ap.add_argument("--pairs", default="", help="extra comparisons a:b,c:d (no multiplicity correction)")
    ap.add_argument("--methods", default=",".join(METHODS))
    args = ap.parse_args()
    R = load(args.dirs)
    meths = [m for m in args.methods.split(",") if m in R]
    print("Per-seed final test accuracy (EMA), 40 labels:\n")
    print("| method | " + " | ".join(f"s{s}" for s in range(1, 11)) + " | mean ± sd | n |")
    print("|---|" + "---|" * 12)
    for m in meths:
        cells = [f"{R[m][s]['final']:.1f}" if s in R[m] else "—" for s in range(1, 11)]
        v = np.array([x["final"] for x in R[m].values()])
        print(f"| {m} | " + " | ".join(cells) + f" | {v.mean():.2f} ± {v.std(ddof=1):.2f} | {len(v)} |")
    print("\nMeans (final / val-selected / peak / peak - final / minutes per run):\n")
    print("| method | n | final | val-sel. | peak | drop | min/run |")
    print("|---|---|---|---|---|---|---|")
    for m in meths:
        rs = list(R[m].values())
        f = np.mean([x["final"] for x in rs])
        print(f"| {m} | {len(rs)} | {f:.2f} | {np.mean([x['val'] for x in rs]):.2f} | "
              f"{np.mean([x['peak'] for x in rs]):.2f} | {np.mean([x['peak'] - x['final'] for x in rs]):.2f} | "
              f"{np.nanmean([x['min'] for x in rs]):.1f} |")
    for key in ("final", "val"):
        rows, ps = [], []
        for m in meths:
            if m == args.ref:
                continue
            common, d, x, y = paired(R, m, args.ref, key)
            if len(common) < 2:
                continue
            p = stats.ttest_rel(x, y).pvalue
            w = stats.wilcoxon(x, y).pvalue if len(common) >= 5 and np.any(d != 0) else np.nan
            rows.append((m, common, d, p, w))
            ps.append(p if len(common) >= args.min_n else np.nan)
        tested = [i for i, p in enumerate(ps) if np.isfinite(p)]
        adj = np.full(len(ps), np.nan)
        if tested:
            adj[tested] = holm([ps[i] for i in tested])
        print(f"\nPaired difference to {args.ref} ({key}; Holm over the {len(tested)} methods with "
              f"n >= {args.min_n}):\n")
        print("| method | seeds | mean ± sd | wins | t-test p | Wilcoxon p | Holm p |")
        print("|---|---|---|---|---|---|---|")
        for (m, common, d, p, w), a in zip(rows, adj):
            seeds = f"{common[0]}–{common[-1]}" if common == list(range(common[0], common[-1] + 1)) else ",".join(map(str, common))
            print(f"| {m} | {seeds} (n={len(common)}) | {d.mean():+.2f} ± {d.std(ddof=1):.2f} | "
                  f"{int((d > 0).sum())}/{len(d)} | {p:.3f} | {w:.3f} | "
                  + (f"{a:.3f}" if np.isfinite(a) else "—") + " |")
    if args.pairs:
        print("\nOther comparisons (no multiplicity correction):\n")
        print("| a − b | seeds | final | val-sel. |")
        print("|---|---|---|---|")
        for pr in args.pairs.split(","):
            a, b = pr.split(":")
            cells = []
            for key in ("final", "val"):
                common, d, x, y = paired(R, a, b, key)
                if len(common) < 2:
                    cells.append("—")
                    continue
                p = stats.ttest_rel(x, y).pvalue
                cells.append(f"{d.mean():+.2f} ± {d.std(ddof=1):.2f}, {int((d > 0).sum())}/{len(d)}, p = {p:.3f}")
            common = sorted(set(R.get(a, {})) & set(R.get(b, {})))
            print(f"| {a} − {b} | n={len(common)} | {cells[0]} | {cells[1]} |")


if __name__ == "__main__":
    main()
