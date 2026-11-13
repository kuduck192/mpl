#!/usr/bin/env python3
"""Aggregate result JSONs downloaded from Kaggle (results/<slug>/results/*.json).

  python analysis/aggregate.py results/fmnist-ssl-pilot-01 [more dirs ...]
  python analysis/aggregate.py --group 'regex' results/*   # group runs by name with seed stripped
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def load(dirs):
    runs = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            if os.path.basename(f) == "summary.json":
                continue
            try:
                r = json.load(open(f))
            except Exception:
                continue
            if "final" not in r:
                continue
            r["_file"] = f
            runs.append(r)
    return runs


def strip_seed(name):
    return re.sub(r"_s\d+$", "", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--keys", default="test_ema,val_ema,stu_test_ema,unl_ema")
    ap.add_argument("--by_seed", action="store_true", help="group runs that differ only by _s<seed>")
    args = ap.parse_args()
    runs = load(args.dirs)
    keys = args.keys.split(",")
    if not args.by_seed:
        print(f"{'name':40s} {'iter':>6s} {'min':>6s} " + " ".join(f"{k:>13s}" for k in keys))
        for r in sorted(runs, key=lambda r: r["args"]["name"]):
            fi = r["final"]
            vals = " ".join(f"{fi.get(k, float('nan')):13.4f}" for k in keys)
            print(f"{r['args']['name']:40s} {fi.get('iter', 0):6d} {r.get('minutes', 0):6.1f} {vals}")
        return
    groups = defaultdict(list)
    for r in runs:
        groups[strip_seed(r["args"]["name"])].append(r)
    print(f"{'group':36s} {'n':>2s} " + " ".join(f"{k:>17s}" for k in keys))
    for g in sorted(groups):
        rs = groups[g]
        cells = []
        for k in keys:
            v = np.array([r["final"].get(k, np.nan) for r in rs], dtype=float) * 100
            cells.append(f"{np.nanmean(v):7.2f} ± {np.nanstd(v):5.2f}")
        print(f"{g:36s} {len(rs):2d} " + " ".join(f"{c:>17s}" for c in cells))


if __name__ == "__main__":
    main()
