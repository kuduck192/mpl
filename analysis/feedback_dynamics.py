#!/usr/bin/env python3
"""Feedback-size dynamics of one-network runs (report section 6.1), per seed:

  |c| first / last   raw RMS size of the feedback actually used, at the first and the last
                     evaluation record (each record averages the steps logged since the previous)
  shrink             first / last
  L_l anchor         labeled loss at the anchor (the network after the look-ahead step), last record
  |c|/|c_opt|        pessimistic feedback relative to the optimistic one on the same anchor (last)
  cos                cosine between the two (centered, per-example weighted), last record
  median meta        median of the logged meta loss over the records from 45% of training on
                     (iterations 9k-20k of 20k), and the range of |c| over the same records

  python analysis/feedback_dynamics.py results/fmnist-ssl-* --methods abl_selfopt,abl_self
  python analysis/feedback_dynamics.py results/fmnist-ssl-* --methods abl_selfoptraw,abl_selfraw
  python analysis/feedback_dynamics.py results/fmnist-ssl-* --pair abl_selfraw:abl_selfoptraw

--pair a:b prints, per seed and iteration, |c| of run a over |c| of run b (the ratio of the meta
steps when both use the exact meta-gradient scale with the same lambda_m and learning rate).
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def load(dirs, npc=4, seeds=range(0, 11)):
    R = defaultdict(dict)
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
            R[m.group(1)][int(m.group(3))] = r
    return R


def at(h, key, i):
    v = h[i].get(key)
    return np.nan if v is None else float(v)


def rng(v, fmt="{:.2g}"):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return "—" if not len(v) else (fmt.format(v.min()) + " to " + fmt.format(v.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--methods", default="abl_selfopt,abl_self,abl_selfoptraw,abl_selfraw")
    ap.add_argument("--pair", default="", help="a:b, ratio of |c| between two runs on the same seed")
    ap.add_argument("--from_frac", type=float, default=0.45)
    args = ap.parse_args()
    R = load(args.dirs)
    for meth in args.methods.split(","):
        if meth not in R:
            continue
        print(f"\n{meth}:\n")
        print("| seed | |c| first | |c| last | shrink | L_l anchor | |c|/|c_opt| | cos | median meta | |c| range (2nd half) |")
        print("|---|---|---|---|---|---|---|---|---|")
        agg = defaultdict(list)
        for sd, r in sorted(R[meth].items()):
            h = [x for x in r["hist"] if "c_norm" in x]
            if not h:
                continue
            it = r["args"]["iters"]
            late = [x for x in h if x["iter"] >= args.from_frac * it]
            c0, c1 = at(h, "c_norm", 0), at(h, "c_norm", -1)
            copt = at(h, "c_opt_norm", -1)
            row = {"c0": c0, "c1": c1, "shrink": c0 / c1, "Ll": at(h, "lab_loss_anchor", -1),
                   "ratio": c1 / copt if np.isfinite(copt) else np.nan, "cos": at(h, "cos_c_copt", -1),
                   "meta": float(np.median([x["meta"] for x in late if "meta" in x])),
                   "lo": min(x["c_norm"] for x in late), "hi": max(x["c_norm"] for x in late)}
            for k, v in row.items():
                agg[k].append(v)
            print(f"| {sd} | {c0:.3g} | {c1:.3g} | {row['shrink']:.1f} | {row['Ll']:.2g} | "
                  + (f"{row['ratio']:.1f}" if np.isfinite(row["ratio"]) else "—") + " | "
                  + (f"{row['cos']:.2f}" if np.isfinite(row["cos"]) else "—")
                  + f" | {row['meta']:+.3g} | {row['lo']:.2g}–{row['hi']:.2g} |")
        sh = np.array(agg["shrink"])
        print(f"\nn = {len(sh)}; shrink {rng(sh, '{:.2g}')} (median {np.median(sh):.2g}); L_l anchor {rng(agg['Ll'])}; "
              f"|c|/|c_opt| {rng(agg['ratio'], '{:.0f}')}; cos {rng(agg['cos'])}; median meta {rng(agg['meta'], '{:+.2f}')}; "
              f"|c| in the 2nd half {rng(agg['lo'] + agg['hi'])}")
    if args.pair:
        a, b = args.pair.split(":")
        common = sorted(set(R.get(a, {})) & set(R.get(b, {})))
        print(f"\n|c| of {a} over |c| of {b}, same seed (rows: iteration):\n")
        print("| iter | " + " | ".join(f"s{s}" for s in common) + " | median |")
        print("|---|" + "---|" * (len(common) + 1))
        ha = {s: {x["iter"]: x["c_norm"] for x in R[a][s]["hist"] if "c_norm" in x} for s in common}
        hb = {s: {x["iter"]: x["c_norm"] for x in R[b][s]["hist"] if "c_norm" in x} for s in common}
        its = sorted(set.intersection(*[set(ha[s]) & set(hb[s]) for s in common])) if common else []
        for i in its:
            v = [ha[s][i] / hb[s][i] for s in common]
            print(f"| {i} | " + " | ".join(f"{x:.1f}" for x in v) + f" | {np.median(v):.1f} |")


if __name__ == "__main__":
    main()
