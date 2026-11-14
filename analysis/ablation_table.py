#!/usr/bin/env python3
"""Ablation table (40 labels by default): final / peak / validation-selected test accuracy of
the EMA teacher, late drop (peak - final), and feedback diagnostics averaged over the last k
evaluation records, per method over the seeds given.

  python analysis/ablation_table.py results/fmnist-ssl-main-a results/fmnist-ssl-main-b \
      results/fmnist-ssl-ablate-a results/fmnist-ssl-ablate-b --seeds 1-3 --npc 4
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

DIAG = [("lab_loss_anchor", "L_l anchor"), ("lab_loss_adv", "L_l perturbed"), ("ratio", "|c|/|c_opt|"),
        ("cos_c_copt", "cos(c, c_opt)"), ("fiber_kl", "KL to anchor"), ("pl_acc_masked", "PL acc (mask)")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--seeds", default="1-3")
    ap.add_argument("--npc", type=int, default=4)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--methods", default="fixmatch,fmtuned,mpl,ebpl,pbpl,pbpltuned,abl_samfb,abl_self,abl_selfopt,abl_selfraw,abl_selfoptraw,abl_fibersam,abl_labsam,abl_pbplflex,flexmatch")
    args = ap.parse_args()
    lo, hi = map(int, args.seeds.split("-"))
    seeds = set(range(lo, hi + 1))
    runs = defaultdict(dict)
    for d in args.dirs:
        for f in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            if "final" not in r or "args" not in r or not r.get("hist"):
                continue
            m = re.match(r"(.+)_(\d+)_s(\d+)$", r["args"]["name"])
            if not m or int(m.group(2)) != args.npc or int(m.group(3)) not in seeds:
                continue
            if r.get("stopped_early"):
                continue
            runs[m.group(1)][int(m.group(3))] = r
    head = ["method", "n", "final", "peak", "drop", "val-sel."] + [h for _, h in DIAG] + ["min/run"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for meth in args.methods.split(","):
        rs = runs.get(meth)
        if not rs:
            continue
        fin, peak, vsel, mins = [], [], [], []
        dg = defaultdict(list)
        for sd, r in sorted(rs.items()):
            h = [x for x in r["hist"] if "test_ema" in x]
            fin.append(r["final"]["test_ema"])
            peak.append(max(x["test_ema"] for x in h))
            vsel.append(max(h, key=lambda x: x.get("val_ema", -1))["test_ema"])
            mins.append(r.get("minutes", np.nan))
            last = r["hist"][-args.k:]
            for key, _ in DIAG:
                if key == "ratio":
                    v = [x["c_norm"] / x["c_opt_norm"] for x in last if x.get("c_opt_norm")]
                else:
                    v = [x[key] for x in last if x.get(key) is not None]
                if v:
                    dg[key].append(np.mean(v))
        fin, peak, vsel = (np.array(x) * 100 for x in (fin, peak, vsel))
        cells = [meth, str(len(fin)), f"{fin.mean():.2f}", f"{peak.mean():.2f}", f"{(peak - fin).mean():.2f}",
                 f"{vsel.mean():.2f}"]
        for key, _ in DIAG:
            v = dg.get(key)
            if not v:
                cells.append("—")
            elif key == "pl_acc_masked":
                cells.append(f"{100 * np.mean(v):.1f}")
            elif key in ("ratio", "cos_c_copt"):
                cells.append(f"{np.mean(v):.2f}")
            else:
                cells.append(f"{np.mean(v):.3g}")
        cells.append(f"{np.nanmean(mins):.1f}")
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
