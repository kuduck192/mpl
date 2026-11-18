#!/usr/bin/env python3
"""Wave8: variants from the variational view of pseudo-labeling (docs/bayes_view.md), 40 labels.

  q-step / Y_u-step on FixMatch : fmsamfull, fmsamfulladv, labsamadv, fmemapl
  SAM on L_l under other priors  : supsam, plsam, mtsam, flexsam  (vs sup, pseudolabel, meanteacher, flexmatch)

Prints the means of every arm and reference on the seeds given (default 1-5), the paired
comparisons of each family (paired t-test, Wilcoxon, Holm within the family), and pseudo-label
diagnostics averaged over the last quarter of training (mask rate and accuracy of the pseudo-labels
used, of FixMatch's point pseudo-labels where they differ, and their agreement).

  python analysis/wave8_table.py results/fmnist-ssl-*
  python analysis/wave8_table.py results/fmnist-ssl-* --metric test      # live teacher
  python analysis/wave8_table.py results/fmnist-ssl-* --metric unl_ema   # pseudo-label accuracy on X_u
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_meta_grid import paired, report  # noqa: E402
from tenseed_table import load  # noqa: E402

ARMS = ["fixmatch", "abl_labsam", "fmsamfull", "fmsamfulladv", "labsamadv", "fmemapl",
        "sup", "supsam", "pseudolabel", "plsam", "meanteacher", "mtsam", "flexmatch", "flexsam"]
FAM_Q = [("SAM on L_l + lam_u L_u − FixMatch", "fmsamfull", "fixmatch"),
         ("SAM on L_l + lam_u L_u − SAM on L_l", "fmsamfull", "abl_labsam"),
         ("SAM-full, adversarial PL − SAM-full", "fmsamfulladv", "fmsamfull"),
         ("SAM-full, adversarial PL − SAM on L_l", "fmsamfulladv", "abl_labsam"),
         ("SAM on L_l, adversarial PL − SAM on L_l", "labsamadv", "abl_labsam"),
         ("EMA pseudo-labels − FixMatch", "fmemapl", "fixmatch")]
FAM_P = [("SAM on L_l, FixMatch (reference)", "abl_labsam", "fixmatch"),
         ("SAM on L_l, supervised", "supsam", "sup"),
         ("SAM on L_l, Pseudo-Label", "plsam", "pseudolabel"),
         ("SAM on L_l, Mean Teacher", "mtsam", "meanteacher"),
         ("SAM on L_l, FlexMatch", "flexsam", "flexmatch")]
DIAG = ["mask", "pl_acc_masked", "pl_acc_all", "mask_point", "pl_acc_masked_point", "pl_agree",
        "lab_loss_adv", "loss_center", "loss_adv"]


def diagnostics(dirs, seeds, frac=0.75):
    """method -> key -> mean over runs of the mean over records with iter >= frac * iters."""
    acc = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            if "args" not in r or not r.get("hist") or r.get("stopped_early"):
                continue
            m = re.match(r"(.+)_4_s(\d+)$", r["args"].get("name", ""))
            if not m or m.group(1) not in ARMS or int(m.group(2)) not in seeds:
                continue
            late = [h for h in r["hist"] if h.get("iter", 0) >= frac * r["args"]["iters"]]
            for k in DIAG:
                v = [h[k] for h in late if k in h]
                if v:
                    acc.setdefault(m.group(1), {}).setdefault(k, []).append(float(np.mean(v)))
    return {m: {k: float(np.mean(v)) for k, v in kv.items()} for m, kv in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--metric", default="test_ema", choices=["test_ema", "unl_ema", "test"])
    ap.add_argument("--seeds", default="1-5", help="e.g. 1-5 or 1-10")
    args = ap.parse_args()
    a, b = (int(x) for x in args.seeds.split("-"))
    seeds = range(a, b + 1)
    R = load(args.dirs, seeds=seeds, metric=args.metric)
    what = {"test_ema": "EMA teacher, test accuracy", "unl_ema": "EMA teacher, pseudo-label accuracy on X_u",
            "test": "live teacher, test accuracy"}[args.metric]
    print(f"{what}, 40 labels, final iterate, seeds {args.seeds}: mean ± sd (n)\n")
    print("| arm | accuracy |\n|---|---|")
    for m in ARMS:
        if m in R:
            v = np.array([x["final"] for x in R[m].values()])
            sd = f" ± {v.std(ddof=1):.2f}" if len(v) > 1 else ""
            print(f"| {m} | {v.mean():.2f}{sd} ({len(v)}) |")
        else:
            print(f"| {m} | not run |")
    report("q-step and Y_u-step variants of FixMatch", [(lab, paired(R, x, y)) for lab, x, y in FAM_Q])
    report("SAM on L_l under different priors -log P(h | X)", [(lab, paired(R, x, y)) for lab, x, y in FAM_P])
    D = diagnostics(args.dirs, seeds)
    print("\nPseudo-label diagnostics, mean over the last quarter of training (fractions; losses in nats)\n")
    cols = [k for k in DIAG if any(k in D.get(m, {}) for m in ARMS)]
    print("| arm | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols))
    for m in ARMS:
        if m in D:
            print(f"| {m} | " + " | ".join(f"{D[m][k]:.3f}" if k in D[m] else "" for k in cols) + " |")


if __name__ == "__main__":
    main()
