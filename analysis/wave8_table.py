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
  python analysis/wave8_table.py results/fmnist-ssl-* --ref_avg          # two runs per seed as references
  python analysis/wave8_table.py results/fmnist-ssl-* --traj             # pseudo-label trajectories

--ref_avg replaces the FixMatch and SAM-on-L_l references by the per-seed mean of their two runs
(fixmatch + fixmatchft, abl_labsam + labsamft; the second run of each was trained with the same
flags and finetuned only after its final iterate), which lowers the rerun noise of the reference.
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
REF_AVG = {"fixmatch": ("fixmatch", "fixmatchft"), "abl_labsam": ("abl_labsam", "labsamft")}
TRAJ = [("FixMatch", "fixmatch", "abl_labsam"), ("FlexMatch", "flexmatch", "flexsam"),
        ("Pseudo-Label", "pseudolabel", "plsam"), ("Mean Teacher", "meanteacher", "mtsam"),
        ("FixMatch, Y_u/q variants", "fmsamfull", "fmsamfulladv"), ("FixMatch, Y_u variants", "labsamadv", "fmemapl")]
TRAJ_KEYS = ["mask", "pl_acc_masked", "test_ema"]
TRAJ_ITERS = [1000, 2000, 4000, 8000, 12000, 16000, 20000]
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


def histories(dirs, seeds):
    """method -> list of hist (one per run), for the arms and the rerun references."""
    want = set(ARMS) | {m for v in REF_AVG.values() for m in v}
    out = {}
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            if "args" not in r or not r.get("hist") or r.get("stopped_early"):
                continue
            m = re.match(r"(.+)_4_s(\d+)$", r["args"].get("name", ""))
            if m and m.group(1) in want and int(m.group(2)) in seeds:
                out.setdefault(m.group(1), []).append(r["hist"])
    return out


def trajectory(hists, key, it, w=500):
    """mean over runs of the mean of `key` over records within w iterations of `it`."""
    v = []
    for h in hists:
        x = [r[key] for r in h if key in r and abs(r["iter"] - it) <= w]
        if x:
            v.append(float(np.mean(x)))
    return float(np.mean(v)) if v else float("nan")


def print_traj(H):
    print("\nTrajectories, mean over runs (records within 500 iterations of each column)\n")
    for key in TRAJ_KEYS:
        print(f"| {key} | " + " | ".join(str(i) for i in TRAJ_ITERS) + " |\n|---|" + "---|" * len(TRAJ_ITERS))
        for _, a, b in TRAJ:
            for m in (a, b):
                if m in H:
                    print(f"| {m} ({len(H[m])}) | " + " | ".join(f"{trajectory(H[m], key, i):.3f}" for i in TRAJ_ITERS) + " |")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--metric", default="test_ema", choices=["test_ema", "unl_ema", "test"])
    ap.add_argument("--seeds", default="1-5", help="e.g. 1-5 or 1-10")
    ap.add_argument("--ref_avg", action="store_true", help="two runs per seed for fixmatch and abl_labsam")
    ap.add_argument("--traj", action="store_true", help="also print pseudo-label trajectories")
    args = ap.parse_args()
    a, b = (int(x) for x in args.seeds.split("-"))
    seeds = range(a, b + 1)
    R = load(args.dirs, seeds=seeds, metric=args.metric)
    if args.ref_avg:
        for m, src in REF_AVG.items():
            common = set.intersection(*(set(R.get(x, {})) for x in src))
            R[m] = {s_: {"final": float(np.mean([R[x][s_]["final"] for x in src]))} for s_ in common}
    what = {"test_ema": "EMA teacher, test accuracy", "unl_ema": "EMA teacher, pseudo-label accuracy on X_u",
            "test": "live teacher, test accuracy"}[args.metric]
    print(f"{what}, 40 labels, final iterate, seeds {args.seeds}: mean ± sd (n)"
          + (" [fixmatch, abl_labsam: mean of two runs per seed]" if args.ref_avg else "") + "\n")
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
    if args.traj:
        print_traj(histories(args.dirs, seeds))


if __name__ == "__main__":
    main()
