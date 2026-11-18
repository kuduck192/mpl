#!/usr/bin/env python3
"""Figure for wave8 (docs/figures/wave8.png), 40 labels, seeds 1-5.

  (a) paired difference in test accuracy of the EMA teacher for every comparison of
      analysis/wave8_table.py: one dot per seed, the mean and its 95% t-interval;
  (b) accuracy of the pseudo-labels the teacher trains on (masked, mean over the last quarter of
      training) against final test accuracy, one arrow per prior from the plain method to the
      same method with SAM on L_l, and the FixMatch variants of the Y_u- and q-steps as points.

  python analysis/wave8_figure.py results/fmnist-ssl-*
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fm_meta_grid import paired  # noqa: E402
from tenseed_table import load  # noqa: E402
from wave8_table import FAM_P, FAM_Q, diagnostics  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2.0, "savefig.dpi": 200, "figure.facecolor": "white",
})
PRIORS = [("FixMatch", "fixmatch", "abl_labsam"), ("FlexMatch", "flexmatch", "flexsam"),
          ("Pseudo-Label", "pseudolabel", "plsam"), ("Mean Teacher", "meanteacher", "mtsam")]
VARIANTS = [("SAM-full", "fmsamfull"), ("SAM-full, adv. PL", "fmsamfulladv"),
            ("SAM on L_l, adv. PL", "labsamadv"), ("EMA PL", "fmemapl")]


def forest(ax, R):
    rows = [(lab, paired(R, a, b), SERIES[0]) for lab, a, b in FAM_Q] + \
           [(lab, paired(R, a, b), SERIES[1]) for lab, a, b in FAM_P]
    rows = [r for r in rows if r[1] is not None]
    for k, (lab, r, col) in enumerate(rows):
        y = len(rows) - 1 - k
        d = r["d"]
        half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        ax.plot(d, np.full(len(d), y), "o", ms=3.5, mfc="white", mec=INK2, mew=0.8, zorder=2)
        ax.plot([d.mean() - half, d.mean() + half], [y, y], color=col, lw=2, solid_capstyle="round", zorder=3)
        ax.plot([d.mean()], [y], "o", ms=6, color=col, mec="white", mew=1.2, zorder=4)
    ax.axvline(0, color=INK2, lw=0.8, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=7, color=INK)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("paired difference in test accuracy (points)")
    ax.plot([], [], "o", ms=6, color=SERIES[0], label="Y_u- / q-step variants of FixMatch")
    ax.plot([], [], "o", ms=6, color=SERIES[1], label="SAM on L_l, plain vs + SAM")
    ax.plot([], [], "o", ms=3.5, mfc="white", mec=INK2, label="one seed")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("(a) paired by seed; bar = 95% t-interval of the mean", fontsize=8, color=INK, loc="left")


def arrows(ax, R, D):
    def pt(m):
        if m not in R or m not in D or "pl_acc_masked" not in D[m]:
            return None
        return 100 * D[m]["pl_acc_masked"], float(np.mean([v["final"] for v in R[m].values()]))
    for name, a, b in PRIORS:
        pa, pb = pt(a), pt(b)
        if pa is None:
            continue
        ax.plot(*pa, "o", ms=5, color=INK2, zorder=3)
        if pb is not None:
            ax.annotate("", xy=pb, xytext=pa, zorder=2,
                        arrowprops=dict(arrowstyle="-|>", color=SERIES[1], lw=1.6, shrinkA=3, shrinkB=3))
            ax.plot(*pb, "o", ms=5, color=SERIES[1], zorder=3)
        ax.annotate(name, pa, xytext=(-4, -2), textcoords="offset points", ha="right", va="top", fontsize=7, color=INK)
    for name, m in VARIANTS:
        p = pt(m)
        if p is None:
            continue
        ax.plot(*p, "s", ms=5, color=SERIES[0], zorder=3)
        ax.annotate(name, p, xytext=(4, 2), textcoords="offset points", fontsize=7, color=INK)
    ax.margins(x=0.18, y=0.08)
    ax.plot([], [], "o", ms=5, color=INK2, label="plain method")
    ax.plot([], [], "o", ms=5, color=SERIES[1], label="+ SAM on L_l (arrow)")
    ax.plot([], [], "s", ms=5, color=SERIES[0], label="FixMatch variant")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlabel("accuracy of the pseudo-labels used (%, last quarter)")
    ax.set_ylabel("test accuracy, EMA teacher (%)")
    ax.set_title("(b) arm means, seeds 1-5", fontsize=8, color=INK, loc="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--seeds", default="1-5")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures", "wave8.png"))
    args = ap.parse_args()
    a, b = (int(x) for x in args.seeds.split("-"))
    seeds = range(a, b + 1)
    R = load(args.dirs, seeds=seeds)
    D = diagnostics(args.dirs, seeds)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.6), gridspec_kw={"width_ratios": [1.15, 1]})
    forest(ax1, R)
    arrows(ax2, R, D)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
