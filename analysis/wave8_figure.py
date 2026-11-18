#!/usr/bin/env python3
"""Figure for wave8 (docs/figures/wave8.png), 40 labels, seeds 1-5.

  (a) paired difference in test accuracy of the EMA teacher for every comparison of
      analysis/wave8_table.py: one dot per seed, the mean and its 95% t-interval; the supervised
      row uses the live teacher, since the EMA of a run with no loss on X_u is unstable (its
      BatchNorm buffers are copied from the live network; report section 17);
  (b) accuracy of the pseudo-labels the teacher trains on (masked, mean over the last quarter of
      training) against final test accuracy, one arrow per prior from the plain method to the
      same method with SAM on L_l.

docs/figures/wave8_traj.png: for each prior, the fraction of the unlabeled batch that passes the
threshold, the accuracy of those pseudo-labels and the test accuracy of the live teacher along
training, plain method vs + SAM on L_l (mean over seeds 1-5).

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
from wave8_table import FAM_P, FAM_Q, diagnostics, histories  # noqa: E402

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
LIVE_ROWS = {"supsam"}  # rows drawn with the live teacher
# label offsets (points) and alignment of each prior's name next to its plain-method point
LABEL_AT = {"FixMatch": ((-8, 0), "right"), "FlexMatch": ((-5, 6), "right"),
            "Pseudo-Label": ((5, -7), "left"), "Mean Teacher": ((7, 3), "left")}


def forest(ax, R, R_live):
    def pr(lab, a, b):
        return (lab + " (live)", paired(R_live, a, b)) if a in LIVE_ROWS else (lab, paired(R, a, b))
    rows = [pr(lab, a, b) + (SERIES[0],) for lab, a, b in FAM_Q] + \
           [pr(lab, a, b) + (SERIES[1],) for lab, a, b in FAM_P]
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
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=3, handletextpad=0.3,
              columnspacing=1.0)
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
        off, ha = LABEL_AT.get(name, ((5, -7), "left"))
        ax.annotate(name, pa, xytext=off, textcoords="offset points", ha=ha, va="center", fontsize=7, color=INK)
    ax.margins(x=0.12, y=0.1)
    ax.plot([], [], "o", ms=5, color=INK2, label="plain method")
    ax.plot([], [], "o", ms=5, color=SERIES[1], label="+ SAM on L_l (arrow)")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlabel("accuracy of the pseudo-labels used (%, last quarter)")
    ax.set_ylabel("test accuracy, EMA teacher (%)")
    ax.set_title("(b) arm means, seeds 1-5", fontsize=8, color=INK, loc="left")


def curve(hists, key):
    its = sorted({r["iter"] for h in hists for r in h if key in r})
    ys = [np.mean([r[key] for h in hists for r in h if r["iter"] == i and key in r]) for i in its]
    return np.array(its), np.array(ys)


def traj_figure(H, out):
    pri = [p for p in PRIORS if p[1] in H]
    keys = [("mask", "fraction of X_u batch used"), ("pl_acc_masked", "accuracy of used pseudo-labels"),
            ("test", "test accuracy, live teacher")]
    lims = {"mask": (0, 1.02), "pl_acc_masked": (0.3, 0.9), "test": (0.45, 0.8)}
    fig, axes = plt.subplots(len(keys), len(pri), figsize=(2.15 * len(pri), 5.6), squeeze=False)
    for j, (name, a, b) in enumerate(pri):
        for i, (key, ylab) in enumerate(keys):
            ax = axes[i][j]
            for m, col, lab in ((a, INK2, "plain"), (b, SERIES[1], "+ SAM on L_l")):
                if m in H:
                    x, y = curve(H[m], key)
                    ax.plot(x / 1000, y, color=col, lw=2, label=lab)
            ax.set_ylim(*lims[key])
            ax.set_xticks([0, 10, 20])
            if i == 0:
                ax.set_title(name, fontsize=8, color=INK)
            if i == len(keys) - 1:
                ax.set_xlabel("iteration (thousands)")
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--seeds", default="1-5")
    ap.add_argument("--out", default=os.path.join(ROOT, "docs", "figures", "wave8.png"))
    args = ap.parse_args()
    a, b = (int(x) for x in args.seeds.split("-"))
    seeds = range(a, b + 1)
    R = load(args.dirs, seeds=seeds)
    R_live = load(args.dirs, seeds=seeds, metric="test")
    D = diagnostics(args.dirs, seeds)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    forest(ax1, R, R_live)
    arrows(ax2, R, D)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    print("wrote", args.out)
    traj_figure(histories(args.dirs, seeds), args.out.replace(".png", "_traj.png"))


if __name__ == "__main__":
    main()
