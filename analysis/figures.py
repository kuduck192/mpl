#!/usr/bin/env python3
"""Static figures for the report (PNG).  Palette: validated categorical order (dataviz
reference palette, light mode), neutral ink for text, recessive grid, one y-axis per panel.

  python analysis/figures.py screen      # Figs from the validation screen
  python analysis/figures.py degeneracy  # static degeneracy diagnostic (checks/degeneracy_static.py)
  python analysis/figures.py figure1     # Figure 1 protocol (after ablate-b)
  python analysis/figures.py main        # paired differences vs FixMatch (after main-a/b)
  python analysis/figures.py curves      # training curves of the main runs (after main-a/b)
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# colour follows the method in every figure; FixMatch is the neutral reference
COL = {"mpl": SERIES[0], "pbpl": SERIES[1], "ebpl": SERIES[2], "pbpl_r0.3": SERIES[3], "fixmatch": INK2}
plt.rcParams.update({
    "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2.0, "savefig.dpi": 200, "figure.facecolor": "white",
})


def _load(pattern):
    out = {}
    for f in glob.glob(os.path.join(ROOT, pattern), recursive=True):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        if "args" in r and "hist" in r:
            out[r["args"]["name"]] = r
    return out


def _label_end(ax, x, y, text, color):
    ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7, color=INK)
    ax.plot([x], [y], "o", ms=4, color=color)


def screen():
    R = _load("results/fmnist-ssl-screen-*/results/scr_*.json")
    os.makedirs(OUT, exist_ok=True)
    # (A) held-out labeled loss of the anchor vs labeled loss of the pessimistic student
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    h = R["scr_pess_lm0.1_r0.1"]["hist"]
    it = [d["iter"] for d in h]
    a = [d["lab_loss_anchor"] for d in h]
    p = [d["lab_loss_adv"] for d in h]
    ho = R["scr_opt_lm0.1"]["hist"]
    ax.plot(it, a, color=SERIES[0], label="anchor student (optimistic feedback)")
    ax.plot(it, p, color=SERIES[1], ls="--", label="worst admissible student, r = 0.1 (PBPL)")
    ax.plot([d["iter"] for d in ho], [d["lab_loss_anchor"] for d in ho], color=SERIES[2], ls=":",
            label="anchor student of an MPL run")
    ax.set_xlabel("iteration")
    ax.set_ylabel("held-out labeled CE of the student")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_title("Held-out labeled loss: anchor vs worst admissible student (40 labels, seed 0)", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "screen_feedback.png"))
    # (B) sensitivity to lambda_m (validation accuracy at 10k iterations)
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    fm = 100 * R["scr_fixmatch"]["final"]["val_ema"]
    ax.axhline(fm, color=INK2, lw=1.0, ls="-")
    ax.annotate("FixMatch", (1.02, fm), xycoords=("axes fraction", "data"), va="center", fontsize=7, color=INK2)
    for i, (lab, pat, col) in enumerate([("MPL", "scr_opt_lm{}", COL["mpl"]),
                                         ("PBPL, r = 0.1", "scr_pess_lm{}_r0.1", COL["pbpl"]),
                                         ("PBPL, r = 0.3", "scr_pess_lm{}_r0.3", COL["pbpl_r0.3"]),
                                         ("EBPL, r = 3", "scr_exp_lm{}_r3.0", COL["ebpl"])]):
        xs, ys = [], []
        for lm in [0.1, 0.3, 1.0]:
            n = pat.format(lm)
            if n in R:
                xs.append(lm)
                ys.append(100 * R[n]["final"]["val_ema"])
        ax.plot(xs, ys, "-o", ms=4, color=col, label=lab)
    ax.set_xscale("log")
    ax.minorticks_off()
    ax.set_xticks([0.1, 0.3, 1.0])
    ax.set_xticklabels(["0.1", "0.3", "1.0"])
    ax.set_xlabel(r"meta weight $\lambda_m$ (running-RMS normalised feedback)")
    ax.set_ylabel("validation accuracy (%)")
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("Validation screen: 40 labels, seed 0, 10k iterations", fontsize=8, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "screen_lambda.png"))
    print("wrote", OUT)


def degeneracy():
    f = sorted(glob.glob(os.path.join(ROOT, "results/**/degeneracy_static.json"), recursive=True))
    d = json.load(open(f[0]))
    rows = [r for v in d["results"].values() for r in v]
    keys = [("gT_opt", "optimistic (MPL)", SERIES[0], "-"), ("gT_pess_r0.1", "pessimistic, r = 0.1", SERIES[1], "--"),
            ("gT_pess_r0.3", "pessimistic, r = 0.3", SERIES[3], "-."), ("gT_exp_r3.0", "expected, r = 3", SERIES[2], ":")]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for k, lab, col, ls in keys:
        by = defaultdict(list)
        for r in rows:
            by[r["iter"]].append((r["L_l"], r[k]))
        its = sorted(by)
        x = [np.mean([a for a, _ in by[i]]) for i in its]
        y = [np.mean([b for _, b in by[i]]) for i in its]
        ax.plot(x, y, ls=ls, marker="o", ms=4, color=col, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel("labeled loss of the supervised-only model (training proceeds →)")
    ax.set_ylabel(r"teacher meta-gradient norm $\|\nabla_\phi\|$")
    ax.legend(fontsize=7)
    ax.set_title("Feedback at supervised-only pairs (CNN, 100 labels, mean of 3 seeds)", fontsize=8, color=INK)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "degeneracy_static.png"))
    print("wrote", OUT)


def figure1():
    f = sorted(glob.glob(os.path.join(ROOT, "results/**/figure1.json"), recursive=True))
    d = json.load(open(f[0]))
    rows = [r for v in d["results"].values() for r in v]
    fig, ax = plt.subplots(figsize=(4.6, 2.9))
    for k, lab, col, ls in [("Ll_pess", "pessimistic student", SERIES[1], "--"),
                            ("Ll_anchor", "trajectory student (anchor)", SERIES[2], "-"),
                            ("Ll_opt", "optimistic student", SERIES[0], ":")]:
        by = defaultdict(list)
        for r in rows:
            by[r["rho"]].append(r[k])
        xs = sorted(by)
        m = np.array([np.mean(by[x]) for x in xs])
        s = np.array([np.std(by[x]) for x in xs])
        ax.plot(xs, m, ls=ls, marker="o", ms=4, color=col, label=lab)
        ax.fill_between(xs, m - s, m + s, color=col, alpha=0.15, lw=0)
    ax.set_xlabel("pseudo-label error rate ρ")
    ax.set_ylabel("labeled CE")
    ax.legend(fontsize=7)
    ax.set_title("Admissible students around the anchor (FashionMNIST CNN, 3 seeds)", fontsize=8, color=INK)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "figure1_fmnist.png"))
    print("wrote", OUT)


def main():
    R = _load("results/fmnist-ssl-main-*/results/*.json")
    res = defaultdict(dict)
    for n, r in R.items():
        m = re.match(r"(.+)_(\d+)_s(\d+)$", n)
        if m:
            res[(m.group(1), int(m.group(2)))][int(m.group(3))] = 100 * r["final"]["test_ema"]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.6), sharey=True)
    for ax, npc in zip(axes, (4, 10)):
        base = res.get(("fixmatch", npc), {})
        for i, (meth, lab) in enumerate([("mpl", "MPL"), ("ebpl", "EBPL"), ("pbpl", "PBPL")]):
            v = res.get((meth, npc), {})
            common = sorted(set(v) & set(base))
            dif = [v[s] - base[s] for s in common]
            ax.scatter([i] * len(dif), dif, s=18, color=COL[meth], zorder=3)
            if dif:
                ax.plot([i - 0.2, i + 0.2], [np.mean(dif)] * 2, color=INK, lw=1.5)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(3))
        ax.set_xticklabels(["MPL", "EBPL", "PBPL"])
        ax.set_title(f"{10 * npc} labels", fontsize=8, color=INK)
    axes[0].set_ylabel("test acc. − FixMatch, same seed (pp)")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "main_paired.png"))
    print("wrote", OUT)


def curves():
    """Mean over seeds of test accuracy (EMA teacher) and of the masked pseudo-label accuracy
    along training, FixMatch vs the three bilevel feedback rules, 40 and 100 labels."""
    R = _load("results/fmnist-ssl-main-*/results/*.json")
    runs = defaultdict(list)
    for n, r in R.items():
        m = re.match(r"(.+)_(\d+)_s(\d+)$", n)
        if m:
            runs[(m.group(1), int(m.group(2)))].append(r["hist"])
    meths = [("fixmatch", "FixMatch"), ("mpl", "MPL"), ("ebpl", "EBPL"), ("pbpl", "PBPL")]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.6))
    for j, npc in enumerate((4, 10)):
        for row, key, ylab, scale in [(0, "test_ema", "test accuracy (%)", 100),
                                      (1, "pl_acc_masked", "pseudo-label accuracy\nabove threshold (%)", 100)]:
            ax = axes[row, j]
            for meth, lab in meths:
                hs = runs.get((meth, npc), [])
                if not hs:
                    continue
                its = sorted(set.intersection(*[{d["iter"] for d in h if key in d} for h in hs]))
                y = [scale * np.mean([next(d[key] for d in h if d["iter"] == i) for h in hs]) for i in its]
                ax.plot(its, y, color=COL[meth], lw=1.6 if meth != "fixmatch" else 1.2,
                        ls="-" if meth != "fixmatch" else "--", label=f"{lab} (n={len(hs)})")
            ax.set_ylabel(ylab)
            if row == 0:
                ax.set_title(f"{10 * npc} labels", fontsize=8, color=INK)
                lo = ax.get_ylim()
                ax.set_ylim(max(lo[0], 50), lo[1])
            else:
                ax.set_xlabel("iteration")
        axes[0, j].legend(fontsize=6.5, loc="lower right")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "main_curves.png"))
    print("wrote", OUT)


if __name__ == "__main__":
    globals()[sys.argv[1]]()
