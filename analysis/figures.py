#!/usr/bin/env python3
"""Static figures for the report (PNG).  Palette: validated categorical order (dataviz
reference palette, light mode), neutral ink for text, recessive grid, one y-axis per panel.

  python analysis/figures.py screen      # Figs from the validation screen
  python analysis/figures.py degeneracy  # static degeneracy diagnostic (checks/degeneracy_static.py)
  python analysis/figures.py figure1     # Figure 1 protocol (after ablate-b)
  python analysis/figures.py main        # paired differences vs FixMatch (after main-a/b)
  python analysis/figures.py curves      # training curves of the main runs (after main-a/b)
  python analysis/figures.py dynamics    # feedback diagnostics along the main runs
  python analysis/figures.py replication # paired differences, seeds 1-5 vs 6-10 (after rep-b)
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
COL = {"mpl": SERIES[0], "pbpl": SERIES[1], "ebpl": SERIES[2], "pbpl_r0.3": SERIES[3], "fixmatch": INK2,
       "fmtuned": SERIES[4]}
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
    """Prop. N2 on the CNN: teacher meta-gradient norm at supervised-only pairs, per feedback
    rule, along supervised-only training (mean over seeds); also prints a markdown table."""
    files = sorted(glob.glob(os.path.join(ROOT, "results/**/degeneracy_static*.json"), recursive=True))
    rows = []
    for f in files:
        d = json.load(open(f))
        rows += [r for v in d["results"].values() for r in v]
    by = defaultdict(list)
    for r in rows:
        by[r["iter"]].append(r)
    its = sorted(by)
    keys = [("gT_opt", "optimistic (MPL)", COL["mpl"], "-"),
            ("gT_pess_r0.1", "pessimistic on the fiber, r = 0.1 (PBPL)", COL["pbpl"], "-"),
            ("gT_pess_r0.3", "pessimistic, r = 0.3 (≤ 8% of iterates admissible)", COL["pbpl_r0.3"], "-"),
            ("gT_exp_r3.0", "expected, r = 3 (EBPL)", COL["ebpl"], "-"),
            ("gT_sam_r0.1", "unconstrained SAM step, r = 0.1 (off the fiber)", INK2, "--")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
    for k, lab, col, ls in keys:
        if not all(k in r for r in rows):
            continue
        m = [np.mean([r[k] for r in by[i]]) for i in its]
        axes[0].plot(its, m, ls=ls, marker="o", ms=3.5, color=col, label=lab, lw=1.6)
        if k != "gT_opt":
            ratio = [np.mean([r[k] / r["gT_opt"] for r in by[i]]) for i in its]
            axes[1].plot(its, ratio, ls=ls, marker="o", ms=3.5, color=col, lw=1.6)
    axes[1].axhline(1.0, color=INK2, lw=0.8)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.minorticks_off()
        ax.set_xticks(its)
        ax.set_xticklabels([str(i) for i in its])
        ax.set_xlabel("supervised-only training iterations")
    axes[0].set_ylabel(r"teacher meta-gradient norm $\|g_T\|$")
    axes[1].set_ylabel("ratio to the optimistic rule")
    n_seeds = len({r["seed"] for r in rows})
    axes[0].set_title(f"Feedback at supervised-only pairs (mean of {n_seeds} seed{'s' if n_seeds > 1 else ''})",
                      fontsize=8, color=INK)
    axes[1].set_title("Ratio to the optimistic feedback", fontsize=8, color=INK)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "degeneracy_static.png"))
    # markdown table
    cols = [("L_l", "{:.1e}"), ("gT_opt", "{:.3g}"), ("gT_pess_r0.1", "{:.3g}"), ("pess_r0.1_kl", "{:.1e}"),
            ("gT_pess_r0.3", "{:.3g}"), ("pess_r0.3_kl", "{:.1e}"), ("pess_r0.3_adm", "{:.2f}"),
            ("gT_exp_r3.0", "{:.3g}"), ("exp_r3.0_kl", "{:.1e}"), ("gT_sam_r0.1", "{:.3g}"), ("sam_r0.1_kl", "{:.1e}")]
    print("| iter | " + " | ".join(c for c, _ in cols) + " |")
    print("|---|" + "---|" * len(cols))
    for i in its:
        cells = []
        for c, fm in cols:
            v = [r[c] for r in by[i] if c in r]
            cells.append(fm.format(np.mean(v)) if v else "—")
        print(f"| {i} | " + " | ".join(cells) + " |")
    print("wrote", OUT)


def figure1():
    """Figure 1 protocol of the review on FashionMNIST.  Cells whose anchor did not fit the
    pseudo-labels (L_u(anchor) > 0.05) are dropped: there the tolerance 2 L_u(anchor) admits
    students far from the fiber and the spread is an artefact."""
    f = sorted(glob.glob(os.path.join(ROOT, "results/**/figure1.json"), recursive=True))
    d = json.load(open(f[0]))
    rows = [r for v in d["results"].values() for r in v]
    ok = [r for r in rows if r["Lu_anchor"] <= 0.05]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    for k, lab, col, ls in [("Ll_pess", "pessimistic student", SERIES[1], "--"),
                            ("Ll_anchor", "trajectory student (anchor)", SERIES[2], "-"),
                            ("Ll_opt", "optimistic student", SERIES[0], ":")]:
        by = defaultdict(list)
        for r in ok:
            by[r["rho"]].append(r[k])
        xs = sorted(by)
        m = np.array([np.mean(by[x]) for x in xs])
        ax.plot(xs, m, ls=ls, marker="o", ms=4, color=col, label=lab)
    ax.set_xlabel("pseudo-label error rate ρ")
    ax.set_ylabel("labeled CE")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("Labeled loss of admissible students (the three curves overlap)", fontsize=8, color=INK)
    ax = axes[1]
    by = defaultdict(list)
    for r in ok:
        by[r["rho"]].append(max(r["Ll_pess"] - r["Ll_opt"], 1e-5))
    xs = sorted(by)
    for x in xs:
        ax.scatter([x] * len(by[x]), by[x], s=12, color=SERIES[1], alpha=0.6, lw=0)
    ax.plot(xs, [np.mean(by[x]) for x in xs], color=SERIES[1], marker="o", ms=4)
    ax.set_yscale("log")
    ax.set_xlabel("pseudo-label error rate ρ")
    ax.set_ylabel("L_l(pess) − L_l(opt)  (nats)")
    ax.set_title("Pessimistic–optimistic gap (dots: seeds, line: mean)", fontsize=8, color=INK)
    fig.suptitle(f"Figure 1 protocol, FashionMNIST CNN, 3 seeds ({len(ok)}/{len(rows)} cells with a fitted anchor)",
                 fontsize=8, color=INK)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "figure1_fmnist.png"))
    print("wrote", OUT, len(ok), "/", len(rows), "cells")


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


def dynamics():
    """Feedback along the main training runs: labelled loss of the anchor student and of the
    admissible student the feedback is taken at, and the cosine between the feedback actually
    used and the optimistic feedback at the same anchor (mean over seeds)."""
    R = _load("results/fmnist-ssl-main-*/results/*.json")
    runs = defaultdict(list)
    for n, r in R.items():
        m = re.match(r"(.+)_(\d+)_s(\d+)$", n)
        if m:
            runs[(m.group(1), int(m.group(2)))].append(r["hist"])

    def mean_curve(hs, key):
        its = sorted(set.intersection(*[{d["iter"] for d in h if d.get(key) is not None} for h in hs]))
        return its, [np.mean([next(d[key] for d in h if d["iter"] == i) for h in hs]) for i in its]
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.4))
    for j, npc in enumerate((4, 10)):
        ax = axes[0, j]
        for meth, key, lab, ls in [("mpl", "lab_loss_anchor", "MPL: anchor student", "-"),
                                   ("pbpl", "lab_loss_anchor", "PBPL: anchor student", "-"),
                                   ("pbpl", "lab_loss_adv", "PBPL: pessimistic student (r = 0.1)", "--"),
                                   ("ebpl", "lab_loss_adv", "EBPL: perturbed student (r = 3)", ":")]:
            hs = runs.get((meth, npc), [])
            if hs:
                its, y = mean_curve(hs, key)
                ax.plot(its, y, color=COL[meth], ls=ls, lw=1.6, label=lab)
        ax.set_yscale("log")
        ax.minorticks_off()
        ax.set_title(f"{10 * npc} labels", fontsize=8, color=INK)
        ax.set_ylabel("held-out labelled loss\nof the student")
        ax = axes[1, j]
        for meth, lab in [("pbpl", "PBPL"), ("ebpl", "EBPL")]:
            hs = runs.get((meth, npc), [])
            if hs:
                its, y = mean_curve(hs, "cos_c_copt")
                ax.plot(its, y, color=COL[meth], lw=1.6, label=lab)
        ax.set_ylabel("cos(feedback used,\noptimistic feedback)")
        ax.set_xlabel("iteration")
    axes[0, 1].legend(fontsize=6.5, loc="upper right")
    axes[1, 0].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "main_dynamics.png"))
    print("wrote", OUT)


def replication():
    """40 labels: test accuracy minus FixMatch on the same seed, for the seeds of the main table
    (1-5) and the replication seeds (6-10).  Dots: seeds; bar and number: mean."""
    R = _load("results/fmnist-ssl-*/results/*.json")
    res = defaultdict(dict)
    for n, r in R.items():
        m = re.match(r"(.+)_4_s(\d+)$", n)
        if m and r.get("final", {}).get("iter", 0) >= r["args"].get("iters", 1):
            res[m.group(1)][int(m.group(2))] = 100 * r["final"]["test_ema"]
    meths = [("fmtuned", "FixMatch\ntuned"), ("mpl", "MPL"), ("pbpl", "PBPL")]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7), sharey=True)
    for ax, (lo, hi), title in zip(axes, ((1, 5), (6, 10)),
                                   ("Seeds 1–5 (main table)", "Seeds 6–10 (replication)")):
        base = res["fixmatch"]
        for i, (meth, _) in enumerate(meths):
            common = [s for s in range(lo, hi + 1) if s in res[meth] and s in base]
            dif = [res[meth][s] - base[s] for s in common]
            ax.scatter([i] * len(dif), dif, s=22, color=COL[meth], zorder=3, edgecolor="white", lw=0.8)
            if dif:
                mu = float(np.mean(dif))
                ax.plot([i - 0.22, i + 0.22], [mu] * 2, color=INK, lw=1.5, zorder=4)
                ax.annotate(f"{mu:+.1f}", (i + 0.24, mu), va="center", fontsize=7, color=INK)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xticks(range(len(meths)))
        ax.set_xticklabels([lab for _, lab in meths])
        ax.set_xlim(-0.5, len(meths) - 0.3)
        ax.set_title(title, fontsize=8, color=INK)
    axes[0].set_ylabel("test acc. − FixMatch, same seed (pp)")
    fig.suptitle("40 labels, final iterate: paired difference to default FixMatch", fontsize=8, color=INK)
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, "replication_paired.png"))
    print("wrote", os.path.join(OUT, "replication_paired.png"))


if __name__ == "__main__":
    globals()[sys.argv[1]]()
