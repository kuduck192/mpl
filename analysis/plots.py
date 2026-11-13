#!/usr/bin/env python3
"""Curves from result JSONs.

  python analysis/plots.py curves  OUT.png KEY  run1.json run2.json ...
  python analysis/plots.py feedback OUT.png run.json [run2.json ...]   # c_norm vs c_opt_norm
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PALETTE = ["#2a6fdb", "#e0592a", "#2b9348", "#8e44ad", "#c0392b", "#7f8c8d", "#d4a017"]


def load(f):
    r = json.load(open(f))
    return r["args"]["name"], r["hist"]


def curves(out, key, files):
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=150)
    for i, f in enumerate(files):
        name, h = load(f)
        xs = [d["iter"] for d in h if key in d]
        ys = [d[key] for d in h if key in d]
        ax.plot(xs, ys, label=name, color=PALETTE[i % len(PALETTE)], lw=1.6)
    ax.set_xlabel("iteration")
    ax.set_ylabel(key)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out)


def feedback(out, files):
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=150)
    for i, f in enumerate(files):
        name, h = load(f)
        col = PALETTE[i % len(PALETTE)]
        xs = [d["iter"] for d in h if "c_norm" in d]
        ax.plot(xs, [d["c_norm"] for d in h if "c_norm" in d], color=col, lw=1.6, label=f"{name}: feedback used")
        if any("c_opt_norm" in d for d in h):
            ax.plot(xs, [d.get("c_opt_norm", float("nan")) for d in h if "c_norm" in d], color=col, lw=1.2,
                    ls="--", label=f"{name}: optimistic feedback at same anchor")
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("RMS of meta feedback c = J v")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "curves":
        curves(sys.argv[2], sys.argv[3], sys.argv[4:])
    elif cmd == "feedback":
        feedback(sys.argv[2], sys.argv[3:])


def degeneracy(out, files):
    """Degeneracy diagnostic: for each feedback rule, mean over seeds of the teacher's
    movement away from the supervised-only start and of the pseudo-label accuracy."""
    import collections
    import numpy as np
    groups = collections.defaultdict(list)
    for f in files:
        name, h = load(f)
        groups[name.rsplit("_s", 1)[0]].append(h)
    keys = [("drift", "||phi - phi_0||"), ("pl_changed", "pseudo-labels changed (frac.)"),
            ("unl_raw", "pseudo-label accuracy on X_u"), ("c_norm", "RMS meta feedback (raw)")]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.0), dpi=150)
    for gi, (g, hs) in enumerate(sorted(groups.items())):
        col = PALETTE[gi % len(PALETTE)]
        for ax, (k, lab) in zip(axes, keys):
            xs = [d["iter"] for d in hs[0] if k in d]
            ys = np.array([[d.get(k, np.nan) for d in h if k in d][:len(xs)] for h in hs])
            m = np.nanmean(ys, 0)
            ax.plot(xs, m, color=col, lw=1.6, label=g)
            if len(hs) > 1:
                ax.fill_between(xs, np.nanmin(ys, 0), np.nanmax(ys, 0), color=col, alpha=0.15, lw=0)
            ax.set_title(lab, fontsize=8)
            ax.grid(alpha=0.3)
            if k == "c_norm":
                ax.set_yscale("log")
    axes[0].legend(fontsize=7, frameon=False)
    for ax in axes:
        ax.set_xlabel("iteration", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__" and sys.argv[1] == "degeneracy":
    degeneracy(sys.argv[2], sys.argv[3:])
