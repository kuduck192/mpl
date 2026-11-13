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
