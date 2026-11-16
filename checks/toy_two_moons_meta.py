#!/usr/bin/env python3
"""Two Moons toy: pseudo-labels as free optimisation variables, set by the argmin of the
first-order meta signal of one random student.

Bilevel pseudo-labeling with the teacher replaced by free pseudo-labels t_i (one per
unlabeled point, separate from any network).  The student theta is trained only on
(X_u, t), never on the labels.  To first order in the student step size eta,

    L_l(theta - eta * grad_theta L_u(t, theta)) = L_l(theta) + (eta / N_u) sum_i <t_i, c_i> + const,
    c_i = J_f(x_i; theta) grad_theta L_l(theta)          (one JVP),

so the meta objective is linear in t and its minimiser over the simplex is the one-hot
pseudo-label  PL_i = argmin_k c_ik:  the class whose logit at x_i rises most along a descent
step on the labeled loss.  In tangent-kernel form c_i = sum_j K(x_i, x_j) (p_j - y_j), i.e.
the pseudo-label is a vote of the labeled residuals weighted by the student's tangent kernel.

Rules (one random student = one random initialisation of the MLP):
  dynamic : PL re-set to argmin c at the current student before every student step
            (MPL with free pseudo-labels and an infinitely aggressive teacher step);
  frozen  : PL = argmin c at the random initial student, kept fixed; the same student
            trains on it.
References: label spreading on a kNN graph (manifold-aware pseudo-labels), a student trained
on the labeled points only, a student trained on the true labels of X_u, and the argmin of the
meta signal averaged over 64 random students.

  python checks/toy_two_moons_meta.py            # runs (4 processes, a few minutes on CPU)
  python checks/toy_two_moons_meta.py --plot     # figures from results/toy-two-moons/results.json
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "results", "toy-two-moons", "results.json")
FIG = os.path.join(ROOT, "docs", "figures")
LOG_ITERS = [0, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000]


# ----------------------------------------------------------------------------------------
# data, model
# ----------------------------------------------------------------------------------------
def moons(n, noise, rng):
    n0 = n // 2
    t0, t1 = rng.uniform(0, math.pi, n0), rng.uniform(0, math.pi, n - n0)
    X = np.concatenate([np.stack([np.cos(t0), np.sin(t0)], 1), np.stack([1 - np.cos(t1), 0.5 - np.sin(t1)], 1)])
    y = np.concatenate([np.zeros(n0, int), np.ones(n - n0, int)])
    return X + noise * rng.standard_normal(X.shape), y


def make_data(n_l, lab_draw, noise=0.1, n_u=1000, n_test=4000):
    """One fixed draw of the moons (unlabeled, test, a disjoint labeled pool); the labeled
    points (n_l per class) are drawn from the pool by lab_draw."""
    rng = np.random.default_rng(0)
    Xu, yu = moons(n_u, noise, rng)
    Xt, yt = moons(n_test, noise, rng)
    Xp, yp = moons(400, noise, rng)
    rl = np.random.default_rng(10_000 + lab_draw)
    il = np.concatenate([rl.choice(np.where(yp == k)[0], n_l, replace=False) for k in (0, 1)])
    return Xu, yu, Xt, yt, Xp[il], yp[il]


def mlp(width=64):
    return nn.Sequential(nn.Linear(2, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 2))


def tensors(Xu, Xt, Xl):
    mu, sd = Xu.mean(0), Xu.std(0)
    return [torch.tensor((a - mu) / sd, dtype=torch.float32) for a in (Xu, Xt, Xl)]


def meta_signal(net, Xu_, Xl_, yl_):
    """c = J_f(X_u) grad L_l at the current student (N_u x 2), and L_l."""
    params = {n: p.detach() for n, p in net.named_parameters()}
    p_ = {n: p.clone().requires_grad_(True) for n, p in params.items()}
    Ll = F.cross_entropy(functional_call(net, p_, (Xl_,)), yl_)
    v = dict(zip(p_.keys(), torch.autograd.grad(Ll, list(p_.values()))))
    _, c = jvp(lambda q: functional_call(net, q, (Xu_,)), (params,), (v,))
    return c, Ll.item()


# ----------------------------------------------------------------------------------------
# runs
# ----------------------------------------------------------------------------------------
def run(job):
    """One student (random init `init`) trained for `iters` full-batch SGD steps on the
    pseudo-labels given by `rule`."""
    torch.set_num_threads(1)
    n_l, lab_draw, init, rule, iters, lr = (job[k] for k in ("n_l", "lab_draw", "init", "rule", "iters", "lr"))
    Xu, yu, Xt, yt, Xl, yl = make_data(n_l, lab_draw)
    Xu_, Xt_, Xl_ = tensors(Xu, Xt, Xl)
    yu_, yt_, yl_ = torch.tensor(yu), torch.tensor(yt), torch.tensor(yl)
    torch.manual_seed(1000 * lab_draw + init)
    net = mlp()
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    pl, hist = None, []
    for k in range(iters + 1):
        if rule in ("dynamic", "frozen"):
            c, Ll = meta_signal(net, Xu_, Xl_, yl_)
            new = c.argmin(1)
            flip = float((new != pl).float().mean()) if pl is not None and rule == "dynamic" else float("nan")
            if rule == "dynamic" or pl is None:
                pl = new
            c_rms = c.pow(2).mean().sqrt().item()
        else:  # references: supervised-only (labeled points) and oracle (true labels of X_u)
            with torch.no_grad():
                Ll = F.cross_entropy(net(Xl_), yl_).item()
            pl, flip, c_rms = yu_, float("nan"), float("nan")
        if k in LOG_ITERS:
            with torch.no_grad():
                pu, pt = net(Xu_).argmax(1), net(Xt_).argmax(1)
            hist.append({"it": k, "pl_acc": (pl == yu_).float().mean().item() if rule != "sup" else float("nan"),
                         "stu_acc": (pt == yt_).float().mean().item(), "stu_acc_u": (pu == yu_).float().mean().item(),
                         "agree": (pu == pl).float().mean().item() if rule != "sup" else float("nan"),
                         "Ll": Ll, "flip": flip, "frac0": (pl == 0).float().mean().item(), "c_rms": c_rms})
        if k == iters:
            break
        opt.zero_grad()
        loss = F.cross_entropy(net(Xl_), yl_) if rule == "sup" else F.cross_entropy(net(Xu_), pl)
        loss.backward()
        opt.step()
    out = dict(job, hist=hist)
    if rule in ("dynamic", "frozen") and job.get("keep_pl"):
        out["pl"] = pl.tolist()
        with torch.no_grad():  # decision function on a grid, for the snapshot figure
            g = np.stack(np.meshgrid(np.linspace(-1.6, 2.6, 120), np.linspace(-1.1, 1.6, 90)), -1).reshape(-1, 2)
            out["grid_p1"] = F.softmax(net(tensors(Xu, g, Xl)[1]), 1)[:, 1].reshape(90, 120).tolist()
    return out


def label_spreading(Xu, Xl, yl, k=10, alpha=0.99):
    """Zhou et al. (2004) on a symmetric kNN graph with Gaussian weights (median kNN distance)."""
    X = np.concatenate([Xl, Xu])
    n = len(X)
    D2 = ((X[:, None] - X[None]) ** 2).sum(-1)
    nn_ = np.argsort(D2, 1)[:, 1:k + 1]
    rows = np.repeat(np.arange(n), k)
    s2 = np.median(D2[rows, nn_.ravel()])
    W = np.zeros((n, n))
    W[rows, nn_.ravel()] = np.exp(-D2[rows, nn_.ravel()] / (2 * s2))
    W = np.maximum(W, W.T)
    d = W.sum(1)
    S = W / np.sqrt(d[:, None] * d[None])
    Y = np.zeros((n, 2))
    Y[np.arange(len(yl)), yl] = 1
    return np.linalg.solve(np.eye(n) - alpha * S, Y)[len(yl):].argmax(1)


def statics(n_l, lab_draw, n_students=64):
    """Per labeled draw: label spreading, and the argmin of the meta signal of each of 64 random
    students and of their average (signals normalised to unit RMS before averaging)."""
    torch.set_num_threads(1)
    Xu, yu, Xt, yt, Xl, yl = make_data(n_l, lab_draw)
    Xu_, Xt_, Xl_ = tensors(Xu, Xt, Xl)
    yl_ = torch.tensor(yl)
    accs, C = [], 0
    for s in range(n_students):
        torch.manual_seed(50_000 + 1000 * lab_draw + s)
        c, _ = meta_signal(mlp(), Xu_, Xl_, yl_)
        c = c - c.mean(1, keepdim=True)
        C = C + c / c.pow(2).mean().sqrt()
        accs.append(float((c.argmin(1).numpy() == yu).mean()))
    ls = label_spreading(Xu, Xl, yl)
    return {"n_l": n_l, "lab_draw": lab_draw, "ls_acc": float((ls == yu).mean()), "ls_pl": ls.tolist(),
            "single_accs": accs, "avg_acc": float((C.argmin(1).numpy() == yu).mean()),
            "Xl": Xl.tolist(), "yl": yl.tolist()}


def _round(o, sig=5):
    """Round floats to `sig` significant digits (keeps the results file small)."""
    if isinstance(o, float):
        return float(f"{o:.{sig}g}") if math.isfinite(o) else o
    if isinstance(o, list):
        return [_round(x, sig) for x in o]
    if isinstance(o, dict):
        return {k: _round(v, 3 if k == "grid_p1" else sig) for k, v in o.items()}
    return o


def main_runs(args):
    jobs = []
    for n_l in (1, 5):
        for d in range(args.lab_draws):
            for i in range(args.inits):
                for rule in ("dynamic", "frozen"):
                    jobs.append({"n_l": n_l, "lab_draw": d, "init": i, "rule": rule, "iters": args.iters,
                                 "lr": args.lr, "keep_pl": i == 0})
            for i in range(2):
                jobs.append({"n_l": n_l, "lab_draw": d, "init": i, "rule": "sup", "iters": args.iters, "lr": args.lr})
            jobs.append({"n_l": n_l, "lab_draw": d, "init": 0, "rule": "oracle", "iters": args.iters, "lr": args.lr})
    t0 = time.time()
    with Pool(args.procs) as pool:
        runs = pool.map(run, jobs, chunksize=1)
        st = pool.starmap(statics, [(n_l, d) for n_l in (1, 5) for d in range(args.lab_draws)])
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    meta = {"iters": args.iters, "lr": args.lr, "width": 64, "n_u": 1000, "n_test": 4000, "noise": 0.1,
            "minutes": (time.time() - t0) / 60}
    json.dump(_round({"meta": meta, "runs": runs, "statics": st}), open(OUT_JSON, "w"))
    print(f"{len(runs)} runs in {meta['minutes']:.1f} min -> {OUT_JSON}")


# ----------------------------------------------------------------------------------------
# summary and figures
# ----------------------------------------------------------------------------------------
def summary(R):
    runs, st = R["runs"], R["statics"]
    for n_l in (1, 5):
        print(f"\n=== {n_l} labeled point(s) per class")
        S = [s for s in st if s["n_l"] == n_l]
        single = np.concatenate([s["single_accs"] for s in S])
        per_draw = lambda key: ", ".join(f"{s[key]:.3f}" for s in S)  # noqa: E731 (Python 3.11 f-strings)
        print(f"label spreading PL acc: {np.mean([s['ls_acc'] for s in S]):.3f} "
              f"(per draw {per_draw('ls_acc')})")
        print(f"argmin of the meta signal, one random student: mean {single.mean():.3f}, median "
              f"{np.median(single):.3f}, range {single.min():.3f}-{single.max():.3f}, "
              f"share of students at <= 0.55: {(single <= 0.55).mean():.2f}")
        print(f"argmin of the average over 64 random students: {np.mean([s['avg_acc'] for s in S]):.3f} "
              f"(per draw {per_draw('avg_acc')})")
        for rule in ("dynamic", "frozen", "sup", "oracle"):
            rs = [r for r in runs if r["n_l"] == n_l and r["rule"] == rule]
            fin = [r["hist"][-1] for r in rs]
            ini = [r["hist"][0] for r in rs]
            line = (f"{rule:8s} n={len(rs):3d} final student acc {np.mean([h['stu_acc'] for h in fin]):.3f} "
                    f"(range {min(h['stu_acc'] for h in fin):.3f}-{max(h['stu_acc'] for h in fin):.3f})")
            if rule in ("dynamic", "frozen"):
                d_pl = np.array([f["pl_acc"] - i["pl_acc"] for f, i in zip(fin, ini)])
                gap = np.array([h["stu_acc"] - h["pl_acc"] for h in fin])
                line += (f" | PL acc start {np.mean([h['pl_acc'] for h in ini]):.3f} -> end "
                         f"{np.mean([h['pl_acc'] for h in fin]):.3f} (change {d_pl.mean():+.3f}, |change| max "
                         f"{np.abs(d_pl).max():.3f}) | student - PL {gap.mean():+.3f} | agree "
                         f"{np.mean([h['agree'] for h in fin]):.3f} | L_l end median {np.median([h['Ll'] for h in fin]):.1e}")
                if rule == "dynamic":
                    late = [h["flip"] for r in rs for h in r["hist"] if h["it"] >= 500]
                    early = [h["flip"] for r in rs for h in r["hist"] if 1 <= h["it"] <= 50]
                    line += f" | flip rate it 1-50 {np.nanmean(early):.4f}, it >= 500 {np.nanmean(late):.4f}"
            print(line)
        dyn = {(r["lab_draw"], r["init"]): r["hist"][-1] for r in runs if r["n_l"] == n_l and r["rule"] == "dynamic"}
        frz = {(r["lab_draw"], r["init"]): r["hist"][-1] for r in runs if r["n_l"] == n_l and r["rule"] == "frozen"}
        d = np.array([dyn[k]["stu_acc"] - frz[k]["stu_acc"] for k in dyn])
        print(f"dynamic - frozen, same student: {d.mean():+.4f} +- {d.std(ddof=1):.4f}, dynamic better in "
              f"{(d > 0.001).sum()}/{len(d)}, worse in {(d < -0.001).sum()}/{len(d)}")
        pl0 = np.array([r["hist"][0]["pl_acc"] for r in runs if r["n_l"] == n_l and r["rule"] == "dynamic"])
        stu = np.array([r["hist"][-1]["stu_acc"] for r in runs if r["n_l"] == n_l and r["rule"] == "dynamic"])
        print(f"corr(PL acc of the random initial student, final student acc), dynamic: {np.corrcoef(pl0, stu)[0, 1]:.3f}")


def plots(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # validated all-pairs (scatter) on white
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    COL = {"dynamic": BLUE, "frozen": ORANGE}
    plt.rcParams.update({
        "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 2.0, "savefig.dpi": 200, "figure.facecolor": "white"})
    runs, st = R["runs"], R["statics"]
    os.makedirs(FIG, exist_ok=True)

    # ---- figure 1: pseudo-label accuracy vs student accuracy ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    for ax, n_l in zip(axes, (1, 5)):
        lo = 0.38
        ax.plot([lo, 1], [lo, 1], color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
        ax.annotate("student = pseudo-labels", (0.62, 0.62), xytext=(4, -4), textcoords="offset points",
                    ha="left", va="top", fontsize=6.5, color=INK2)
        S = [s for s in st if s["n_l"] == n_l]
        ls = np.mean([s["ls_acc"] for s in S])
        ax.axhline(ls, color=AQUA, lw=1.4, zorder=1)
        ax.annotate(f"label spreading pseudo-labels {ls:.3f}", (lo + 0.005, ls), xytext=(0, 3),
                    textcoords="offset points", fontsize=6.5, color=INK)
        sup = np.mean([r["hist"][-1]["stu_acc"] for r in runs if r["n_l"] == n_l and r["rule"] == "sup"])
        ax.axvline(sup, color=INK2, lw=1.0, ls=(0, (1, 2)), zorder=1)
        ax.annotate(f"labeled points only\n{sup:.3f}", (sup, lo + 0.01), xytext=(3, 0), textcoords="offset points",
                    fontsize=6.5, color=INK2, va="bottom")
        for r in runs:
            if r["n_l"] != n_l or r["rule"] != "dynamic":
                continue
            h = [x for x in r["hist"] if x["it"] >= 1]
            ax.plot([x["stu_acc"] for x in h], [x["pl_acc"] for x in h], color=BLUE, lw=0.6, alpha=0.30, zorder=2)
        for rule, mk in (("frozen", "o"), ("dynamic", "o")):
            rs = [r for r in runs if r["n_l"] == n_l and r["rule"] == rule]
            x = [r["hist"][-1]["stu_acc"] for r in rs]
            y = [r["hist"][-1]["pl_acc"] for r in rs]
            if rule == "frozen":
                ax.scatter(x, y, s=22, facecolors="white", edgecolors=ORANGE, linewidths=1.1, zorder=3)
            else:
                ax.scatter(x, y, s=12, color=BLUE, edgecolors="white", linewidths=0.4, zorder=4)
        ax.set_xlim(lo, 1.0)
        ax.set_ylim(lo, 1.02)
        ax.set_xlabel("student test accuracy (end of training)")
        ax.set_title(f"{n_l} labeled point{'s' if n_l > 1 else ''} per class, "
                     f"{len([r for r in runs if r['n_l'] == n_l and r['rule'] == 'dynamic'])} random students",
                     fontsize=8, color=INK, loc="left")
    axes[0].set_ylabel("pseudo-label accuracy on X_u")
    handles = [Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="re-set every step (dynamic), end"),
               Line2D([], [], color=BLUE, lw=0.8, alpha=0.5, label="dynamic, path from step 1"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=ORANGE, ms=5,
                      label="set once at the random initial student (frozen)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Pseudo-labels = argmin of the first-order meta signal of the student (Two Moons)",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    p1 = os.path.join(FIG, "toy_moons_pl_vs_student.png")
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)

    # ---- figure 2: dynamics and one snapshot ---------------------------------------------
    fig = plt.figure(figsize=(7.4, 7.6))
    gs = fig.add_gridspec(3, 3, height_ratios=(1, 1, 1.05))

    def band(ax, rule, key, n_l, ls_, label=None):
        rs = [r for r in runs if r["n_l"] == n_l and r["rule"] == rule]
        its = np.array([x["it"] for x in rs[0]["hist"] if x["it"] >= 1])
        M = np.array([[x[key] for x in r["hist"] if x["it"] >= 1] for r in rs])
        med = np.nanmedian(M, 0)
        ax.fill_between(its, np.nanpercentile(M, 25, 0), np.nanpercentile(M, 75, 0), color=COL[rule], alpha=0.12, lw=0)
        ax.plot(its, med, color=COL[rule], ls=ls_, lw=1.6)
        if label:
            ax.annotate(label, (its[-1], med[-1]), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=6.5, color=INK)
        return med

    for row, n_l in enumerate((1, 5)):
        tag = f"{n_l} labeled point{'s' if n_l > 1 else ''} per class"
        ax = fig.add_subplot(gs[row, 0])
        band(ax, "dynamic", "pl_acc", n_l, "-")
        band(ax, "dynamic", "stu_acc", n_l, (0, (3, 2)))
        band(ax, "frozen", "stu_acc", n_l, (0, (3, 2)))
        ax.set_xscale("log")
        ax.set_ylim(0.38, 1.0)
        ax.set_xlim(1, 3000)
        ax.set_title(f"{'ad'[row]}. accuracy, {tag}", fontsize=8, loc="left", color=INK)
        ax.set_xlabel("student step")
        if row == 0:
            ax.legend(handles=[Line2D([], [], color=BLUE, lw=1.6, label="pseudo-labels, dynamic"),
                               Line2D([], [], color=BLUE, lw=1.6, ls=(0, (3, 2)), label="student, dynamic"),
                               Line2D([], [], color=ORANGE, lw=1.6, ls=(0, (3, 2)), label="student, frozen")],
                      fontsize=6.5, loc="lower right")
        ax = fig.add_subplot(gs[row, 1])
        band(ax, "dynamic", "Ll", n_l, "-")
        band(ax, "frozen", "Ll", n_l, "-")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-8, 30)
        ax.set_xlim(1, 3000)
        ax.set_title(f"{'be'[row]}. labeled loss of the student", fontsize=8, loc="left", color=INK)
        ax.set_xlabel("student step")
        if row == 0:
            ax.legend(handles=[Line2D([], [], color=BLUE, lw=1.6, label="dynamic"),
                               Line2D([], [], color=ORANGE, lw=1.6, label="frozen")], fontsize=6.5, loc="lower left")
        ax = fig.add_subplot(gs[row, 2])
        band(ax, "dynamic", "flip", n_l, "-")
        ax.set_xscale("log")
        ax.set_xlim(1, 3000)
        ax.set_ylim(0, 0.45)
        ax.set_title(f"{'cf'[row]}. pseudo-labels flipped per step", fontsize=8, loc="left", color=INK)
        ax.set_xlabel("student step")
    # snapshot: one labeled draw, initial student 0; frozen vs dynamic vs label spreading
    d = int(np.argsort([np.median([r["hist"][-1]["stu_acc"] for r in runs if r["n_l"] == 1 and
                                   r["rule"] == "dynamic" and r["lab_draw"] == dd]) for dd in
                        sorted({r["lab_draw"] for r in runs})])[len({r['lab_draw'] for r in runs}) // 2])
    Xu, yu, _, _, Xl, yl = make_data(1, d)
    S = next(s for s in st if s["n_l"] == 1 and s["lab_draw"] == d)
    panels = [("frozen", "g. frozen (set at the random init)"), ("dynamic", "h. dynamic (end of training)"),
              ("ls", "i. label spreading")]
    gx, gy = np.meshgrid(np.linspace(-1.6, 2.6, 120), np.linspace(-1.1, 1.6, 90))
    for j, (rule, title) in enumerate(panels):
        ax = fig.add_subplot(gs[2, j])
        if rule == "ls":
            pl = np.array(S["ls_pl"])
        else:
            r = next(r for r in runs if r["n_l"] == 1 and r["rule"] == rule and r["lab_draw"] == d and r["init"] == 0)
            pl = np.array(r["pl"])
            ax.contour(gx, gy, np.array(r["grid_p1"]), levels=[0.5], colors=[INK2], linewidths=1.0)
        cols = np.where(pl == 0, BLUE, ORANGE)
        wrong = pl != yu
        ax.scatter(Xu[~wrong, 0], Xu[~wrong, 1], s=3, c=cols[~wrong], lw=0)
        ax.scatter(Xu[wrong, 0], Xu[wrong, 1], s=7, c=cols[wrong], marker="x", lw=0.6)
        for k_, c_ in ((0, BLUE), (1, ORANGE)):
            m = np.array(yl) == k_
            ax.scatter(np.array(Xl)[m, 0], np.array(Xl)[m, 1], s=60, c=c_, edgecolors=INK, linewidths=1.0, zorder=5)
        ax.set_title(f"{title}\npseudo-label acc {(pl == yu).mean():.3f}", fontsize=7.5, loc="left", color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.set_aspect("equal")
    fig.text(0.01, 0.005, "Bands: interquartile range over 40 random students; lines: median.  Bottom row (1 labeled point "
             "per class, one student): colour = pseudo-label, x = wrong pseudo-label, large dots = labeled points, "
             "line = student decision boundary.", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    p2 = os.path.join(FIG, "toy_moons_dynamics.png")
    fig.savefig(p2, bbox_inches="tight")
    plt.close(fig)
    print(p1, p2, f"(snapshot: labeled draw {d})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--lab_draws", type=int, default=5)
    ap.add_argument("--inits", type=int, default=8)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    if not a.plot:
        main_runs(a)
    R = json.load(open(OUT_JSON))
    summary(R)
    plots(R)
