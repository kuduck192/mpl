#!/usr/bin/env python3
"""Two Moons toy: free pseudo-labels anchored to a FixMatch teacher by a KL term.

Same data and random students as checks/toy_two_moons_meta.py / toy_two_moons_unroll.py.  A
teacher network phi is trained with the labeled loss and the FixMatch loss (no meta term); its
predictions q = softmax(f_phi(X_u)) are the anchor.  The pseudo-labels t are free variables,

    t* = argmin_t  meta_K(t) + lam * mean_i KL(t_i || q_i),

where meta_K is the meta objective of the student theta_0 (labeled loss after K unrolled steps on
(X_u, t), K = 1, 2, ..., or after convergence, IFT), entering through its unit-RMS signal
g = dF/dt / rms (the normalisation the main experiments use).  For a linear meta loss the
solution is t_i ~ q_i exp(-g_i / lam): the teacher's pseudo-labels tilted by the meta signal.
It is computed by proximal mirror descent from t = q,

    z <- (z + beta * lam * log q - beta * g(t)) / (1 + beta * lam),        t = softmax(z),

whose fixed point is exactly z = log q - g / lam (computed as the damped fixed-point iteration
z <- (1 - gamma) z + gamma (log q - g(t) / lam), gamma = 0.5, which reaches it in about 20 steps for
any lam > 0; the last-step residual is logged).  lam = 0 is the free problem of
toy_two_moons_unroll.py started at q; lam = inf is t = q (the teacher).  Logged per K and lam:
pseudo-label accuracy, flips against the teacher split into corrections (teacher wrong, t right)
and corruptions (teacher right, t wrong), KL(t || q), the meta objective at t, at q and at the
true labels, the accuracy of a fresh copy of the student trained on argmax t, and for the IFT the
conjugate-gradient iterations, residuals and the stationarity of the inner solution.

  python checks/toy_two_moons_anchor.py          # runs (4 processes)
  python checks/toy_two_moons_anchor.py --plot   # summary, logs and figures from results/toy-two-moons/anchor.json
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
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy_two_moons_meta import FIG, ROOT, _round, make_data, mlp, tensors  # noqa: E402
from toy_two_moons_unroll import Inner, train_on, unrolled  # noqa: E402

OUT_JSON = os.path.join(ROOT, "results", "toy-two-moons", "anchor.json")
KS = ["1", "2", "5", "10", "20", "50", "IFT"]
LAMS = [0.0, 0.1, 1.0, 10.0]  # lam = inf (t = q) is recorded separately
SNAP = (1, 0, 0)  # (n_l, labeled draw, init) whose IFT log is printed


# ----------------------------------------------------------------------------------------
# FixMatch teacher (labeled loss + FixMatch loss)
# ----------------------------------------------------------------------------------------
def fixmatch_teacher(spec):
    torch.set_num_threads(1)
    n_l, d = spec["n_l"], spec["lab_draw"]
    Xu, yu, Xt, yt, Xl, yl = make_data(n_l, d)
    Xu_, Xt_, Xl_ = tensors(Xu, Xt, Xl)
    yu_, yt_, yl_ = torch.tensor(yu), torch.tensor(yt), torch.tensor(yl)
    torch.manual_seed(77_000 + 1000 * d)
    net = mlp()
    gen = torch.Generator().manual_seed(99_000 + 1000 * d)
    opt = torch.optim.SGD(net.parameters(), lr=spec["lr"], momentum=0.9)
    for _ in range(spec["iters"]):
        ib = torch.randint(0, len(yu), (spec["bu"],), generator=gen)
        xb = Xu_[ib]
        xw = xb + spec["sigma_w"] * torch.randn(xb.shape, generator=gen)
        xs = xb + spec["sigma_s"] * torch.randn(xb.shape, generator=gen)
        with torch.no_grad():
            conf, hard = F.softmax(net(xw), 1).max(1)
            mask = (conf >= spec["tau"]).float()
        loss = F.cross_entropy(net(Xl_), yl_) + spec["lam_u"] * (F.cross_entropy(net(xs), hard, reduction="none") * mask).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        q = F.softmax(net(Xu_), 1)
        return {"n_l": n_l, "lab_draw": d, "q": q.tolist(), "pl_acc": (q.argmax(1) == yu_).float().mean().item(),
                "test_acc": (net(Xt_).argmax(1) == yt_).float().mean().item(), "conf": q.max(1).values.mean().item()}


# ----------------------------------------------------------------------------------------
# anchored pseudo-labels
# ----------------------------------------------------------------------------------------
def unit(g):
    g = g - g.mean(1, keepdim=True)
    return g / g.pow(2).mean().sqrt().clamp_min(1e-30)


def prox_step(z, logq, g, lam, beta, gamma=0.5):
    """lam > 0: damped fixed-point step on the exact minimiser of <t, g> + lam KL(t || q), i.e.
    z <- (1 - gamma) z + gamma (log q - g / lam); lam = 0: plain mirror step z <- z - beta g."""
    if lam == 0:
        return z - beta * g
    return (1 - gamma) * z + gamma * (logq - g / lam)


def fp_residual(z_new, z_old):
    """Fixed-point residual: RMS change of the class-centred logits in the last step."""
    c = lambda a: a - a.mean(1, keepdim=True)  # noqa: E731
    return (c(z_new) - c(z_old)).pow(2).mean().sqrt().item()


def anchored_unrolled(net, Xu_, Xl_, yl_, K, logq, lam, eta, M, beta):
    z, resid = logq.clone(), float("nan")
    for _ in range(M):
        g, _ = unrolled(net, Xu_, Xl_, yl_, K, F.softmax(z, 1), eta)
        z_new = prox_step(z, logq, unit(g), lam, beta)
        resid, z = fp_residual(z_new, z), z_new
    t = F.softmax(z, 1)
    _, Fe = unrolled(net, Xu_, Xl_, yl_, K, t, eta, need_grad=False)
    return t, Fe, {"fp_residual": resid}


def anchored_ift(I, th0, logq, lam, eta, M, beta, rho, log=None, yu_=None):
    z = logq.clone()
    t = F.softmax(z, 1)
    th = I.train(th0, t, 1000, eta)
    lm = I.lam_max(th, t)
    its, res, stat = [], [], []
    for m in range(M):
        if m:
            th = I.train(th, t, 100, eta)
        b, Fi = I.lab(th)
        v, info = I.cg(th, t, b, rho * lm, "ggn")
        gn = I.grad_norm(th, t)
        its.append(info["cg_iters"])
        res.append(info["cg_relres"])
        stat.append(gn)
        if log is not None:
            log.append(f"  lam={lam:g} m={m:2d}  F_inf={Fi:.3e}  PL acc={(t.argmax(1) == yu_).float().mean().item():.3f}  "
                       f"KL(t||q)={(t * (t.clamp_min(1e-12).log() - logq)).sum(1).mean().item():.3f}  "
                       f"|grad L_u(theta*)|={gn:.1e}  CG {info['cg_iters']} it, rel.res {info['cg_relres']:.1e}")
        z_new = prox_step(z, logq, unit(I.signal(th, v)), lam, beta)
        resid, z = fp_residual(z_new, z), z_new
        t = F.softmax(z, 1)
    th = I.train(th, t, 300, eta)
    _, Fe = I.lab(th)
    return t, Fe, {"lam_max": lm, "cg_iters_median": float(np.median(its)), "cg_relres_max": float(max(res)),
                   "stationarity_median": float(np.median(stat)), "fp_residual": resid}


def job(spec):
    torch.set_num_threads(1)
    n_l, d, init, eta, M, beta, steps, rho = (spec[k] for k in ("n_l", "lab_draw", "init", "eta", "M", "beta", "steps", "rho"))
    Xu, yu, Xt, yt, Xl, yl = make_data(n_l, d)
    Xu_, Xt_, Xl_ = tensors(Xu, Xt, Xl)
    yu_, yt_, yl_ = torch.tensor(yu), torch.tensor(yt), torch.tensor(yl)
    q = torch.tensor(spec["q"])
    logq = q.clamp_min(1e-12).log()
    pq = q.argmax(1)
    torch.manual_seed(1000 * d + init)  # the same random student as the other toys
    net = mlp()
    init_state = {k: v.clone() for k, v in net.state_dict().items()}
    th0 = {n: p.detach().clone() for n, p in net.named_parameters()}
    truth = F.one_hot(yu_, 2).float()
    I = Inner(net, Xu_, Xl_, yl_)
    snap = (n_l, d, init) == SNAP
    out = {"n_l": n_l, "lab_draw": d, "init": init, "teacher_pl_acc": (pq == yu_).float().mean().item(),
           "train_teacher": train_on(init_state, Xu_, Xt_, Xl_, yl_, yt_, pq, steps, eta), "K": {}}
    for K in KS:
        if K == "IFT":
            th_q = I.train(th0, q, 1000, eta)
            th_t = I.train(th0, truth, 1000, eta)
            F_q, F_true = I.lab(th_q)[1], I.lab(th_t)[1]
        else:
            F_q = unrolled(net, Xu_, Xl_, yl_, int(K), q, eta, need_grad=False)[1]
            F_true = unrolled(net, Xu_, Xl_, yl_, int(K), truth, eta, need_grad=False)[1]
        rec = {"F_teacher": F_q, "F_true": F_true, "lam": {}}
        log = [] if (snap and K == "IFT") else None
        for lam in LAMS:
            t0 = time.time()
            if K == "IFT":
                t, Fe, info = anchored_ift(I, th0, logq, lam, eta, M, beta, rho, log, yu_)
            else:
                t, Fe, info = anchored_unrolled(net, Xu_, Xl_, yl_, int(K), logq, lam, eta, M, beta)
            pt = t.argmax(1)
            right_q, right_t = pq == yu_, pt == yu_
            rec["lam"][f"{lam:g}"] = dict(
                info, pl_acc=right_t.float().mean().item(), flips=(pt != pq).float().mean().item(),
                corrections=(~right_q & right_t).float().mean().item(), corruptions=(right_q & ~right_t).float().mean().item(),
                kl=(t * (t.clamp_min(1e-12).log() - logq)).sum(1).mean().item(), F=Fe,
                train=train_on(init_state, Xu_, Xt_, Xl_, yl_, yt_, pt, steps, eta), seconds=time.time() - t0)
        if log is not None:
            rec["log"] = log
        out["K"][K] = rec
    return out


def main_runs(a):
    tspec = {"iters": a.teacher_iters, "lr": 0.1, "bu": 128, "tau": 0.95, "lam_u": 1.0, "sigma_w": a.sigma_w,
             "sigma_s": a.sigma_s}
    t0 = time.time()
    with Pool(a.procs) as pool:
        teachers = pool.map(fixmatch_teacher, [dict(tspec, n_l=n_l, lab_draw=d) for n_l in (1, 5) for d in range(a.lab_draws)])
        Q = {(T["n_l"], T["lab_draw"]): T for T in teachers}
        specs = [{"n_l": n_l, "lab_draw": d, "init": i, "eta": a.eta, "M": a.M, "beta": a.beta, "steps": a.steps,
                  "rho": a.rho, "q": Q[(n_l, d)]["q"]} for n_l in (1, 5) for d in range(a.lab_draws) for i in range(a.inits)]
        runs = pool.map(job, specs, chunksize=1)
    meta = {"Ks": KS, "lams": LAMS, "eta": a.eta, "M": a.M, "beta": a.beta, "train_steps": a.steps, "rho": a.rho,
            "teacher": tspec, "minutes": (time.time() - t0) / 60}
    for T in teachers:
        T.pop("q")
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(_round({"meta": meta, "teachers": teachers, "runs": runs}), open(OUT_JSON, "w"))
    print(f"{len(runs)} students in {meta['minutes']:.1f} min -> {OUT_JSON}")


# ----------------------------------------------------------------------------------------
# summary and figure
# ----------------------------------------------------------------------------------------
def summary(R):
    runs = R["runs"]
    for n_l in (1, 5):
        rs = [r for r in runs if r["n_l"] == n_l]
        T = [t for t in R["teachers"] if t["n_l"] == n_l]
        print(f"\n=== {n_l} labeled point(s) per class: {len(T)} FixMatch teachers, PL acc "
              + ", ".join(f"{t['pl_acc']:.3f}" for t in T) + f"; {len(rs)} random students")
        print(f"  lam = inf (t = q): PL acc {np.mean([r['teacher_pl_acc'] for r in rs]):.3f}, student trained on q "
              f"{np.mean([r['train_teacher']['stu_acc'] for r in rs]):.3f}")
        print("     K   lam | PL acc (- teacher) | flips: corrections / corruptions | KL(t||q) | F: t / teacher / truth | "
              "student acc (- student on q)")
        for K in KS:
            for lam in LAMS:
                X = [r["K"][K]["lam"][f"{lam:g}"] for r in rs]
                dpl = np.array([x["pl_acc"] - r["teacher_pl_acc"] for x, r in zip(X, rs)])
                dst = np.array([x["train"]["stu_acc"] - r["train_teacher"]["stu_acc"] for x, r in zip(X, rs)])
                print(f"  {K:>4s} {lam:5g} | {np.mean([x['pl_acc'] for x in X]):.3f} ({dpl.mean():+.3f}, {(dpl > 0.002).sum()}/{(dpl < -0.002).sum()}) | "
                      f"{np.mean([x['corrections'] for x in X]):.3f} / {np.mean([x['corruptions'] for x in X]):.3f} | "
                      f"{np.median([x['kl'] for x in X]):.3f} | {np.median([x['F'] for x in X]):.1e} / "
                      f"{np.median([r['K'][K]['F_teacher'] for r in rs]):.1e} / {np.median([r['K'][K]['F_true'] for r in rs]):.1e} | "
                      f"{np.mean([x['train']['stu_acc'] for x in X]):.3f} ({dst.mean():+.3f}, {(dst > 0.002).sum()}/{(dst < -0.002).sum()})")
        for lam in LAMS[1:]:
            fr = [r["K"][K]["lam"][f"{lam:g}"]["fp_residual"] for r in rs for K in KS]
            print(f"  fixed-point residual (RMS logit change in the last step), lam = {lam:g}: median {np.median(fr):.1e}, max {max(fr):.1e}")
        X = [r["K"]["IFT"]["lam"][f"{lam:g}"] for r in rs for lam in LAMS]
        print(f"  IFT logs: lambda_max median {np.median([x['lam_max'] for x in X]):.2f}, CG iterations median "
              f"{np.median([x['cg_iters_median'] for x in X]):.0f}, rel. residual max {max(x['cg_relres_max'] for x in X):.1e}, "
              f"inner stationarity |grad L_u(theta*)| median {np.median([x['stationarity_median'] for x in X]):.1e}")
    snap = next((r for r in runs if (r["n_l"], r["lab_draw"], r["init"]) == SNAP), None)
    if snap:
        print(f"\nIFT log of one student (n_l, draw, init = {SNAP}; teacher PL acc {snap['teacher_pl_acc']:.3f}):")
        for line in snap["K"]["IFT"]["log"]:
            if " m= 0 " in line or " m=19 " in line or " m= 5 " in line:
                print(line)


def plots(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    BLUE, ORANGE = "#2a78d6", "#eb6834"
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({
        "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 2.0, "savefig.dpi": 200, "figure.facecolor": "white"})
    runs = R["runs"]
    seq = LinearSegmentedColormap.from_list("seq", ["#bcd5f3", BLUE, "#0b2f5e"])
    kcol = {K: (ORANGE if K == "IFT" else seq(i / 5)) for i, K in enumerate(KS)}
    xs = np.arange(len(LAMS) + 1)
    xl = [f"{lam:g}" for lam in LAMS] + ["inf\n(teacher)"]
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.0))
    for row, n_l in enumerate((1, 5)):
        rs = [r for r in runs if r["n_l"] == n_l]
        for j, (title, f, base) in enumerate((
                ("pseudo-label acc. - teacher", lambda x: x["pl_acc"], lambda r: r["teacher_pl_acc"]),
                ("corrections (solid) / corruptions (dashed)", None, None),
                ("student acc. - student trained on q", lambda x: x["train"]["stu_acc"], lambda r: r["train_teacher"]["stu_acc"]))):
            ax = axes[row, j]
            for K in KS:
                if f is None:
                    for key, ls_ in (("corrections", "-"), ("corruptions", (0, (3, 2)))):
                        ys = [np.mean([r["K"][K]["lam"][f"{lam:g}"][key] for r in rs]) for lam in LAMS] + [0.0]
                        ax.plot(xs, ys, color=kcol[K], ls=ls_, lw=1.2 if K != "IFT" else 1.6, marker="o", ms=2.5)
                else:
                    ys = [np.mean([f(r["K"][K]["lam"][f"{lam:g}"]) - base(r) for r in rs]) for lam in LAMS] + [0.0]
                    ax.plot(xs, ys, color=kcol[K], lw=1.2 if K != "IFT" else 1.6, marker="o", ms=2.5)
            if f is not None:
                ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xticks(xs)
            ax.set_xticklabels(xl)
            ax.set_xlabel("KL weight lam")
            ax.set_title(f"{'abcdef'[3 * row + j]}. {title}, {n_l} lab./class", fontsize=7.3, loc="left", color=INK)
    handles = [Line2D([], [], color=kcol[K], lw=1.4, marker="o", ms=3, label=(f"K = {K}" if K != "IFT" else "IFT (K = inf)"))
               for K in KS]
    fig.legend(handles=handles, loc="lower center", ncol=len(KS), fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
    fig.text(0.01, 0.045, "Means over 20 random students per row (5 labeled draws x 4 inits; one FixMatch teacher per draw).  "
             "lam = 0: 20 free mirror-descent steps from q (no KL, so only the step budget limits the move; lam = 0.1 "
             "reaches its fixed point and moves further); lam = inf: t = q.", fontsize=6.3, color=INK2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    p = os.path.join(FIG, "toy_moons_anchor.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--lab_draws", type=int, default=5)
    ap.add_argument("--inits", type=int, default=4)
    ap.add_argument("--eta", type=float, default=0.1)
    ap.add_argument("--M", type=int, default=20, help="proximal mirror-descent steps")
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--steps", type=int, default=1000, help="training steps of the student on the pseudo-labels")
    ap.add_argument("--rho", type=float, default=1e-2, help="IFT damping relative to lambda_max")
    ap.add_argument("--teacher_iters", type=int, default=3000)
    ap.add_argument("--sigma_w", type=float, default=0.03)
    ap.add_argument("--sigma_s", type=float, default=0.3)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    if not a.plot:
        main_runs(a)
    R = json.load(open(OUT_JSON))
    summary(R)
    plots(R)
