#!/usr/bin/env python3
"""Two Moons toy: K-step unrolled and implicit (IFT) hypergradients for free pseudo-labels.

Same data, students and labeled draws as checks/toy_two_moons_meta.py.  Pseudo-labels t (soft,
one per unlabeled point) are free variables; the student theta is trained only on (X_u, t).

Unrolled (K = 1, 2, ..., 100): from the random student theta_0,
    theta_{k+1} = theta_k - eta * grad_theta L_u(t, theta_k),     F_K(t) = L_l(theta_K(t)),
and g^(K) = dF_K/dt by backpropagation through the K steps.  K = 1 is the one-step look-ahead of
meta pseudo labels (= the first-order signal of toy_two_moons_meta.py up to O(eta)).

Implicit (IFT, K = infinity): at the student trained to convergence on t, theta*(t),
    dF/dt_ik = (1/N) [J_i(theta*) v]_k + const_i,     (H + lambda I) v = grad L_l(theta*),
H = Hessian of L_u in theta (exact, may be indefinite) or its Gauss-Newton part G (positive
semi-definite, the main variant); v by conjugate gradient with Hessian- / GGN-vector products;
lambda = rho * lambda_max(H).  Logged: lambda_max, CG iterations, relative residual, negative
curvature, the inner gradient norm, and the cosine between the IFT signal and the signal of K
steps unrolled from theta* (unrolled gradient descent ~ IFT with damping ~ 1 / (eta K)).

Pseudo-label sets per student and per K (or IFT):
  static          : argmin of the signal at uniform t;
  from uniform    : M mirror-descent steps on the logits of t against F_K (or F_inf) from uniform t;
  from the truth  : the same optimisation started at the true labels.
F at the true labels is recorded too.  A fresh copy of the same random student is then trained
on each set (it never sees the labels) and its test accuracy is recorded.

  python checks/toy_two_moons_unroll.py          # runs (4 processes, about 20 minutes on CPU)
  python checks/toy_two_moons_unroll.py --plot   # summary, logs and figures from results/toy-two-moons/unroll.json
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp, vjp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from toy_two_moons_meta import FIG, ROOT, _round, label_spreading, make_data, mlp, tensors  # noqa: E402

OUT_JSON = os.path.join(ROOT, "results", "toy-two-moons", "unroll.json")
KS = [1, 2, 5, 10, 20, 50, 100]
RHOS = [1e-1, 1e-2, 1e-3]           # IFT damping, relative to lambda_max(H)
RHO_MAIN = 1e-2                     # damping of the IFT optimisation
K_FROM_STAR = [10, 100, 1000]       # unrolled from theta*, compared with the IFT signal
SNAP = (1, 1, 0)                    # (n_l, labeled draw, init) of the maps and of the printed logs


# ----------------------------------------------------------------------------------------
# unrolled hypergradient
# ----------------------------------------------------------------------------------------
def unrolled(net, Xu_, Xl_, yl_, K, t, eta, need_grad=True, start=None):
    """F_K(t) and, if need_grad, dF_K/dt through K unrolled student steps from the student's own
    parameters (or from `start`, a parameter dict)."""
    src = start if start is not None else {n: p for n, p in net.named_parameters()}
    th = {n: p.detach().clone().requires_grad_(True) for n, p in src.items()}
    t = t.detach().clone().requires_grad_(need_grad)
    for _ in range(K):
        Lu = -(t * F.log_softmax(functional_call(net, th, (Xu_,)), 1)).sum(1).mean()
        g = torch.autograd.grad(Lu, list(th.values()), create_graph=need_grad)
        th = {n: w - eta * gi for (n, w), gi in zip(th.items(), g)}
    Ll = F.cross_entropy(functional_call(net, th, (Xl_,)), yl_)
    return (torch.autograd.grad(Ll, t)[0] if need_grad else None), Ll.item()


# ----------------------------------------------------------------------------------------
# implicit hypergradient
# ----------------------------------------------------------------------------------------
def flat(d):
    return torch.cat([v.reshape(-1) for v in d.values()])


def unflat(x, like):
    out, i = {}, 0
    for n, v in like.items():
        out[n] = x[i:i + v.numel()].view_as(v)
        i += v.numel()
    return out


class Inner:
    """Inner problem min_theta L_u(t, theta) of one student and the linear algebra of the IFT."""

    def __init__(self, net, Xu_, Xl_, yl_):
        self.net, self.Xu, self.Xl, self.yl = net, Xu_, Xl_, yl_

    def Lu(self, th, t):
        return -(t * F.log_softmax(functional_call(self.net, th, (self.Xu,)), 1)).sum(1).mean()

    def train(self, th, t, steps, eta):
        th = {n: w.detach().clone() for n, w in th.items()}
        for _ in range(steps):
            p = {n: w.requires_grad_(True) for n, w in th.items()}
            g = torch.autograd.grad(self.Lu(p, t), list(p.values()))
            th = {n: (w - eta * gi).detach() for (n, w), gi in zip(p.items(), g)}
        return th

    def grad_norm(self, th, t):
        p = {n: w.detach().clone().requires_grad_(True) for n, w in th.items()}
        return flat(dict(zip(p, torch.autograd.grad(self.Lu(p, t), list(p.values()))))).norm().item()

    def lab(self, th):
        p = {n: w.detach().clone().requires_grad_(True) for n, w in th.items()}
        Ll = F.cross_entropy(functional_call(self.net, p, (self.Xl,)), self.yl)
        return flat(dict(zip(p, torch.autograd.grad(Ll, list(p.values()))))), Ll.item()

    def hvp(self, th, t, vec):
        """Exact Hessian of L_u times vec."""
        p = {n: w.detach().clone().requires_grad_(True) for n, w in th.items()}
        g = torch.autograd.grad(self.Lu(p, t), list(p.values()), create_graph=True)
        dot = sum((gi * vi).sum() for gi, vi in zip(g, unflat(vec, p).values()))
        return flat(dict(zip(p, torch.autograd.grad(dot, list(p.values())))))

    def ggnvp(self, th, t, vec):
        """Gauss-Newton part of the Hessian of L_u times vec: J^T (diag p - p p^T) J vec / N."""
        p0 = {n: w.detach() for n, w in th.items()}
        f = lambda q: functional_call(self.net, q, (self.Xu,))  # noqa: E731
        z, Jv = jvp(f, (p0,), (unflat(vec, p0),))
        pr = F.softmax(z, 1)
        u = (pr * Jv - pr * (pr * Jv).sum(1, keepdim=True)) / z.shape[0]
        _, back = vjp(f, p0)
        return flat(back(u)[0])

    def lam_max(self, th, t, iters=20):
        v = torch.randn(sum(w.numel() for w in th.values()), generator=torch.Generator().manual_seed(0))
        v = v / v.norm()
        lam = 0.0
        for _ in range(iters):
            w = self.hvp(th, t, v)
            lam = (v @ w).item()
            v = w / w.norm()
        return lam

    def cg(self, th, t, b, damp, curvature="ggn", iters=300, tol=1e-4):
        Av = self.ggnvp if curvature == "ggn" else self.hvp
        x, r = torch.zeros_like(b), b.clone()
        p, rs, neg, k = r.clone(), r @ r, False, 0
        for k in range(1, iters + 1):
            Ap = Av(th, t, p) + damp * p
            pAp = p @ Ap
            if pAp <= 0:
                neg = True
                break
            a = rs / pAp
            x, r = x + a * p, r - a * Ap
            rs_new = r @ r
            if rs_new.sqrt() < tol * b.norm():
                rs = rs_new
                break
            p, rs = r + (rs_new / rs) * p, rs_new
        return x, {"cg_iters": k, "cg_relres": (rs.sqrt() / b.norm()).item(), "neg_curv": neg}

    def signal(self, th, v):
        _, c = jvp(lambda q: functional_call(self.net, q, (self.Xu,)), (th,), (unflat(v, th),))
        return c - c.mean(1, keepdim=True)


def cosine(a, b):
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-30)).item()


# ----------------------------------------------------------------------------------------
# runs
# ----------------------------------------------------------------------------------------
def train_on(init_state, Xu_, Xt_, Xl_, yl_, yt_, pl, steps, lr):
    net = mlp()
    net.load_state_dict(init_state)
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(net(Xu_), pl).backward()
        opt.step()
    with torch.no_grad():
        return {"stu_acc": (net(Xt_).argmax(1) == yt_).float().mean().item(),
                "agree": (net(Xu_).argmax(1) == pl).float().mean().item(),
                "Ll": F.cross_entropy(net(Xl_), yl_).item()}


def md_step(z, g, beta):
    g = g - g.mean(1, keepdim=True)
    return z - beta * g / g.pow(2).mean().sqrt().clamp_min(1e-30)


def md_unrolled(net, Xu_, Xl_, yl_, yu_, K, z0, eta, M, beta):
    """M mirror-descent steps on the pseudo-label logits against F_K."""
    z, trace, g0, F0 = z0.clone(), [], None, None
    uniform_start = bool((z0 == 0).all())
    for m in range(M):
        g, Fk = unrolled(net, Xu_, Xl_, yl_, K, F.softmax(z, 1), eta)
        if m == 0:
            g0, F0 = g - g.mean(1, keepdim=True), Fk
        acc = float("nan") if (m == 0 and uniform_start) else (F.softmax(z, 1).argmax(1) == yu_).float().mean().item()
        trace.append({"m": m, "pl_acc": acc, "F": Fk})
        z = md_step(z, g, beta)
    t = F.softmax(z, 1)
    _, F_end = unrolled(net, Xu_, Xl_, yl_, K, t, eta, need_grad=False)
    trace.append({"m": M, "pl_acc": (t.argmax(1) == yu_).float().mean().item(), "F": F_end})
    return t, g0, F0, F_end, trace


def md_ift(I, th_start, yu_, z0, damp, eta, M, beta, inner0, inner_warm, log=None):
    """M mirror-descent steps against F_inf(t) = L_l(theta*(t)) with IFT hypergradients; theta*
    warm-started from the previous outer step."""
    z, trace = z0.clone(), []
    uniform_start = bool((z0 == 0).all())
    th = I.train(th_start, F.softmax(z, 1), inner0, eta)
    for m in range(M + 1):
        t = F.softmax(z, 1)
        if m:
            th = I.train(th, t, inner_warm, eta)
        b, Fi = I.lab(th)
        gn = I.grad_norm(th, t)
        acc = float("nan") if (m == 0 and uniform_start) else (t.argmax(1) == yu_).float().mean().item()
        rec = {"m": m, "pl_acc": acc, "F": Fi, "inner_grad": gn}
        if m < M:
            v, info = I.cg(th, t, b, damp, "ggn")
            rec.update(info)
            z = md_step(z, I.signal(th, v), beta)
        trace.append(rec)
        if log is not None:
            log.append(f"  IFT m={m:2d}  F_inf={Fi:.3e}  PL acc={acc:.3f}  |grad L_u(theta*)|={gn:.1e}"
                       + (f"  CG {rec['cg_iters']} it, rel.res {rec['cg_relres']:.1e}" if m < M else ""))
    return F.softmax(z, 1), th, trace


def job(spec):
    torch.set_num_threads(1)
    n_l, d, init, eta, M, beta, steps = (spec[k] for k in ("n_l", "lab_draw", "init", "eta", "M", "beta", "steps"))
    Xu, yu, Xt, yt, Xl, yl = make_data(n_l, d)
    Xu_, Xt_, Xl_ = tensors(Xu, Xt, Xl)
    yu_, yt_, yl_ = torch.tensor(yu), torch.tensor(yt), torch.tensor(yl)
    torch.manual_seed(1000 * d + init)  # the same random student as toy_two_moons_meta.py
    net = mlp()
    init_state = {k: v.clone() for k, v in net.state_dict().items()}
    th0 = {n: p.detach().clone() for n, p in net.named_parameters()}
    truth = F.one_hot(yu_, 2).float()
    z_truth = 3.0 * (2 * truth - 1)  # logits +-3: t = 0.9975 on the true class
    snap = (n_l, d, init) == SNAP
    tr = lambda pl: train_on(init_state, Xu_, Xt_, Xl_, yl_, yt_, pl, steps, eta)  # noqa: E731
    out = {"n_l": n_l, "lab_draw": d, "init": init, "K": {}}

    # ---- unrolled, K = 1 ... 100 --------------------------------------------------------
    g1 = None
    for K in KS:
        t0 = time.time()
        _, F_true = unrolled(net, Xu_, Xl_, yl_, K, truth, eta, need_grad=False)
        t, g0, F_uniform, F_end, trace_u = md_unrolled(net, Xu_, Xl_, yl_, yu_, K, torch.zeros(len(yu), 2), eta, M, beta)
        g1 = g0.clone() if K == 1 else g1
        t_tr, _, _, F_end_tr, trace_t = md_unrolled(net, Xu_, Xl_, yl_, yu_, K, z_truth, eta, M, beta)
        pl_static, pl_opt, pl_tr = g0.argmin(1), t.argmax(1), t_tr.argmax(1)
        _, F_static = unrolled(net, Xu_, Xl_, yl_, K, F.one_hot(pl_static, 2).float(), eta, need_grad=False)
        rec = {"pl_acc_static": (pl_static == yu_).float().mean().item(),
               "pl_acc_opt": (pl_opt == yu_).float().mean().item(),
               "pl_acc_from_truth": (pl_tr == yu_).float().mean().item(),
               "cos_g1": cosine(g0, g1), "agree_g1": (pl_static == g1.argmin(1)).float().mean().item(),
               "g_rms": g0.pow(2).mean().sqrt().item(), "F_uniform": F_uniform, "F_static": F_static,
               "F_opt": F_end, "F_true": F_true, "F_from_truth": F_end_tr, "conf_opt": t.max(1).values.mean().item(),
               "trace_uniform": trace_u, "trace_truth": trace_t,
               "train_static": tr(pl_static), "train_opt": tr(pl_opt), "train_from_truth": tr(pl_tr),
               "seconds": time.time() - t0}
        if snap:
            rec["signal"] = (g0[:, 0] - g0[:, 1]).tolist()  # > 0: the signal prefers class 1
            rec["pl_opt"], rec["pl_from_truth"] = pl_opt.tolist(), pl_tr.tolist()
        out["K"][str(K)] = rec

    # ---- implicit (IFT) ------------------------------------------------------------------
    t0 = time.time()
    I = Inner(net, Xu_, Xl_, yl_)
    uni = torch.full((len(yu), 2), 0.5)
    th_u = I.train(th0, uni, 1000, eta)
    b, Ll_u = I.lab(th_u)
    lm = I.lam_max(th_u, uni)
    c_first = I.signal(th_u, b)  # first-order signal at theta*(uniform)
    ift = {"lam_max": lm, "inner_grad_uniform": I.grad_norm(th_u, uni), "F_uniform": Ll_u, "static": {}}
    sig_main = None
    for curv in ("ggn", "hess"):
        for rho in RHOS:
            v, info = I.cg(th_u, uni, b, rho * lm, curv)
            s = I.signal(th_u, v)
            key = f"{curv}_{rho:g}"
            ift["static"][key] = dict(info, pl_acc=(s.argmin(1) == yu_).float().mean().item(),
                                      cos_first=cosine(s, c_first), v_ratio=(v.norm() / b.norm()).item())
            if curv == "ggn":
                net_star = mlp()
                net_star.load_state_dict(th_u)
                ift["static"][key]["cos_unroll_from_star"] = {}
                for K in K_FROM_STAR:
                    gK, _ = unrolled(net_star, Xu_, Xl_, yl_, K, uni, eta)
                    ift["static"][key]["cos_unroll_from_star"][str(K)] = cosine(gK - gK.mean(1, keepdim=True), s)
            if curv == "ggn" and rho == RHO_MAIN:
                sig_main = s
    pl_static = sig_main.argmin(1)
    ift["pl_acc_static"] = (pl_static == yu_).float().mean().item()
    th_true = I.train(th0, truth, 1000, eta)
    ift["F_true"] = I.lab(th_true)[1]
    ift["inner_grad_truth"] = I.grad_norm(th_true, truth)
    log = [] if snap else None
    damp = RHO_MAIN * lm
    if log is not None:
        log.append(f"IFT, Gauss-Newton, damping {RHO_MAIN:g} x lambda_max = {damp:.2e}; from uniform t:")
    t_u, _, tr_u = md_ift(I, th0, yu_, torch.zeros(len(yu), 2), damp, eta, M, beta, 1000, 100, log)
    if log is not None:
        log.append("from the true labels:")
    t_t, _, tr_t = md_ift(I, th0, yu_, z_truth, damp, eta, M, beta, 1000, 100, log)
    pl_opt, pl_tr = t_u.argmax(1), t_t.argmax(1)
    ift.update({"pl_acc_opt": (pl_opt == yu_).float().mean().item(), "pl_acc_from_truth": (pl_tr == yu_).float().mean().item(),
                "F_opt": tr_u[-1]["F"], "F_from_truth": tr_t[-1]["F"], "trace_uniform": tr_u, "trace_truth": tr_t,
                "train_static": tr(pl_static), "train_opt": tr(pl_opt), "train_from_truth": tr(pl_tr),
                "seconds": time.time() - t0})
    if snap:
        ift["signal"] = (sig_main[:, 0] - sig_main[:, 1]).tolist()
        ift["pl_opt"], ift["pl_from_truth"], ift["log"] = pl_opt.tolist(), pl_tr.tolist(), log
    out["IFT"] = ift
    return out


def main_runs(a):
    specs = [{"n_l": n_l, "lab_draw": d, "init": i, "eta": a.eta, "M": a.M, "beta": a.beta, "steps": a.steps}
             for n_l in (1, 5) for d in range(a.lab_draws) for i in range(a.inits)]
    t0 = time.time()
    with Pool(a.procs) as pool:
        runs = pool.map(job, specs, chunksize=1)
    ls = {}
    for n_l in (1, 5):
        for d in range(a.lab_draws):
            Xu, yu, _, _, Xl, yl = make_data(n_l, d)
            ls[f"{n_l}_{d}"] = float((label_spreading(Xu, Xl, yl) == yu).mean())
    meta = {"Ks": KS, "rhos": RHOS, "rho_main": RHO_MAIN, "eta": a.eta, "M": a.M, "beta": a.beta,
            "train_steps": a.steps, "minutes": (time.time() - t0) / 60}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(_round({"meta": meta, "runs": runs, "label_spreading": ls}), open(OUT_JSON, "w"))
    print(f"{len(runs)} students in {meta['minutes']:.1f} min -> {OUT_JSON}")


# ----------------------------------------------------------------------------------------
# summary and figures
# ----------------------------------------------------------------------------------------
def entries(r):
    """(label, record) for K = 1 ... 100 and the IFT."""
    return [(str(K), r["K"][str(K)]) for K in KS] + [("IFT", r["IFT"])]


def summary(R):
    runs = R["runs"]
    med = lambda xs: float(np.median(xs))  # noqa: E731
    for n_l in (1, 5):
        rs = [r for r in runs if r["n_l"] == n_l]
        ls = np.mean([v for k, v in R["label_spreading"].items() if k.startswith(f"{n_l}_")])
        print(f"\n=== {n_l} labeled point(s) per class, {len(rs)} random students; label spreading {ls:.3f}; medians")
        print("    K | PL acc: static / opt from uniform / opt from truth | student acc on these | "
              "F: true labels / uniform start / opt from uniform / opt from truth")
        for j, (lab, _) in enumerate(entries(rs[0])):
            X = [entries(r)[j][1] for r in rs]
            print(f"  {lab:>3s} | {med([x['pl_acc_static'] for x in X]):.3f} / {med([x['pl_acc_opt'] for x in X]):.3f} / "
                  f"{med([x['pl_acc_from_truth'] for x in X]):.3f} | {med([x['train_static']['stu_acc'] for x in X]):.3f} / "
                  f"{med([x['train_opt']['stu_acc'] for x in X]):.3f} / {med([x['train_from_truth']['stu_acc'] for x in X]):.3f} | "
                  f"{med([x['F_true'] for x in X]):.2e} / {med([x['F_uniform'] for x in X]):.2e} / "
                  f"{med([x['F_opt'] for x in X]):.2e} / {med([x['F_from_truth'] for x in X]):.2e}")
        below = [np.mean([e["F_opt"] < e["F_true"] for e in [entries(r)[j][1] for r in rs]]) for j in range(len(KS) + 1)]
        print("  share of students whose optimised pseudo-labels beat the true labels on F:",
              ", ".join(f"{lab}: {b:.2f}" for (lab, _), b in zip(entries(rs[0]), below)))
        I = [r["IFT"] for r in rs]
        print(f"  IFT logs: lambda_max(H) median {med([x['lam_max'] for x in I]):.2f}; |grad L_u(theta*)| at uniform "
              f"{med([x['inner_grad_uniform'] for x in I]):.1e}, at the truth {med([x['inner_grad_truth'] for x in I]):.1e}")
        for curv in ("ggn", "hess"):
            for rho in RHOS:
                S = [x["static"][f"{curv}_{rho:g}"] for x in I]
                line = (f"    {curv:4s} damping {rho:g}: CG iters {med([s['cg_iters'] for s in S]):.0f}, rel.res "
                        f"{med([s['cg_relres'] for s in S]):.1e}, negative curvature in {np.mean([s['neg_curv'] for s in S]):.2f}, "
                        f"|v|/|b| {med([s['v_ratio'] for s in S]):.1e}, argmin PL acc {med([s['pl_acc'] for s in S]):.3f}, "
                        f"cos with first-order {med([s['cos_first'] for s in S]):.3f}")
                if curv == "ggn":
                    line += ", cos with unroll from theta* " + ", ".join(
                        f"K={K}: {med([s['cos_unroll_from_star'][str(K)] for s in S]):.2f}" for K in K_FROM_STAR)
                print(line)
        tr = [x["trace_uniform"] for x in I]
        print(f"    IFT optimisation (GGN, damping {RHO_MAIN:g}): CG iters median {med([e['cg_iters'] for t in tr for e in t if 'cg_iters' in e]):.0f}, "
              f"rel.res max {max(e['cg_relres'] for t in tr for e in t if 'cg_relres' in e):.1e}, inner |grad| median "
              f"{med([e['inner_grad'] for t in tr for e in t]):.1e}")
    snap = next(r for r in runs if (r["n_l"], r["lab_draw"], r["init"]) == SNAP)
    print(f"\nIFT log of one student (n_l, draw, init = {SNAP}):")
    for line in snap["IFT"]["log"][:8] + ["  ..."] + snap["IFT"]["log"][-4:]:
        print(line)


def plots(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({
        "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 2.0, "savefig.dpi": 200, "figure.facecolor": "white"})
    runs = R["runs"]
    diverging = LinearSegmentedColormap.from_list("div", [BLUE, "#d9d8d3", ORANGE])
    seq = LinearSegmentedColormap.from_list("seq", ["#bcd5f3", BLUE, "#0b2f5e"])
    COL = {"static": BLUE, "uniform": ORANGE, "truth": AQUA}
    XPOS = KS + [400]                       # the IFT sits right of K = 100
    XLAB = [str(k) for k in KS] + ["IFT"]
    paths = []

    def fmt_k_axis(ax):
        ax.set_xscale("log")
        ax.set_xticks(XPOS)
        ax.set_xticklabels(XLAB)
        ax.minorticks_off()
        ax.axvspan(150, 600, color=GRID, alpha=0.35, lw=0)
        ax.set_xlabel("unrolled student steps K (IFT = converged)")

    # ---- figure 1: the signal and the optimised pseudo-labels on one student ----------------
    snap = next(r for r in runs if (r["n_l"], r["lab_draw"], r["init"]) == SNAP)
    n_l, d, init = SNAP
    Xu, yu, _, _, Xl, yl = make_data(n_l, d)
    show = [("1", snap["K"]["1"]), ("5", snap["K"]["5"]), ("20", snap["K"]["20"]), ("100", snap["K"]["100"]),
            ("IFT", snap["IFT"])]
    fig, axes = plt.subplots(3, len(show), figsize=(8.6, 5.6))
    for j, (lab, rec) in enumerate(show):
        name = f"K = {lab}" if lab != "IFT" else "IFT (GGN)"
        s = np.array(rec["signal"])
        sc = np.percentile(np.abs(s), 95)
        ax = axes[0, j]
        ax.scatter(Xu[:, 0], Xu[:, 1], s=2.5, c=np.clip(s / sc, -1, 1), cmap=diverging, vmin=-1, vmax=1, lw=0)
        ax.set_title(f"{name}: signal at uniform t\nargmin PL acc {rec['pl_acc_static']:.3f}", fontsize=7, loc="left", color=INK)
        for row, key, start in ((1, "pl_opt", "from uniform"), (2, "pl_from_truth", "from the truth")):
            ax = axes[row, j]
            pl = np.array(rec[key])
            wrong = pl != yu
            cols = np.where(pl == 0, BLUE, ORANGE)
            ax.scatter(Xu[~wrong, 0], Xu[~wrong, 1], s=2.5, c=cols[~wrong], lw=0)
            ax.scatter(Xu[wrong, 0], Xu[wrong, 1], s=6, c=cols[wrong], marker="x", lw=0.5)
            Fv = rec["F_opt"] if row == 1 else rec["F_from_truth"]
            acc = rec["pl_acc_opt"] if row == 1 else rec["pl_acc_from_truth"]
            ax.set_title(f"optimised {start}\nPL acc {acc:.3f}, F {Fv:.1e}", fontsize=7, loc="left", color=INK)
        for ax in axes[:, j]:
            for k_, c_ in ((0, BLUE), (1, ORANGE)):
                m = np.array(yl) == k_
                ax.scatter(np.array(Xl)[m, 0], np.array(Xl)[m, 1], s=45, c=c_, edgecolors=INK, linewidths=0.9, zorder=5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_aspect("equal")
    fig.text(0.01, 0.0, "One random student, 1 labeled point per class (large dots).  Row 1: dF/dt at uniform pseudo-labels "
             "(orange = prefers class 1, blue = class 0, grey = indifferent; scale per panel).  Rows 2-3: pseudo-labels after "
             f"30 mirror-descent steps on F, from uniform and from the true labels (x = wrong).  F at the true labels: "
             + ", ".join(f"{lab} {rec['F_true']:.1e}" for lab, rec in show) + ".", fontsize=6, color=INK2, wrap=True)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    paths.append(os.path.join(FIG, "toy_moons_unroll_signal.png"))
    fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)

    # ---- figure 2: accuracy and meta objective against K -------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.2))

    def band(ax, rs, f, color, ls_="-", marker="o"):
        M_ = np.array([[f(e) for _, e in entries(r)] for r in rs])
        ax.fill_between(XPOS[:-1], np.percentile(M_[:, :-1], 25, 0), np.percentile(M_[:, :-1], 75, 0), color=color,
                        alpha=0.12, lw=0)
        ax.plot(XPOS[:-1], np.median(M_[:, :-1], 0), color=color, ls=ls_, lw=1.5, marker=marker, ms=3)
        q = np.percentile(M_[:, -1], [25, 50, 75])
        ax.errorbar([XPOS[-1]], [q[1]], yerr=[[q[1] - q[0]], [q[2] - q[1]]], color=color, marker=marker, ms=4,
                    lw=1.2, capsize=2)

    for row, n_l in enumerate((1, 5)):
        rs = [r for r in runs if r["n_l"] == n_l]
        tag = f"{n_l} labeled point{'s' if n_l > 1 else ''} per class"
        ls = np.mean([v for k, v in R["label_spreading"].items() if k.startswith(f"{n_l}_")])
        ax = axes[row, 0]
        band(ax, rs, lambda e: e["pl_acc_static"], COL["static"])
        band(ax, rs, lambda e: e["pl_acc_opt"], COL["uniform"])
        band(ax, rs, lambda e: e["pl_acc_from_truth"], COL["truth"])
        ax.axhline(ls, color=INK2, lw=0.9, ls=(0, (4, 2)))
        ax.annotate(f"label spreading {ls:.3f}", (1, ls), xytext=(2, -9), textcoords="offset points", fontsize=6.3, color=INK2)
        ax.set_ylim(0.4, 1.02)
        ax.set_title(f"{'ad'[row]}. pseudo-label accuracy, {tag}", fontsize=7.5, loc="left", color=INK)
        ax = axes[row, 1]
        band(ax, rs, lambda e: e["train_static"]["stu_acc"], COL["static"])
        band(ax, rs, lambda e: e["train_opt"]["stu_acc"], COL["uniform"])
        band(ax, rs, lambda e: e["train_from_truth"]["stu_acc"], COL["truth"])
        ax.set_ylim(0.4, 1.02)
        ax.set_title(f"{'be'[row]}. student trained on them (test)", fontsize=7.5, loc="left", color=INK)
        ax = axes[row, 2]
        band(ax, rs, lambda e: e["F_true"], INK2, (0, (4, 2)), "s")
        band(ax, rs, lambda e: e["F_opt"], COL["uniform"])
        band(ax, rs, lambda e: e["F_from_truth"], COL["truth"])
        ax.set_yscale("log")
        ax.set_title(f"{'cf'[row]}. meta objective F = L_l(student)", fontsize=7.5, loc="left", color=INK)
        for ax in axes[row]:
            fmt_k_axis(ax)
    axes[0, 0].legend(handles=[Line2D([], [], color=COL["static"], marker="o", ms=3, lw=1.5, label="argmin of the signal at uniform t"),
                               Line2D([], [], color=COL["uniform"], marker="o", ms=3, lw=1.5, label="optimised from uniform t"),
                               Line2D([], [], color=COL["truth"], marker="o", ms=3, lw=1.5, label="optimised from the true labels")],
                      fontsize=6, loc="lower left")
    axes[0, 2].legend(handles=[Line2D([], [], color=INK2, ls=(0, (4, 2)), marker="s", ms=3, lw=1.5, label="true labels"),
                               Line2D([], [], color=COL["uniform"], marker="o", ms=3, lw=1.5, label="optimised from uniform"),
                               Line2D([], [], color=COL["truth"], marker="o", ms=3, lw=1.5, label="optimised from the truth")],
                      fontsize=6, loc="lower left")
    fig.text(0.01, 0.0, "Median and interquartile range over 40 random students.  IFT: student trained to convergence, "
             f"Gauss-Newton curvature, damping {RHO_MAIN:g} x lambda_max(H).", fontsize=6.3, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    paths.append(os.path.join(FIG, "toy_moons_unroll_summary.png"))
    fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)

    # ---- figure 3: optimisation paths in the (F, pseudo-label accuracy) plane -----------------
    fig, axes = plt.subplots(2, 4, figsize=(8.6, 4.4), sharey=True)
    for row, n_l in enumerate((1, 5)):
        rs = [r for r in runs if r["n_l"] == n_l]
        for j, lab in enumerate(["5", "20", "100", "IFT"]):
            ax = axes[row, j]
            get = (lambda r: r["IFT"]) if lab == "IFT" else (lambda r, lab=lab: r["K"][lab])
            xs_all = []
            for key, col in (("trace_uniform", COL["uniform"]), ("trace_truth", COL["truth"])):
                T = [get(r)[key] for r in rs]
                L = min(len(t) for t in T)
                Fm = np.median([[e["F"] for e in t[:L]] for t in T], 0)
                Am = np.nanmedian(np.array([[e["pl_acc"] for e in t[:L]] for t in T], float), 0)
                ok = np.isfinite(Am)
                ax.plot(Fm[ok], Am[ok], color=col, lw=1.4)
                ax.plot(Fm[ok][-1], Am[ok][-1], "o", color=col, ms=4)
                xs_all += list(Fm[ok])
            Ft = np.median([get(r)["F_true"] for r in rs])
            ax.plot([Ft], [1.0], marker="*", color=INK, ms=8, ls="")
            xs_all.append(Ft)
            ax.set_xscale("log")
            lo_, hi_ = np.log10(min(xs_all)), np.log10(max(xs_all))
            ticks = sorted({float(f"{v:.1g}") for v in 10 ** np.linspace(lo_ + 0.05 * (hi_ - lo_), hi_ - 0.05 * (hi_ - lo_), 3)})
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{v:.0e}".replace("e-0", "e-") if v < 0.01 else f"{v:g}" for v in ticks])
            ax.set_ylim(0.45, 1.03)
            name = f"K = {lab}" if lab != "IFT" else "IFT"
            ax.set_title(f"{'abcdefgh'[4 * row + j]}. {name}, {n_l} lab./class", fontsize=7.5, loc="left", color=INK)
            ax.set_xlabel("meta objective F")
        axes[row, 0].set_ylabel("pseudo-label accuracy")
    axes[0, 0].legend(handles=[Line2D([], [], color=COL["uniform"], lw=1.4, marker="o", ms=4, label="from uniform t"),
                               Line2D([], [], color=COL["truth"], lw=1.4, marker="o", ms=4, label="from the true labels"),
                               Line2D([], [], color=INK, marker="*", ms=8, ls="", label="true labels")],
                      fontsize=6, loc="lower left")
    fig.text(0.01, 0.0, "Median over 40 random students of each mirror-descent step (30 steps, dot = end).  Moving left "
             "= the bilevel objective is better solved; down = pseudo-labels get worse.", fontsize=6.3, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    paths.append(os.path.join(FIG, "toy_moons_unroll_paths.png"))
    fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)

    # ---- figure 4: pseudo-label accuracy vs student accuracy, coloured by K ------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6), sharey=True)
    norm = matplotlib.colors.LogNorm(vmin=min(KS), vmax=max(KS))
    for ax, n_l in zip(axes, (1, 5)):
        lo = 0.38
        ax.plot([lo, 1], [lo, 1], color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
        rs = [r for r in runs if r["n_l"] == n_l]
        for K in KS:
            ax.scatter([r["K"][str(K)]["train_opt"]["stu_acc"] for r in rs], [r["K"][str(K)]["pl_acc_opt"] for r in rs],
                       s=9, color=seq(norm(K)), edgecolors="white", linewidths=0.3, zorder=2)
        ax.scatter([r["IFT"]["train_opt"]["stu_acc"] for r in rs], [r["IFT"]["pl_acc_opt"] for r in rs], s=14,
                   facecolors="white", edgecolors=ORANGE, linewidths=0.9, zorder=2)
        mx = [np.median([r["K"][str(K)]["train_opt"]["stu_acc"] for r in rs]) for K in KS]
        my = [np.median([r["K"][str(K)]["pl_acc_opt"] for r in rs]) for K in KS]
        ax.plot(mx, my, color=INK, lw=1.0, zorder=3)
        ax.scatter(mx, my, s=18, color=[seq(norm(K)) for K in KS], edgecolors=INK, linewidths=0.6, zorder=4)
        for K, x_, y_ in zip(KS, mx, my):
            if K in (1, 10, 100):
                ax.annotate(f"K={K}", (x_, y_), xytext=(4, 2), textcoords="offset points", fontsize=6.3, color=INK)
        ix = np.median([r["IFT"]["train_opt"]["stu_acc"] for r in rs])
        iy = np.median([r["IFT"]["pl_acc_opt"] for r in rs])
        ax.scatter([ix], [iy], s=30, facecolors="white", edgecolors=ORANGE, linewidths=1.4, zorder=5)
        ax.annotate("IFT", (ix, iy), xytext=(4, -8), textcoords="offset points", fontsize=6.3, color=INK)
        ax.set_xlim(lo, 1.0)
        ax.set_ylim(lo, 1.0)
        ax.set_xlabel("student test accuracy")
        ax.set_title(f"{n_l} labeled point{'s' if n_l > 1 else ''} per class, 40 random students", fontsize=8, loc="left", color=INK)
    axes[0].set_ylabel("pseudo-label accuracy (optimised from uniform t)")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=seq)
    cb = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02, ticks=KS)
    cb.ax.set_yticklabels([str(k) for k in KS])
    cb.set_label("unrolled steps K", color=INK)
    fig.text(0.01, -0.02, "Filled dots: unrolled K (dark line = median, K = 1 ... 100); open orange: IFT (converged student); "
             "dashed: student = pseudo-labels.", fontsize=6.3, color=INK2)
    paths.append(os.path.join(FIG, "toy_moons_unroll_pl_vs_student.png"))
    fig.savefig(paths[-1], bbox_inches="tight")
    plt.close(fig)
    print(*paths)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--lab_draws", type=int, default=5)
    ap.add_argument("--inits", type=int, default=8)
    ap.add_argument("--eta", type=float, default=0.1, help="student step size, in the unroll and in training")
    ap.add_argument("--M", type=int, default=30, help="mirror-descent steps on the pseudo-labels")
    ap.add_argument("--beta", type=float, default=0.3, help="mirror-descent step on the logits (unit-RMS signal)")
    ap.add_argument("--steps", type=int, default=1000, help="training steps of the student on the pseudo-labels")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    if not a.plot:
        main_runs(a)
    R = json.load(open(OUT_JSON))
    summary(R)
    plots(R)
