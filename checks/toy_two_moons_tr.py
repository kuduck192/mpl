#!/usr/bin/env python3
"""Two Moons toy: TR-PBPL and TR-EBPL meta objectives under the MPL protocols of report Sections 13-15.

The three MPL toys judge pseudo-labels by the labeled loss of one student (MPL / optimistic feedback):
  sec13  toy_two_moons_meta.py    PL = argmin of the first-order signal c = J_f(X_u) grad L_l(theta), dynamic
                                  (re-set before every student step) and frozen (set at the random student);
  sec14  toy_two_moons_unroll.py  free soft PL against F_K(t) = L_l(theta_K(t)), K = 1 ... 100 unrolled steps,
                                  and F_inf(t) = L_l(theta*(t)) by IFT; 30 mirror-descent steps from uniform t
                                  and from the truth, the argmin of the signal at uniform t;
  sec15  toy_two_moons_anchor.py  PL anchored to a FixMatch teacher q: t* = argmin <t, unit(g_K)> + lam KL(t||q).
This script runs the same protocols, data, random students, teachers, step sizes and budgets, with the labeled
loss of the student replaced by a trust-region value at the same student (checks/tr_fiber.py):
  pess (TR-PBPL)  V(theta) = max_{||eps|| <= r} L_l(theta + eps) - beta D_theta(theta + eps)
  exp  (TR-EBPL)  E(theta) = mean_j L_l(y_j(theta)), y_j = proximal retraction of theta + r xi_j onto the fibre
  D_theta(y) = mean_{x in X_u} (delta_y(x) - delta_theta(x))^2 (change of the student's logits on X_u).
The only change inside each protocol is the vector v that stands for grad L_l(theta): c = J_f(X_u) v (sec13),
v back-propagated through the K unrolled steps (sec14, sec15), and v as the right-hand side of the IFT system.
v = grad V or grad E is exact (Danskin / autograd through the retraction; checked by finite differences in
tr_fiber.py).  mode = opt calls the original functions of the three MPL toys unchanged.

  python checks/toy_two_moons_tr.py --protocol sec14 --mode pess --radius 1 --beta 1e4 --shard 0/8 --procs 4
  python checks/toy_two_moons_tr.py --summary            # merge results/toy-two-moons-tr/*.json, tables, figures
"""
import argparse
import glob
import json
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_two_moons_anchor as A15  # noqa: E402
import toy_two_moons_meta as A13  # noqa: E402
import toy_two_moons_unroll as A14  # noqa: E402
from toy_two_moons_meta import FIG, LOG_ITERS, ROOT, _round, make_data, mlp, tensors  # noqa: E402
from toy_two_moons_unroll import Inner, flat, md_step, train_on, unflat  # noqa: E402
from tr_fiber import FlatMLP, expv, lab_loss_grad, pess, unit_directions  # noqa: E402

OUT_DIR = os.path.join(ROOT, "results", "toy-two-moons-tr")
TR_KEYS = ("V", "Ll_anchor", "Ll_adv", "rms_dlogit", "max_dlogit", "eps_norm", "cos_v_opt", "cos_v_fixed", "kkt",
           "iters", "n_steps", "start", "loss_sd", "fp_res", "q_last_drop", "v_norm", "gopt_norm")


# ----------------------------------------------------------------------------------------
# the meta objective at a student anchor
# ----------------------------------------------------------------------------------------
class Meta:
    """Value and gradient (flat, float64, CPU) of the meta objective at a student given as a flat vector.
    pess keeps its adversary eps between calls (persistent, as in ssl_fmnist.py) until reset()."""

    def __init__(self, cfg, Xu_, Xl_, yl_, xi_seed):
        self.cfg = cfg
        self.mode = cfg["mode"]
        self.f = FlatMLP()
        self.dev = cfg.get("tr_device", "cpu")
        self.Xu = Xu_.double().to(self.dev)
        self.Xl = Xl_.double().to(self.dev)
        self.yl = yl_.to(self.dev)
        self.N = Xu_.shape[0]
        self.eps, self.LA, self.age = None, None, 0
        self.xis = (unit_directions(self.f.d, cfg["M_exp"], xi_seed, device=self.dev)
                    if self.mode == "exp" else None)
        self.n_eval, self.seconds = 0, 0.0

    def reset(self):
        self.eps = None

    def __call__(self, theta, cold=False, need_grad=True, T=None, refresh=1):
        """cold: multi-start (grad + random starts, T_cold steps); otherwise warm start (+ grad start unless
        warm_only), T steps.  refresh: rebuild the preconditioner every `refresh` calls."""
        c = self.cfg
        th = theta.detach().to(self.dev, torch.float64)
        t0 = time.time()
        if self.mode == "opt":
            L, g = lab_loss_grad(self.f, th, self.Xl, self.yl)
            v, info = g, {"V": L, "Ll_anchor": L}
        elif self.mode == "pess":
            if cold or self.eps is None:
                v, e, info = pess(self.f, self.Xu, self.Xl, self.yl, th, c["radius"], c["fbeta"], warm=self.eps,
                                  starts=("warm", "zero"), n_random=c["n_random"], seed=self.n_eval,
                                  T=c["T_cold"])
            else:
                starts = ("warm",) if c.get("warm_only") else ("warm", "zero")
                v, e, info = pess(self.f, self.Xu, self.Xl, self.yl, th, c["radius"], c["fbeta"], warm=self.eps,
                                  starts=starts, T=T or c["T_warm"])
            self.eps = e
        else:
            v, e, info = expv(self.f, self.Xu, self.Xl, self.yl, th, c["radius"], c["fbeta"], self.xis,
                              nc=c["nc"], need_grad=need_grad)
        self.n_eval += 1
        self.seconds += time.time() - t0
        info = {k: info[k] for k in TR_KEYS if k in info}
        return (v.detach().cpu() if v is not None else None), info["V"], info


def as_dict(v, like):
    """Flat float64 vector -> parameter dict shaped and typed like `like`."""
    return unflat(v.to(next(iter(like.values())).dtype), like)


# ----------------------------------------------------------------------------------------
# sec13: argmin of the first-order signal c = J_f(X_u) v
# ----------------------------------------------------------------------------------------
def signal13(net, Xu_, meta, cold, T=None, refresh=1):
    params = {n: p.detach() for n, p in net.named_parameters()}
    v, val, info = meta(flat(params), cold=cold, T=T, refresh=refresh)
    _, c = jvp(lambda q: functional_call(net, q, (Xu_,)), (params,), (as_dict(v, params),))
    return c, info


def run13(job):
    """toy_two_moons_meta.run with the meta signal of `mode` (opt: the original function)."""
    torch.set_num_threads(1)
    if job["mode"] == "opt":
        return dict(A13.run(job), mode="opt")
    n_l, lab_draw, init, rule, iters, lr = (job[k] for k in ("n_l", "lab_draw", "init", "rule", "iters", "lr"))
    Xu, yu, Xt, yt, Xl, yl = make_data(n_l, lab_draw)
    Xu_, Xt_, Xl_ = tensors(Xu, Xt, Xl)
    yu_, yt_, yl_ = torch.tensor(yu), torch.tensor(yt), torch.tensor(yl)
    torch.manual_seed(1000 * lab_draw + init)
    net = mlp()
    opt = torch.optim.SGD(net.parameters(), lr=lr)
    meta = Meta(job, Xu_, Xl_, yl_, xi_seed=555_000 + 10_000 * n_l + 1000 * lab_draw + init)
    pl, hist, t0 = None, [], time.time()
    for k in range(iters + 1):
        if rule == "dynamic" or pl is None:
            cold = k == 0 or (k % job["recold"] == 0)
            c, info = signal13(net, Xu_, meta, cold=cold, refresh=job["refresh13"])
            with torch.no_grad():
                Ll = F.cross_entropy(net(Xl_), yl_).item()
            new = c.argmin(1)
            flip = float((new != pl).float().mean()) if pl is not None and rule == "dynamic" else float("nan")
            pl = new
            c_rms = c.pow(2).mean().sqrt().item()
        else:
            flip = float("nan")
            with torch.no_grad():
                Ll = F.cross_entropy(net(Xl_), yl_).item()
        if k in LOG_ITERS:
            with torch.no_grad():
                pu, pt = net(Xu_).argmax(1), net(Xt_).argmax(1)
            h = {"it": k, "pl_acc": (pl == yu_).float().mean().item(), "stu_acc": (pt == yt_).float().mean().item(),
                 "stu_acc_u": (pu == yu_).float().mean().item(), "agree": (pu == pl).float().mean().item(),
                 "Ll": Ll, "flip": flip, "frac0": (pl == 0).float().mean().item(), "c_rms": c_rms}
            if rule == "dynamic" or k == 0:
                h.update({f"tr_{a}": b for a, b in info.items()})
            hist.append(h)
        if k == iters:
            break
        opt.zero_grad()
        F.cross_entropy(net(Xu_), pl).backward()
        opt.step()
    out = dict(job, hist=hist, seconds=time.time() - t0, tr_seconds=meta.seconds, n_eval=meta.n_eval)
    if job.get("keep_pl"):
        out["pl"] = pl.tolist()
        with torch.no_grad():
            g = np.stack(np.meshgrid(np.linspace(-1.6, 2.6, 120), np.linspace(-1.1, 1.6, 90)), -1).reshape(-1, 2)
            out["grid_p1"] = F.softmax(net(tensors(Xu, g, Xl)[1]), 1)[:, 1].reshape(90, 120).tolist()
    return out


def statics13(spec):
    """Per labeled draw: the argmin of the signal of each of 64 random students and of their average."""
    torch.set_num_threads(1)
    n_l, lab_draw, n_students = spec["n_l"], spec["lab_draw"], spec["n_students"]
    Xu, yu, Xt, yt, Xl, yl = make_data(n_l, lab_draw)
    Xu_, Xt_, Xl_ = tensors(Xu, Xt, Xl)
    yl_ = torch.tensor(yl)
    accs, C, infos = [], 0, []
    for s in range(n_students):
        torch.manual_seed(50_000 + 1000 * lab_draw + s)
        net = mlp()
        if spec["mode"] == "opt":
            c, _ = A13.meta_signal(net, Xu_, Xl_, yl_)
            info = {}
        else:
            meta = Meta(spec, Xu_, Xl_, yl_, xi_seed=777_000 + 10_000 * n_l + 1000 * lab_draw + s)
            c, info = signal13(net, Xu_, meta, cold=True)
        c = c - c.mean(1, keepdim=True)
        C = C + c / c.pow(2).mean().sqrt()
        accs.append(float((c.argmin(1).numpy() == yu).mean()))
        infos.append(info)
    return dict(spec, single_accs=accs, avg_acc=float((C.argmin(1).numpy() == yu).mean()), infos=infos)


# ----------------------------------------------------------------------------------------
# sec14 / sec15: unrolled and implicit hypergradients with the trust-region value
# ----------------------------------------------------------------------------------------
def unrolled_tr(net, Xu_, Xl_, yl_, K, t, eta, meta, need_grad=True, cold=False, T=None):
    """F(t) = Phi(theta_K(t)) and dF/dt: K unrolled student steps (as toy_two_moons_unroll.unrolled), then
    v = grad Phi(theta_K) back-propagated through the unroll."""
    th = {n: p.detach().clone().requires_grad_(True) for n, p in net.named_parameters()}
    t = t.detach().clone().requires_grad_(need_grad)
    for _ in range(K):
        Lu = -(t * F.log_softmax(functional_call(net, th, (Xu_,)), 1)).sum(1).mean()
        g = torch.autograd.grad(Lu, list(th.values()), create_graph=need_grad)
        th = {n: w - eta * gi for (n, w), gi in zip(th.items(), g)}
    theta = flat(th)
    v, val, info = meta(theta.detach(), cold=cold, need_grad=need_grad, T=T)
    with torch.no_grad():
        info["Ll"] = F.cross_entropy(functional_call(net, {n: w.detach() for n, w in th.items()}, (Xl_,)), yl_).item()
    if not need_grad:
        return None, val, info
    return torch.autograd.grad(theta, t, grad_outputs=v.to(theta.dtype))[0], val, info


def trace_info(info):
    return {k: info[k] for k in ("Ll", "Ll_adv", "rms_dlogit", "eps_norm", "cos_v_opt", "kkt", "loss_sd") if k in info}


def md_unrolled_tr(net, Xu_, Xl_, yl_, yu_, K, z0, eta, M, beta, meta):
    """toy_two_moons_unroll.md_unrolled with the meta objective of `meta`."""
    meta.reset()
    z, trace, g0, F0 = z0.clone(), [], None, None
    uniform_start = bool((z0 == 0).all())
    for m in range(M):
        g, Fk, info = unrolled_tr(net, Xu_, Xl_, yl_, K, F.softmax(z, 1), eta, meta, cold=(m == 0))
        if m == 0:
            g0, F0 = g - g.mean(1, keepdim=True), Fk
        acc = float("nan") if (m == 0 and uniform_start) else (F.softmax(z, 1).argmax(1) == yu_).float().mean().item()
        trace.append(dict({"m": m, "pl_acc": acc, "F": Fk}, **trace_info(info)))
        z = md_step(z, g, beta)
    t = F.softmax(z, 1)
    _, F_end, info = unrolled_tr(net, Xu_, Xl_, yl_, K, t, eta, meta, need_grad=False)
    trace.append(dict({"m": M, "pl_acc": (t.argmax(1) == yu_).float().mean().item(), "F": F_end}, **trace_info(info)))
    return t, g0, F0, F_end, trace


def meta_at(meta, th, Xl_, yl_, net, cold=False, need_grad=True):
    """meta at a student given as a parameter dict; returns (flat float32 vector, value, info + L_l)."""
    v, val, info = meta(flat(th), cold=cold, need_grad=need_grad)
    with torch.no_grad():
        info["Ll"] = F.cross_entropy(functional_call(net, th, (Xl_,)), yl_).item()
    return (v.float() if v is not None else None), val, info


def md_ift_tr(I, net, th_start, yu_, z0, damp, eta, M, beta, inner0, inner_warm, meta, Xl_, yl_):
    """toy_two_moons_unroll.md_ift with the IFT right-hand side grad Phi(theta*)."""
    meta.reset()
    z, trace = z0.clone(), []
    uniform_start = bool((z0 == 0).all())
    th = I.train(th_start, F.softmax(z, 1), inner0, eta)
    for m in range(M + 1):
        t = F.softmax(z, 1)
        if m:
            th = I.train(th, t, inner_warm, eta)
        b, Fi, info = meta_at(meta, th, Xl_, yl_, net, cold=(m == 0), need_grad=m < M)
        gn = I.grad_norm(th, t)
        acc = float("nan") if (m == 0 and uniform_start) else (t.argmax(1) == yu_).float().mean().item()
        rec = dict({"m": m, "pl_acc": acc, "F": Fi, "inner_grad": gn}, **trace_info(info))
        if m < M:
            v, cg = I.cg(th, t, b, damp, "ggn")
            rec.update(cg)
            z = md_step(z, I.signal(th, v), beta)
        trace.append(rec)
    return F.softmax(z, 1), th, trace


def job14(spec):
    """toy_two_moons_unroll.job with the meta objective of `mode` (opt: the original function)."""
    torch.set_num_threads(1)
    if spec["mode"] == "opt":
        return dict(A14.job(spec), mode="opt")
    n_l, d, init, eta, M, beta, steps = (spec[k] for k in ("n_l", "lab_draw", "init", "eta", "M", "beta_md", "steps"))
    Xu, yu, Xt, yt, Xl, yl = make_data(n_l, d)
    Xu_, Xt_, Xl_ = tensors(Xu, Xt, Xl)
    yu_, yt_, yl_ = torch.tensor(yu), torch.tensor(yt), torch.tensor(yl)
    torch.manual_seed(1000 * d + init)
    net = mlp()
    init_state = {k: v.clone() for k, v in net.state_dict().items()}
    th0 = {n: p.detach().clone() for n, p in net.named_parameters()}
    truth = F.one_hot(yu_, 2).float()
    z_truth = 3.0 * (2 * truth - 1)
    tr = lambda pl: train_on(init_state, Xu_, Xt_, Xl_, yl_, yt_, pl, steps, eta)  # noqa: E731
    meta = Meta(spec, Xu_, Xl_, yl_, xi_seed=555_000 + 10_000 * n_l + 1000 * d + init)
    out = {"n_l": n_l, "lab_draw": d, "init": init, "mode": spec["mode"], "K": {}}
    t_job = time.time()
    for K in spec["Ks"]:
        t0 = time.time()
        meta.reset()
        _, F_true, i_true = unrolled_tr(net, Xu_, Xl_, yl_, K, truth, eta, meta, need_grad=False, cold=True)
        t, g0, F_uniform, F_end, trace_u = md_unrolled_tr(net, Xu_, Xl_, yl_, yu_, K, torch.zeros(len(yu), 2), eta, M, beta, meta)
        t_tr, _, _, F_end_tr, trace_t = md_unrolled_tr(net, Xu_, Xl_, yl_, yu_, K, z_truth, eta, M, beta, meta)
        pl_static, pl_opt, pl_tr = g0.argmin(1), t.argmax(1), t_tr.argmax(1)
        meta.reset()
        _, F_static, _ = unrolled_tr(net, Xu_, Xl_, yl_, K, F.one_hot(pl_static, 2).float(), eta, meta, need_grad=False, cold=True)
        out["K"][str(K)] = {
            "pl_acc_static": (pl_static == yu_).float().mean().item(), "pl_acc_opt": (pl_opt == yu_).float().mean().item(),
            "pl_acc_from_truth": (pl_tr == yu_).float().mean().item(), "g_rms": g0.pow(2).mean().sqrt().item(),
            "F_uniform": F_uniform, "F_static": F_static, "F_opt": F_end, "F_true": F_true, "F_from_truth": F_end_tr,
            "Ll_true": i_true["Ll"], "Ll_opt": trace_u[-1].get("Ll"), "Ll_from_truth": trace_t[-1].get("Ll"),
            "tr_true": i_true, "conf_opt": t.max(1).values.mean().item(), "trace_uniform": trace_u, "trace_truth": trace_t,
            "train_static": tr(pl_static), "train_opt": tr(pl_opt), "train_from_truth": tr(pl_tr), "seconds": time.time() - t0}
        if spec.get("keep_pl"):
            out["K"][str(K)].update({"pl_opt": pl_opt.tolist(), "pl_from_truth": pl_tr.tolist()})
    if spec["ift"]:
        t0 = time.time()
        I = Inner(net, Xu_, Xl_, yl_)
        uni = torch.full((len(yu), 2), 0.5)
        th_u = I.train(th0, uni, 1000, eta)
        meta.reset()
        b, F_u, i_u = meta_at(meta, th_u, Xl_, yl_, net, cold=True)
        lm = I.lam_max(th_u, uni)
        damp = A14.RHO_MAIN * lm
        v, cg = I.cg(th_u, uni, b, damp, "ggn")
        pl_static = I.signal(th_u, v).argmin(1)
        th_true = I.train(th0, truth, 1000, eta)
        meta.reset()
        _, F_true, i_true = meta_at(meta, th_true, Xl_, yl_, net, cold=True, need_grad=False)
        t_u, _, tr_u = md_ift_tr(I, net, th0, yu_, torch.zeros(len(yu), 2), damp, eta, M, beta, 1000, 100, meta, Xl_, yl_)
        t_t, _, tr_t = md_ift_tr(I, net, th0, yu_, z_truth, damp, eta, M, beta, 1000, 100, meta, Xl_, yl_)
        pl_opt, pl_tr = t_u.argmax(1), t_t.argmax(1)
        out["IFT"] = {"lam_max": lm, "F_uniform": F_u, "Ll_uniform": i_u["Ll"], "tr_uniform": i_u, "static_cg": cg,
                      "pl_acc_static": (pl_static == yu_).float().mean().item(), "F_true": F_true, "Ll_true": i_true["Ll"],
                      "tr_true": i_true, "pl_acc_opt": (pl_opt == yu_).float().mean().item(),
                      "pl_acc_from_truth": (pl_tr == yu_).float().mean().item(), "F_opt": tr_u[-1]["F"],
                      "F_from_truth": tr_t[-1]["F"], "Ll_opt": tr_u[-1].get("Ll"), "Ll_from_truth": tr_t[-1].get("Ll"),
                      "trace_uniform": tr_u, "trace_truth": tr_t, "train_static": tr(pl_static), "train_opt": tr(pl_opt),
                      "train_from_truth": tr(pl_tr), "seconds": time.time() - t0}
    out.update({"seconds": time.time() - t_job, "tr_seconds": meta.seconds, "n_eval": meta.n_eval})
    return out


def anchored_unrolled_tr(net, Xu_, Xl_, yl_, K, logq, lam, eta, M, beta, meta):
    meta.reset()
    z, resid = logq.clone(), float("nan")
    for m in range(M):
        g, _, _ = unrolled_tr(net, Xu_, Xl_, yl_, K, F.softmax(z, 1), eta, meta, cold=(m == 0))
        z_new = A15.prox_step(z, logq, A15.unit(g), lam, beta)
        resid, z = A15.fp_residual(z_new, z), z_new
    t = F.softmax(z, 1)
    _, Fe, info = unrolled_tr(net, Xu_, Xl_, yl_, K, t, eta, meta, need_grad=False)
    return t, Fe, dict({"fp_residual": resid}, **trace_info(info))


def anchored_ift_tr(I, net, th0, logq, lam, eta, M, beta, rho, meta, Xl_, yl_):
    meta.reset()
    z = logq.clone()
    t = F.softmax(z, 1)
    th = I.train(th0, t, 1000, eta)
    lm = I.lam_max(th, t)
    its, res, stat = [], [], []
    for m in range(M):
        if m:
            th = I.train(th, t, 100, eta)
        b, _, _ = meta_at(meta, th, Xl_, yl_, net, cold=(m == 0))
        v, info = I.cg(th, t, b, rho * lm, "ggn")
        its.append(info["cg_iters"])
        res.append(info["cg_relres"])
        stat.append(I.grad_norm(th, t))
        z_new = A15.prox_step(z, logq, A15.unit(I.signal(th, v)), lam, beta)
        resid, z = A15.fp_residual(z_new, z), z_new
        t = F.softmax(z, 1)
    th = I.train(th, t, 300, eta)
    _, Fe, info = meta_at(meta, th, Xl_, yl_, net, need_grad=False)
    return t, Fe, dict({"lam_max": lm, "cg_iters_median": float(np.median(its)), "cg_relres_max": float(max(res)),
                        "stationarity_median": float(np.median(stat)), "fp_residual": resid}, **trace_info(info))


def job15(spec):
    """toy_two_moons_anchor.job with the meta objective of `mode` (opt: the original function)."""
    torch.set_num_threads(1)
    if spec["mode"] == "opt":
        return dict(A15.job(spec), mode="opt")
    n_l, d, init, eta, M, beta, steps, rho = (spec[k] for k in ("n_l", "lab_draw", "init", "eta", "M", "beta_md", "steps", "rho"))
    Xu, yu, Xt, yt, Xl, yl = make_data(n_l, d)
    Xu_, Xt_, Xl_ = tensors(Xu, Xt, Xl)
    yu_, yt_, yl_ = torch.tensor(yu), torch.tensor(yt), torch.tensor(yl)
    q = torch.tensor(spec["q"])
    logq = q.clamp_min(1e-12).log()
    pq = q.argmax(1)
    torch.manual_seed(1000 * d + init)
    net = mlp()
    init_state = {k: v.clone() for k, v in net.state_dict().items()}
    th0 = {n: p.detach().clone() for n, p in net.named_parameters()}
    truth = F.one_hot(yu_, 2).float()
    I = Inner(net, Xu_, Xl_, yl_)
    meta = Meta(spec, Xu_, Xl_, yl_, xi_seed=555_000 + 10_000 * n_l + 1000 * d + init)
    out = {"n_l": n_l, "lab_draw": d, "init": init, "mode": spec["mode"], "teacher_pl_acc": (pq == yu_).float().mean().item(),
           "train_teacher": train_on(init_state, Xu_, Xt_, Xl_, yl_, yt_, pq, steps, eta), "K": {}}
    t_job = time.time()
    for K in spec["Ks"]:
        meta.reset()
        if K == "IFT":
            th_q, th_t = I.train(th0, q, 1000, eta), I.train(th0, truth, 1000, eta)
            _, F_q, i_q = meta_at(meta, th_q, Xl_, yl_, net, cold=True, need_grad=False)
            meta.reset()
            _, F_true, i_t = meta_at(meta, th_t, Xl_, yl_, net, cold=True, need_grad=False)
        else:
            _, F_q, i_q = unrolled_tr(net, Xu_, Xl_, yl_, int(K), q, eta, meta, need_grad=False, cold=True)
            meta.reset()
            _, F_true, i_t = unrolled_tr(net, Xu_, Xl_, yl_, int(K), truth, eta, meta, need_grad=False, cold=True)
        rec = {"F_teacher": F_q, "F_true": F_true, "Ll_teacher": i_q["Ll"], "Ll_true": i_t["Ll"], "lam": {}}
        for lam in spec["lams"]:
            t0 = time.time()
            if K == "IFT":
                t, Fe, info = anchored_ift_tr(I, net, th0, logq, lam, eta, M, beta, rho, meta, Xl_, yl_)
            else:
                t, Fe, info = anchored_unrolled_tr(net, Xu_, Xl_, yl_, int(K), logq, lam, eta, M, beta, meta)
            pt = t.argmax(1)
            right_q, right_t = pq == yu_, pt == yu_
            rec["lam"][f"{lam:g}"] = dict(
                info, pl_acc=right_t.float().mean().item(), flips=(pt != pq).float().mean().item(),
                corrections=(~right_q & right_t).float().mean().item(), corruptions=(right_q & ~right_t).float().mean().item(),
                kl=(t * (t.clamp_min(1e-12).log() - logq)).sum(1).mean().item(), F=Fe,
                train=train_on(init_state, Xu_, Xt_, Xl_, yl_, yt_, pt, steps, eta), seconds=time.time() - t0)
        out["K"][K] = rec
    out.update({"seconds": time.time() - t_job, "tr_seconds": meta.seconds, "n_eval": meta.n_eval})
    return out


# ----------------------------------------------------------------------------------------
# job lists, sharding and the runner
# ----------------------------------------------------------------------------------------
def cfg_of(a):
    return {"mode": a.mode, "radius": a.radius, "fbeta": a.beta, "M_exp": a.M_exp, "nc": a.nc, "n_random": a.n_random,
            "T_cold": a.T_cold, "T_warm": a.T_warm, "tr_device": a.tr_device, "warm_only": bool(a.warm_only)}


def tag_of(a):
    if a.mode == "opt":
        return f"{a.protocol}_opt"
    return f"{a.protocol}_{a.mode}_r{a.radius:g}_b{a.beta:g}"


def specs_for(a):
    cfg = cfg_of(a)
    n_ls = [int(x) for x in a.n_ls.split(",")]
    if a.protocol == "sec13":
        jobs = []
        for n_l in n_ls:
            for d in range(a.lab_draws):
                for i in range(a.inits):
                    for rule in a.rules.split(","):
                        jobs.append(dict(cfg, kind="run", n_l=n_l, lab_draw=d, init=i, rule=rule, iters=a.iters, lr=0.1,
                                         keep_pl=i == 0, recold=a.recold, refresh13=a.refresh13))
                if a.statics:
                    jobs.append(dict(cfg, kind="statics", n_l=n_l, lab_draw=d, n_students=a.statics))
        return jobs
    if a.protocol == "sec14":
        Ks = [int(k) for k in a.Ks.split(",")] if a.Ks else A14.KS
        return [dict(cfg, n_l=n_l, lab_draw=d, init=i, eta=0.1, M=30, beta=0.3, beta_md=0.3, steps=1000, Ks=Ks,
                     ift=bool(a.ift), keep_pl=(n_l, d, i) == A14.SNAP)
                for n_l in n_ls for d in range(a.lab_draws) for i in range(a.inits)]
    if a.protocol == "sec15":
        Ks = a.Ks.split(",") if a.Ks else A15.KS
        return [dict(cfg, n_l=n_l, lab_draw=d, init=i, eta=0.1, M=20, beta=0.3, beta_md=0.3, steps=1000, rho=1e-2, Ks=Ks,
                     lams=A15.LAMS) for n_l in n_ls for d in range(a.lab_draws) for i in range(a.inits)]
    raise ValueError(a.protocol)


def dispatch(job):
    t0 = time.time()
    try:
        if job.get("kind") == "statics":
            out = statics13(job)
        elif job.get("kind") == "run":
            out = run13(job)
        elif "lams" in job:
            out = job15(job)
        else:
            out = job14(job)
    except Exception as exc:  # one failing job must not lose the whole shard
        import traceback
        traceback.print_exc()
        out = {k: v for k, v in job.items() if k != "q"}
        out["error"] = repr(exc)
    print(f"  done {job.get('kind', '')} n_l={job['n_l']} draw={job['lab_draw']} init={job.get('init', '-')} "
          f"rule={job.get('rule', '-')} in {time.time() - t0:.0f}s", flush=True)
    return out


def teachers15(n_ls, lab_draws):
    tspec = {"iters": 3000, "lr": 0.1, "bu": 128, "tau": 0.95, "lam_u": 1.0, "sigma_w": 0.03, "sigma_s": 0.3}
    return {(n_l, d): A15.fixmatch_teacher(dict(tspec, n_l=n_l, lab_draw=d)) for n_l in n_ls for d in range(lab_draws)}


def main_runs(a):
    jobs = specs_for(a)
    i, n = (int(x) for x in a.shard.split("/"))
    jobs = jobs[i::n]
    if a.protocol == "sec15":
        torch.set_num_threads(1)
        need = sorted({(j["n_l"], j["lab_draw"]) for j in jobs})
        T = {k: A15.fixmatch_teacher({"iters": 3000, "lr": 0.1, "bu": 128, "tau": 0.95, "lam_u": 1.0, "sigma_w": 0.03,
                                      "sigma_s": 0.3, "n_l": k[0], "lab_draw": k[1]}) for k in need}
        for j in jobs:
            j["q"] = T[(j["n_l"], j["lab_draw"])]["q"]
    print(f"{tag_of(a)} shard {a.shard}: {len(jobs)} jobs, {a.procs} processes", flush=True)
    t0 = time.time()
    if a.procs > 1:
        with mp.get_context("spawn").Pool(a.procs) as pool:
            res = pool.map(dispatch, jobs, chunksize=1)
    else:
        res = [dispatch(j) for j in jobs]
    for r_ in res:
        r_.pop("q", None)
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"{tag_of(a)}_s{i}of{n}.json")
    meta = {"protocol": a.protocol, "cfg": cfg_of(a), "shard": a.shard, "minutes": (time.time() - t0) / 60,
            "host": os.uname().nodename, "torch": torch.__version__}
    json.dump(_round({"meta": meta, "runs": res}), open(path, "w"))
    print(f"-> {path} ({meta['minutes']:.1f} min)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", choices=["sec13", "sec14", "sec15"])
    ap.add_argument("--mode", default="pess", choices=["opt", "pess", "exp"])
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=1e4, help="weight of the logit-change penalty D (fibre)")
    ap.add_argument("--M_exp", type=int, default=16, help="random directions of TR-EBPL (antithetic pairs)")
    ap.add_argument("--nc", type=int, default=8, help="damped Gauss-Newton steps of the TR-EBPL retraction")
    ap.add_argument("--n_random", type=int, default=1, help="random starts of the TR-PBPL ascent (cold calls)")
    ap.add_argument("--T_cold", type=int, default=40, help="TR-PBPL: LM steps per start of a cold call")
    ap.add_argument("--T_warm", type=int, default=10, help="TR-PBPL: LM steps of a warm call (persistent adversary)")
    ap.add_argument("--warm_only", type=int, default=1, help="TR-PBPL warm calls: only the warm start (else also eps = 0)")
    ap.add_argument("--tr_device", default="cpu")
    ap.add_argument("--n_ls", default="1,5")
    ap.add_argument("--lab_draws", type=int, default=5)
    ap.add_argument("--inits", type=int, default=8)
    ap.add_argument("--rules", default="dynamic,frozen")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--statics", type=int, default=64)
    ap.add_argument("--recold", type=int, default=200, help="sec13 dynamic: cold (multi-start) call every recold steps")
    ap.add_argument("--refresh13", type=int, default=20, help="sec13 dynamic: rebuild the preconditioner every n steps")
    ap.add_argument("--Ks", default="")
    ap.add_argument("--ift", type=int, default=1)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.summary:
        import toy_two_moons_tr_summary as S  # noqa: E402
        S.main()
    else:
        main_runs(a)
