#!/usr/bin/env python3
"""Proposition N2 on the real CNN: feedback at supervised-only pairs.

A network w is trained on the labeled set only; at checkpoints w_k we form the
supervised-only pair (teacher = student = w_k) and measure the teacher's meta-gradient
  g_T = d/dphi L_l( theta'(phi) + eps ),   theta'(phi) = theta - eta grad_theta L_u(p_phi(X), theta)
at phi = theta = w_k, for the optimistic (eps = 0), pessimistic (eps = fiber maximiser in
B(0, r), found by the ascent-then-retract solver) and expected (eps = random direction
retracted to the fiber, averaged) rules.  Inputs are clean, the student's inputs X are
unlabeled images disjoint from the labeled set, and BatchNorm uses the statistics of the
batch it is applied to without updating running statistics (as the training-time feedback
does; `--bn eval` uses running statistics instead).  Teacher and student compute the same
function of X, so the pair is exactly self-consistent: theta' = theta and the optimistic
feedback is grad L_l(w_k).  The pessimistic maximiser is taken among *admissible* iterates
only (mean KL to the anchor on X at most --kl_tol); the KL of every reported perturbation
is stored with it.
Prediction (Prop. N2): ||g_T opt|| -> 0 as w_k fits the labels, ||g_T pess|| does not.

  python checks/degeneracy_static.py --out results/degeneracy_static.json [--device cuda]
"""
import argparse
import contextlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ssl_fmnist as S  # noqa: E402


def teacher_grad(f, phi, x_u, c, eta):
    """d/dphi of eta * mean_i <softmax(f_phi(x_i)), c_i>  (= exact meta-gradient, Danskin)."""
    p = {k: v.detach().clone().requires_grad_(True) for k, v in phi.items()}
    t = F.softmax(f(p, x_u), 1)
    l = eta * (t * c).sum(1).mean()
    g = torch.autograd.grad(l, list(p.values()))
    return torch.sqrt(sum((gi ** 2).sum() for gi in g)).item()


def run(args, seed):
    dev = torch.device(args.device)
    xtr, ytr, _, _ = S.load_fmnist(args.data_dir)
    lab, unl, _ = S.make_split(ytr, args.n_per_class, args.n_val, seed)
    unl = np.setdiff1d(unl, lab)
    rng = np.random.RandomState(seed)
    xu_idx = rng.choice(unl, args.n_u, replace=False)
    Xl = S.clean(torch.from_numpy(xtr[lab]).to(dev))
    Yl = torch.from_numpy(ytr[lab]).to(dev)
    Xl_raw = torch.from_numpy(xtr[lab]).to(dev)
    Xu = S.clean(torch.from_numpy(xtr[xu_idx]).to(dev))
    torch.manual_seed(seed)
    net = S.ConvNet(args.width).to(dev)
    opt = S.make_sgd(net, args.lr, 5e-4)
    out = []
    k = 0
    for ck in args.checkpoints:
        net.train()
        while k < ck:  # supervised-only training (weak augmentation), labeled data only
            i = torch.randint(0, len(Xl_raw), (64,), device=dev)
            loss = F.cross_entropy(net(S.weak_aug(Xl_raw[i])), Yl[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            k += 1
        if args.bn == "eval":
            net.eval()
            bn_ctx = contextlib.nullcontext
        else:
            net.train()
            bn_ctx = lambda: S.bn_frozen_stats(net)  # noqa: E731
        w = {n: p.detach().clone() for n, p in net.named_parameters()}

        def f(params, x):
            with bn_ctx():
                return functional_call(net, params, (x,))

        def lab_grad(params):
            p = {n: v.detach().clone().requires_grad_(True) for n, v in params.items()}
            l = F.cross_entropy(f(p, Xl), Yl)
            g = torch.autograd.grad(l, list(p.values()))
            return l.item(), dict(zip(p.keys(), g))

        def c_of(v):  # J_{f_w}(X_u) v
            _, c = jvp(lambda q: f(q, Xu), (w,), (v,))
            return c.detach()

        with torch.no_grad():
            anchor = F.softmax(f(w, Xu), 1)

        def retract(eps, K):
            for _ in range(K):
                p = {n: (w[n] + eps[n]).requires_grad_(True) for n in w}
                D = S.kl_anchor(anchor, f(p, Xu)).mean()
                gD = torch.autograd.grad(D, list(p.values()))
                gD = dict(zip(p.keys(), gD))
                step = 2.0 * D.detach() / (sum((g ** 2).sum() for g in gD.values()) + 1e-20)
                eps = {n: eps[n] - step * gD[n] for n in eps}
            return eps

        def kl_at(eps):
            with torch.no_grad():
                return S.kl_anchor(anchor, f({n: w[n] + eps[n] for n in w}, Xu)).mean().item()

        def ball(eps, r):
            return S.dscale(eps, torch.clamp(r / (S.dnorm(eps) + 1e-12), max=1.0))

        Ll, v_opt = lab_grad(w)
        rec = {"seed": seed, "iter": ck, "bn": args.bn, "L_l": Ll, "grad_norm": S.dnorm(v_opt).item(),
               "gT_opt": teacher_grad(f, w, Xu, c_of(v_opt), args.eta)}
        for r in args.radii:
            # pessimistic: ascent-then-retract from a random direction retracted to the fiber;
            # the maximiser is the best admissible iterate (KL <= kl_tol); eps = 0 is admissible.
            g0 = torch.Generator(device="cpu").manual_seed(seed * 100 + 1)
            eps = {n: torch.randn(v.shape, generator=g0).to(dev) for n, v in w.items()}
            eps = ball(retract(S.dscale(eps, 0.5 * r / S.dnorm(eps)), args.k_init), r)
            best, best_eps, best_kl, n_adm = Ll, S.dzeros(w), 0.0, 0
            for t in range(args.steps + 1):
                l, g = lab_grad(S.dadd(w, eps))
                kl = kl_at(eps)
                if kl <= args.kl_tol:
                    n_adm += 1
                    if l > best:
                        best, best_eps, best_kl = l, eps, kl
                if t == args.steps:
                    break
                eps = S.dadd(eps, g, args.alpha * r / (S.dnorm(g) + 1e-12))
                eps = ball(retract(eps, args.k_proj), r)
            Lp, v_p = lab_grad(S.dadd(w, best_eps))
            rec[f"pess_r{r}_gain"] = Lp - Ll
            rec[f"pess_r{r}_kl"] = best_kl
            rec[f"pess_r{r}_norm"] = S.dnorm(best_eps).item()
            rec[f"pess_r{r}_adm"] = n_adm / (args.steps + 1)
            rec[f"gT_pess_r{r}"] = teacher_grad(f, w, Xu, c_of(v_p), args.eta)
        for r in args.radii_exp:
            acc, kls = None, []
            for m in range(args.n_exp):
                g1 = torch.Generator(device="cpu").manual_seed(seed * 1000 + m)
                xi = {n: torch.randn(v.shape, generator=g1).to(dev) for n, v in w.items()}
                xi = retract(S.dscale(xi, r / S.dnorm(xi)), args.k_init)
                kls.append(kl_at(xi))
                _, v_e = lab_grad(S.dadd(w, xi))
                acc = v_e if acc is None else S.dadd(acc, v_e)
            v_e = S.dscale(acc, 1.0 / args.n_exp)
            rec[f"exp_r{r}_kl"] = float(np.mean(kls))
            rec[f"gT_exp_r{r}"] = teacher_grad(f, w, Xu, c_of(v_e), args.eta)
        print(json.dumps({k2: (round(v2, 6) if isinstance(v2, float) else v2) for k2, v2 in rec.items()}), flush=True)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/degeneracy_static.json")
    ap.add_argument("--data_dir", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n_per_class", type=int, default=10)
    ap.add_argument("--n_val", type=int, default=5000)
    ap.add_argument("--n_u", type=int, default=448)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--eta", type=float, default=0.03)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=[250, 500, 1000, 2000, 4000, 8000])
    ap.add_argument("--radii", type=float, nargs="+", default=[0.1, 0.3])
    ap.add_argument("--radii_exp", type=float, nargs="+", default=[3.0])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--k_proj", type=int, default=2)
    ap.add_argument("--k_init", type=int, default=10, help="retraction steps for a random start")
    ap.add_argument("--kl_tol", type=float, default=2e-3, help="admissibility: mean KL to the anchor on X")
    ap.add_argument("--bn", choices=["batch", "eval"], default="batch")
    ap.add_argument("--n_exp", type=int, default=16)
    args = ap.parse_args()
    S.AMP = False
    res = {sd: run(args, sd) for sd in args.seeds}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"args": vars(args), "results": res}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
