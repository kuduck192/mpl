#!/usr/bin/env python3
"""Proposition N2 on the real CNN: feedback at supervised-only pairs.

A network w is trained on the labeled set only; at checkpoints w_k we form the
supervised-only pair (teacher = student = w_k) and measure the teacher's meta-gradient
  g_T = d/dphi L_l( theta'(phi) + eps ),   theta'(phi) = theta - eta grad_theta L_u(p_phi(X), theta)
at phi = theta = w_k, for the optimistic (eps = 0), pessimistic (eps = fiber maximiser in
B(0, r), found by the ascent-then-retract solver) and expected (eps = random direction
retracted to the fiber, averaged) rules.  BatchNorm is in eval mode, inputs are clean, the
student's inputs X are unlabeled images disjoint from the labeled set, so the pair is
exactly self-consistent: theta' = theta and the optimistic feedback is grad L_l(w_k).
Prediction (Prop. N2): ||g_T opt|| -> 0 as w_k fits the labels, ||g_T pess|| does not.

  python checks/degeneracy_static.py --out results/degeneracy_static.json [--device cuda]
"""
import argparse
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


def teacher_grad(net, phi, x_u, c, eta):
    """d/dphi of eta * mean_i <softmax(f_phi(x_i)), c_i>  (= exact meta-gradient, Danskin)."""
    p = {k: v.detach().clone().requires_grad_(True) for k, v in phi.items()}
    t = F.softmax(functional_call(net, p, (x_u,)), 1)
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
        net.eval()
        w = {n: p.detach().clone() for n, p in net.named_parameters()}

        def f(params, x):
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

        Ll, v_opt = lab_grad(w)
        rec = {"seed": seed, "iter": ck, "L_l": Ll, "grad_norm": S.dnorm(v_opt).item(),
               "gT_opt": teacher_grad(net, w, Xu, c_of(v_opt), args.eta)}
        for r in args.radii:
            # pessimistic: persistent ascent-then-retract, start from a random direction
            g0 = torch.Generator(device="cpu").manual_seed(seed * 100 + 1)
            eps = {n: torch.randn(v.shape, generator=g0).to(dev) for n, v in w.items()}
            eps = S.dscale(eps, 0.5 * r / S.dnorm(eps))
            best, best_eps = -1.0, eps
            for t in range(args.steps):
                l, g = lab_grad(S.dadd(w, eps))
                eps = S.dadd(eps, g, args.alpha * r / (S.dnorm(g) + 1e-12))
                eps = retract(eps, args.k_proj)
                en = S.dnorm(eps)
                eps = S.dscale(eps, torch.clamp(r / (en + 1e-12), max=1.0))
                if l > best:
                    best, best_eps = l, eps
            Lp, v_p = lab_grad(S.dadd(w, best_eps))
            rec[f"pess_r{r}_gain"] = Lp - Ll
            rec[f"pess_r{r}_kl"] = kl_at(best_eps)
            rec[f"gT_pess_r{r}"] = teacher_grad(net, w, Xu, c_of(v_p), args.eta)
        for r in args.radii_exp:
            acc = None
            for m in range(args.n_exp):
                g1 = torch.Generator(device="cpu").manual_seed(seed * 1000 + m)
                xi = {n: torch.randn(v.shape, generator=g1).to(dev) for n, v in w.items()}
                xi = retract(S.dscale(xi, r / S.dnorm(xi)), args.k_proj)
                _, v_e = lab_grad(S.dadd(w, xi))
                acc = v_e if acc is None else S.dadd(acc, v_e)
            v_e = S.dscale(acc, 1.0 / args.n_exp)
            rec[f"gT_exp_r{r}"] = teacher_grad(net, w, Xu, c_of(v_e), args.eta)
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
    ap.add_argument("--n_exp", type=int, default=16)
    args = ap.parse_args()
    S.AMP = False
    res = {sd: run(args, sd) for sd in args.seeds}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"args": vars(args), "results": res}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
