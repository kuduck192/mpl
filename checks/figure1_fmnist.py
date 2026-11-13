#!/usr/bin/env python3
"""Figure 1 of the draft on FashionMNIST, with the protocol of Sec. 6.2 of the review.

For a pseudo-label error rate rho:
  1. pseudo-labels: true labels of X_u, a fraction rho flipped to a random other class,
     softened (confidence 0.9) so that S(Y_u) is non-empty;
  2. anchor a: trained only on L_u(Y_u, .) from a fixed, label-independent init theta_0
     until L_u <= tau_u (or a step budget);
  3. pessimistic student: projected ascent on L_l - beta*L_u inside B(a, r); report the
     largest L_l among iterates with L_u <= 2 tau_u;
  4. optimistic student: same with descent on L_l + beta*L_u, smallest L_l.
Everything is evaluated with the network in eval mode (BatchNorm statistics frozen after
the anchor is trained), clean inputs, full batch.  r is relative: r = r_rel * ||a - theta_0||.

  python checks/figure1_fmnist.py --out results/figure1.json [--device cuda]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import ssl_fmnist as S  # noqa: E402


def soft_targets(y, rho, gen, conf=0.9):
    n = len(y)
    flip = torch.rand(n, generator=gen) < rho
    other = (y + torch.randint(1, S.NUM_CLASSES, (n,), generator=gen)) % S.NUM_CLASSES
    yy = torch.where(flip, other, y)
    t = torch.full((n, S.NUM_CLASSES), (1 - conf) / (S.NUM_CLASSES - 1))
    t[torch.arange(n), yy] = conf
    return t, flip.float().mean().item()


def kl_mean(t, logits):
    return (t * (torch.log(t) - F.log_softmax(logits, 1))).sum(1).mean()


def run(args):
    dev = torch.device(args.device)
    xtr, ytr, _, _ = S.load_fmnist(args.data_dir)
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(ytr))
    lab = np.concatenate([perm[ytr[perm] == c][:args.n_per_class] for c in range(S.NUM_CLASSES)])
    rest = np.setdiff1d(perm, lab)
    unl = rng.choice(rest, args.n_unl, replace=False)
    Xl = S.clean(torch.from_numpy(xtr[lab]).to(dev))
    Yl = torch.from_numpy(ytr[lab]).to(dev)
    Xu = S.clean(torch.from_numpy(xtr[unl]).to(dev))
    Yu = torch.from_numpy(ytr[unl])
    torch.manual_seed(args.seed)
    net0 = S.ConvNet(args.width).to(dev)
    theta0 = {k: v.detach().clone() for k, v in net0.named_parameters()}
    buf0 = {k: v.detach().clone() for k, v in net0.named_buffers()}
    out = []
    for rho in args.rhos:
        gen = torch.Generator().manual_seed(1000 * args.seed + int(rho * 100))
        T, rho_eff = soft_targets(Yu, rho, gen)
        T = T.to(dev)
        net = S.ConvNet(args.width).to(dev)
        net.load_state_dict({**theta0, **buf0}, strict=False)
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        net.train()
        for step in range(args.anchor_steps):  # anchor: fit the pseudo-labels only
            loss = kl_mean(T, net(Xu))
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        a = {k: v.detach().clone() for k, v in net.named_parameters()}
        bufs = {k: v for k, v in net.named_buffers()}

        def f(p, x):
            return functional_call(net, {**p, **bufs}, (x,))

        with torch.no_grad():
            Lu_a = kl_mean(T, f(a, Xu)).item()
            Ll_a = F.cross_entropy(f(a, Xl), Yl).item()
        dist = S.dnorm(S.dadd(a, theta0, -1.0)).item()
        r = args.r_rel * dist
        tau = max(Lu_a, 1e-4)
        res = {"rho": rho, "rho_eff": rho_eff, "Lu_anchor": Lu_a, "Ll_anchor": Ll_a, "r": r, "dist_anchor_init": dist}
        for name, sign in [("pess", +1.0), ("opt", -1.0)]:
            best = Ll_a
            best_info = (Lu_a, 0.0)
            for restart in range(args.restarts):
                g0 = torch.Generator(device="cpu").manual_seed(restart + 7)
                p = {k: (v + 1e-3 * r * torch.randn(v.shape, generator=g0).to(dev) / max(1.0, v.numel() ** 0.5)).requires_grad_(True)
                     for k, v in a.items()}
                for t in range(args.search_steps):
                    Ll = F.cross_entropy(f(p, Xl), Yl)
                    Lu = kl_mean(T, f(p, Xu))
                    obj = sign * Ll - args.beta * Lu
                    g = torch.autograd.grad(obj, list(p.values()))
                    gn = torch.sqrt(sum((gi ** 2).sum() for gi in g)) + 1e-12
                    with torch.no_grad():
                        for (k, v), gi in zip(p.items(), g):
                            v.add_(args.step * r * gi / gn)
                        e = S.dadd({k: v for k, v in p.items()}, a, -1.0)
                        en = S.dnorm(e)
                        if en > r:
                            for k in p:
                                p[k].copy_(a[k] + e[k] * (r / en))
                        Ll_v, Lu_v = Ll.item(), Lu.item()
                        if Lu_v <= 2 * tau and ((sign > 0 and Ll_v > best) or (sign < 0 and Ll_v < best)):
                            best, best_info = Ll_v, (Lu_v, en.item())
            res[f"Ll_{name}"] = best
            res[f"Lu_{name}"] = best_info[0]
            res[f"dist_{name}"] = best_info[1]
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}), flush=True)
        out.append(res)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/figure1.json")
    ap.add_argument("--data_dir", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rhos", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--n_per_class", type=int, default=10)
    ap.add_argument("--n_unl", type=int, default=2000)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--anchor_steps", type=int, default=1500)
    ap.add_argument("--search_steps", type=int, default=300)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--r_rel", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--step", type=float, default=0.05)
    args = ap.parse_args()
    S.AMP = False
    allres = {}
    for sd in args.seeds:
        args.seed = sd
        allres[sd] = run(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"args": vars(args), "results": allres}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
