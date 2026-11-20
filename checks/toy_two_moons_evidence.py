#!/usr/bin/env python3
"""Two Moons toy: pseudo-labels that maximise the evidence P(Y_l, Y_u | X) (the variational view).

Idea under test: good pseudo-labels should help many models, not a single one.  For every q,
    log P(Y_l, Y_u | X) = E_q[log P(Y_l | X_l, h) + log P(Y_u | X_u, h)] - KL(q || P(h | X)) + KL(q || P(h | X, Y_l, Y_u)),
so argmax_{Y_u} P(Y_l, Y_u | X) = argmax_{Y_u} P(Y_u | X, Y_l), the joint mode of the posterior predictive, and
dropping the last KL gives the ELBO.  Question: does the framework generate bilevel algorithms whose
pseudo-labels beat a student trained on the labeled points only?  Same data, student architecture, labeled
draws and training schedule as report Sections 13-15 (checks/toy_two_moons_meta.py), labeled draws 0-9.

A. Exact evidence, students with a tractable posterior: GP regression on +-1 labels,
       log P(y | X) = -1/2 y^T (K + s2 I)^{-1} y + const,
   K = NNGP kernel of the student MLP at its PyTorch initialisation (the student's own prior) or RBF (length
   0.3, standardised inputs), normalised to unit mean prior variance, s2 = 0.01 or 0.1.  The relaxed optimum is
   the GP posterior mean given Y_l (supervised); hard labelings are searched by greedy single flips (ICM) from
   it, by the sphere relaxation ||y_u||^2 = n_u (global trust-region solution) rounded and polished by ICM, and
   by 10 random starts + ICM.  The evidence of the true labeling (and of ICM started there) is logged to
   separate the objective from the search.
B. Bilevel, neural students (the question).  Free soft pseudo-labels t = softmax(z) on X_u.  Students are
   held out (never see Y_l): from random inits theta_0 ~ P they take K = 20 full-batch SGD steps (lr 0.1) on
   (X_u, t), so q is the law of the trained student; z takes 40 mirror-descent steps (unit-RMS hypergradient,
   step 0.3, as in Sec. 14) by back-propagation through the K steps:
       mpl1           one fixed student, mean labeled CE                      (MPL, free pseudo-labels, Sec. 14)
       ens            8 students redrawn at every step, mean labeled CE       ("help many models")
       ens_lme        8 students, -log mean_m exp(-sum_l CE_m)                (Monte Carlo log P(Y_l | X, Y_u))
       elbo_mean      8 students, mean labeled CE + mean CE(t, p) on X_u      (both ELBO data terms, mean weights)
       elbo_sum       8 students, sum labeled CE + sum CE(t, p) on X_u + KL   (the ELBO's own weights; KL of
                      N(theta_K, s^2 I) to N(theta_0, s^2 I), s^2 = init variance 1 / (3 fan_in) of each layer)
       elbo_mean_sup  elbo_mean started at the labeling of the supervised ensemble
       elbo_lab       elbo_mean with students that also train on Y_l (mean labeled CE + mean CE on X_u), i.e. q
                      fitted to the whole ELBO: the hypergradient then tends to its direct part (envelope theorem),
                      the coordinate-ascent direction, so this bilevel problem is expected to be degenerate
   Two repeats (seeds) each.  CE(t, p) is linear in t (no entropy of t), i.e. the hard max over Y_u.
C. Other members of the framework and references:
       sup_ens  labeling of 8 students trained on Y_l only (argmax of the mean log-probability; q-step, no Y_u)
       st1, st8 coordinate ascent of the ELBO: q-step = 1 or 8 fresh students trained on Y_l + Y_u (1000 steps),
                Y_u-step = argmax_y E_q log p_h(y | x); 5 rounds from sup_ens (st1: from one student)
       fixmatch the FixMatch teacher of Sec. 15 (q = point mass, augmentation-consistent likelihood, threshold)
       ls       label spreading (kNN graph); oracle: the true labels
Evaluation: 8 fresh students per draw, never used in any optimisation, trained 3000 full-batch SGD steps
(lr 0.1) on (X_u, pseudo-labels), and on that plus the labeled points, against the same 8 students trained on
the labeled points only (paired by student).

Pre-registered predictions (written before the full run):
  P1  exact evidence, s2 = 0.01: the truth has a higher evidence than the supervised labeling on most draws,
      under the NNGP and the RBF prior; the framework's own searches do not reliably reach it (gp ICM from the
      supervised labeling stays near it).
  P2  ens / ens_lme: pseudo-label accuracy >= mpl1; fresh students beat supervised-only by a few points, more
      with 1 label per class, and stay below label spreading and FixMatch.
  P3  elbo_sum collapses towards one class (the sum over X_u dominates and one class is the easiest labeling to
      fit in K steps); elbo_mean lies near ens.
  P4  st1 / st8 stay near the supervised labeling (it is a fixed point of coordinate ascent); FixMatch >> sup.

  python checks/toy_two_moons_evidence.py               # runs (4 processes, about an hour on CPU)
  python checks/toy_two_moons_evidence.py --summary     # tables and figures from results/toy-two-moons/evidence.json
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # one BLAS thread per worker process
    os.environ.setdefault(_v, "1")

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_two_moons_anchor as A15  # noqa: E402
from toy_two_moons_meta import FIG, ROOT, _round, label_spreading, make_data, tensors  # noqa: E402

OUT_JSON = os.path.join(ROOT, "results", "toy-two-moons", "evidence.json")
SNAP_JSON = os.path.join(ROOT, "results", "toy-two-moons", "evidence_snapshot.json")
H = 64
BILEVEL = ["mpl1", "ens", "ens_lme", "elbo_mean", "elbo_sum", "elbo_mean_sup", "elbo_lab"]
GP_SETS = [("nngp", 0.01), ("rbf", 0.01), ("nngp", 0.1), ("rbf", 0.1)]
GP_EVAL = [("nngp", 0.01), ("rbf", 0.01)]           # GP labelings also given to neural students
FIXMATCH = {"iters": 3000, "lr": 0.1, "bu": 128, "tau": 0.95, "lam_u": 1.0, "sigma_w": 0.03, "sigma_s": 0.3}


# ----------------------------------------------------------------------------------------
# a stack of M students (the 2-64-64-2 ReLU MLP of Sections 13-15), trained in one batch
# ----------------------------------------------------------------------------------------
FANS = [2, 2, H, H, H, H]
SHAPES = [(2, H), (H,), (H, H), (H,), (H, 2), (2,)]


def init_stack(M, gen):
    """PyTorch's default nn.Linear init: weights and biases U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    return [(torch.rand((M,) + s, generator=gen) * 2 - 1) / math.sqrt(f) for s, f in zip(SHAPES, FANS)]


def fwd(P, X):
    W1, b1, W2, b2, W3, b3 = P
    h = torch.relu(torch.einsum("nd,mdh->mnh", X, W1) + b1[:, None])
    h = torch.relu(torch.einsum("mnd,mdh->mnh", h, W2) + b2[:, None])
    return torch.einsum("mnd,mdh->mnh", h, W3) + b3[:, None]  # M x N x 2


def ce_rows(logits, y):
    """M x N cross-entropies; y: N class indices or N x 2 probabilities shared by the students."""
    lp = F.log_softmax(logits, -1)
    if y.dim() == 1:
        return -lp.gather(-1, y[None, :, None].expand(lp.shape[0], -1, 1)).squeeze(-1)
    return -(y[None] * lp).sum(-1)


def train(P0, X, y, steps, lr=0.1):
    """Full-batch SGD of every student on (X, y); the students do not interact."""
    P = [p.detach().clone().requires_grad_(True) for p in P0]
    opt = torch.optim.SGD(P, lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        ce_rows(fwd(P, X), y).mean(1).sum().backward()
        opt.step()
    return [p.detach() for p in P]


def unroll(P0, t, X, K, eta, Xl=None, yl=None):
    """K SGD steps on (X, t) (plus the mean labeled CE if Xl is given) with the graph kept, so that the result is
    differentiable in t."""
    P = [p.clone().requires_grad_(True) for p in P0]
    for _ in range(K):
        loss = ce_rows(fwd(P, X), t).mean(1).sum()
        if Xl is not None:
            loss = loss + ce_rows(fwd(P, Xl), yl).mean(1).sum()
        g = torch.autograd.grad(loss, P, create_graph=True)
        P = [p - eta * gi for p, gi in zip(P, g)]
    return P


def md_step(z, g, beta):
    g = g - g.mean(1, keepdim=True)
    return z - beta * g / g.pow(2).mean().sqrt().clamp_min(1e-30)


# ----------------------------------------------------------------------------------------
# B. bilevel pseudo-labels with neural students
# ----------------------------------------------------------------------------------------
def bilevel(variant, D, z0, seed, K=20, M=8, steps=40, beta=0.3, eta=0.1):
    Xu_, Xl_, yl_, yu_ = D["Xu"], D["Xl"], D["yl"], D["yu"]
    gen = torch.Generator().manual_seed(seed)
    fixed = init_stack(1, gen) if variant == "mpl1" else None
    s2 = [1.0 / (3 * f) for f in FANS]
    z, trace = z0.clone(), []
    for s in range(steps + 1):
        zt = z.detach().requires_grad_(True)
        t = F.softmax(zt, 1)
        P0 = fixed if fixed is not None else init_stack(M, gen)
        PK = unroll(P0, t, Xu_, K, eta, *((Xl_, yl_) if variant == "elbo_lab" else ()))
        cel = ce_rows(fwd(PK, Xl_), yl_).sum(1)                       # per student, sum over labeled points
        if variant in ("mpl1", "ens"):
            U = (cel / len(yl_)).mean()
        elif variant == "ens_lme":
            U = -(torch.logsumexp(-cel, 0) - math.log(len(cel)))
        else:
            ceu = ce_rows(fwd(PK, Xu_), t)                            # M x N_u, linear in t
            if variant.startswith("elbo_mean") or variant == "elbo_lab":
                U = (cel / len(yl_)).mean() + ceu.mean()
            else:  # elbo_sum
                kl = sum(((pk - p0) ** 2).flatten(1).sum(1) / (2 * v) for pk, p0, v in zip(PK, P0, s2))
                U = (cel + ceu.sum(1) + kl).mean()
        pl = t.argmax(1)
        trace.append({"step": s, "U": U.item(), "lab_ce": (cel / len(yl_)).mean().item(),
                      "pl_acc": (pl == yu_).float().mean().item(), "frac0": (pl == 0).float().mean().item(),
                      "conf": t.max(1).values.mean().item()})
        if s == steps:
            break
        g = torch.autograd.grad(U, zt)[0]
        if not torch.isfinite(g).all():
            trace[-1]["nonfinite"] = True
            break
        z = md_step(z, g, beta)
    return F.softmax(z, 1).argmax(1), trace


# ----------------------------------------------------------------------------------------
# A. exact evidence with GP students
# ----------------------------------------------------------------------------------------
def relu_expect(k12, k11, k22):
    s = np.sqrt(k11 * k22)
    c = np.clip(k12 / s, -1, 1)
    th = np.arccos(c)
    return s / (2 * np.pi) * (np.sin(th) + (np.pi - th) * c)


def nngp(X):
    """NNGP kernel of the 2-64-64-1 ReLU MLP at PyTorch's init (weight and bias variance 1 / (3 fan_in))."""
    k12 = (X @ X.T + 1) / 6
    kd = np.diag(k12).copy()
    for _ in range(2):
        k12 = relu_expect(k12, kd[:, None], kd[None]) / 3 + 1 / 192
        kd = np.diag(k12).copy()
    return k12


def rbf(X, ell=0.3):
    d2 = (X * X).sum(1)[:, None] + (X * X).sum(1)[None] - 2 * X @ X.T
    return np.exp(-np.maximum(d2, 0) / (2 * ell ** 2))


def icm(A, y, free, max_sweeps=500):
    """Greedy single flips lowering y^T A y (y in {-1, +1}^n), only indices in `free` flip."""
    y = y.astype(float).copy()
    r = A @ y
    dA = np.diag(A)
    for _ in range(max_sweeps):
        nflip = 0
        for i in free[np.argsort(-4 * y[free] * r[free] + 4 * dA[free])]:
            if -4 * y[i] * r[i] + 4 * dA[i] < -1e-10:
                r -= 2 * y[i] * A[:, i]
                y[i] = -y[i]
                nflip += 1
        if nflip == 0:
            break
    return y


def sphere_relax(A, yl, nl):
    """Global minimiser of y^T A y over y_u in R^{n_u} with ||y_u||^2 = n_u (trust-region subproblem), y_l fixed."""
    B, c = A[nl:, nl:], A[nl:, :nl] @ yl
    mu, V = np.linalg.eigh(B)
    w = V.T @ c
    n_u = len(c)
    norm2 = lambda lam: float(np.sum((w / (mu - lam)) ** 2))  # noqa: E731
    hi = mu[0] - 1e-10 * max(1.0, abs(mu[0]))
    if norm2(hi) < n_u:  # hard case
        y = -(V @ (w / (mu - hi)))
        return y + math.sqrt(max(n_u - y @ y, 0.0)) * V[:, 0]
    lo = mu[0] - 1.0
    while norm2(lo) > n_u:
        lo = mu[0] - 2 * (mu[0] - lo)
    lam = brentq(lambda l_: norm2(l_) - n_u, lo, hi, xtol=1e-14, maxiter=500)
    return -(V @ (w / (mu - lam)))


def exact_gp(Xu, yu, Xl, yl, d):
    mu_, sd_ = Xu.mean(0), Xu.std(0)
    X = np.concatenate([(Xl - mu_) / sd_, (Xu - mu_) / sd_])
    nl, free = len(yl), np.arange(len(yl), len(yl) + len(yu))
    ylp = (2 * yl - 1).astype(float)
    truth = np.concatenate([ylp, 2 * yu - 1.0])
    acc = lambda y: float(((y[nl:] > 0).astype(int) == yu).mean())  # noqa: E731
    out, labels = {}, {}
    for kern, s2 in GP_SETS:
        K = nngp(X) if kern == "nngp" else rbf(X)
        K = K / np.mean(np.diag(K))
        A = np.linalg.inv(K + s2 * np.eye(len(X)))
        f = lambda y: float(y @ A @ y)  # noqa: E731   (log evidence = -f / 2 + const)
        mean = K[nl:, :nl] @ np.linalg.solve(K[:nl, :nl] + s2 * np.eye(nl), ylp)
        sup = np.concatenate([ylp, np.where(mean >= 0, 1.0, -1.0)])
        cand = {"icm_sup": icm(A, sup, free)}
        sph = sphere_relax(A, ylp, nl)
        cand["sphere"] = icm(A, np.concatenate([ylp, np.where(sph >= 0, 1.0, -1.0)]), free)
        rng = np.random.default_rng(1000 + d)
        rand = [icm(A, np.concatenate([ylp, rng.choice([-1.0, 1.0], len(yu))]), free) for _ in range(10)]
        cand["random"] = min(rand, key=f)
        best = min(cand, key=lambda k: f(cand[k]))
        from_truth = icm(A, truth, free)
        ft = f(truth)
        key = f"{kern}_{s2:g}"
        out[key] = {"acc_sup": acc(sup), "df_sup": f(sup) - ft, "acc_truth_icm": acc(from_truth),
                    "df_truth_icm": f(from_truth) - ft, "best": best,
                    "acc_best": acc(cand[best]), "df_best": f(cand[best]) - ft, "frac0_best": float((cand[best][nl:] < 0).mean()),
                    **{f"acc_{k}": acc(v) for k, v in cand.items()}, **{f"df_{k}": f(v) - ft for k, v in cand.items()},
                    "truth_beats_sup": f(sup) > ft, "search_gap": f(cand[best]) > ft}
        if (kern, s2) in GP_EVAL:
            labels[f"gp_{kern}_icm"] = (cand["icm_sup"][nl:] > 0).astype(int)
            labels[f"gp_{kern}_best"] = (cand[best][nl:] > 0).astype(int)
            if kern == "nngp":
                labels["gp_nngp_sup"] = (sup[nl:] > 0).astype(int)
    return out, labels


# ----------------------------------------------------------------------------------------
# C. coordinate ascent of the ELBO (not bilevel)
# ----------------------------------------------------------------------------------------
def ensemble_label(P, X):
    return F.log_softmax(fwd(P, X), -1).mean(0).argmax(1)


def self_train(D, M, seed, rounds=5, steps=1000):
    gen = torch.Generator().manual_seed(seed)
    Xu_, Xl_, yl_, yu_ = D["Xu"], D["Xl"], D["yl"], D["yu"]
    P = train(init_stack(M, gen), Xl_, yl_, 3000)
    pl = ensemble_label(P, Xu_)
    trace = [{"round": 0, "pl_acc": (pl == yu_).float().mean().item(), "frac0": (pl == 0).float().mean().item()}]
    for r in range(1, rounds + 1):
        P = train(init_stack(M, gen), torch.cat([Xl_, Xu_]), torch.cat([yl_, pl]), steps)
        new = ensemble_label(P, Xu_)
        trace.append({"round": r, "pl_acc": (new == yu_).float().mean().item(), "frac0": (new == 0).float().mean().item(),
                      "changed": (new != pl).float().mean().item()})
        pl = new
    return pl, trace


# ----------------------------------------------------------------------------------------
# one labeled draw
# ----------------------------------------------------------------------------------------
def compute_labels(n_l, d, a, out):
    """Every pseudo-label set of one labeled draw (each with its own seed, so any subset is reproducible)."""
    Xu, yu, Xt, yt, Xl, yl = make_data(n_l, d)
    Xu_, Xt_, Xl_ = tensors(Xu, Xt, Xl)
    D = {"Xu": Xu_, "Xl": Xl_, "yl": torch.tensor(yl), "yu": torch.tensor(yu)}
    yt_ = torch.tensor(yt)
    labels = {"oracle": D["yu"], "ls": torch.tensor(label_spreading(Xu, Xl, yl))}
    # C. supervised ensemble, coordinate ascent, FixMatch teacher
    P_sup = train(init_stack(8, torch.Generator().manual_seed(900_000 + d)), Xl_, D["yl"], 3000)
    labels["sup_ens"] = ensemble_label(P_sup, Xu_)
    out["sup_ens_test"] = float((F.log_softmax(fwd(P_sup, Xt_), -1).mean(0).argmax(1) == yt_).float().mean())
    for M, name in ((1, "st1"), (8, "st8")):
        labels[name], out["selftrain"][name] = self_train(D, M, 910_000 + 10 * d + M, rounds=a["st_rounds"])
    T = A15.fixmatch_teacher(dict(FIXMATCH, n_l=n_l, lab_draw=d, iters=a["fm_iters"]))
    labels["fixmatch"] = torch.tensor(np.array(T["q"])).argmax(1)
    # A. exact evidence
    out["gp"], gpl = exact_gp(Xu, yu, Xl, yl, d)
    labels.update({k: torch.tensor(v) for k, v in gpl.items()})
    # B. bilevel
    z_sup = 3.0 * (2 * F.one_hot(labels["sup_ens"], 2).float() - 1)
    for vi, v in enumerate(BILEVEL):
        for rep in range(a["reps"]):
            t0 = time.time()
            z0 = z_sup if v == "elbo_mean_sup" else torch.zeros(len(yu), 2)
            seed = 800_000 + 10_000 * n_l + 100 * d + 10 * vi + rep
            pl, trace = bilevel(v, D, z0, seed, K=a["K"], M=a["M"], steps=a["outer"])
            labels[f"{v}#{rep}"] = pl
            out["bilevel"][f"{v}#{rep}"] = {"trace": trace, "seconds": time.time() - t0}
    return labels


def job(spec):
    torch.set_num_threads(1)
    n_l, d, a = spec["n_l"], spec["lab_draw"], spec["args"]
    t_start = time.time()
    Xu, yu, Xt, yt, Xl, yl = make_data(n_l, d)
    Xu_, Xt_, Xl_ = tensors(Xu, Xt, Xl)
    yl_, yu_, yt_ = torch.tensor(yl), torch.tensor(yu), torch.tensor(yt)
    P_eval = init_stack(a["n_eval"], torch.Generator().manual_seed(700_000 + d))
    out = {"n_l": n_l, "lab_draw": d, "sets": {}, "bilevel": {}, "selftrain": {}}

    def test_acc(P):
        with torch.no_grad():
            return (fwd(P, Xt_).argmax(-1) == yt_[None]).float().mean(1).tolist()

    out["sup"] = test_acc(train(P_eval, Xl_, yl_, a["eval_steps"]))
    labels = compute_labels(n_l, d, a, out)
    # evaluation: fresh students on every labeling (alone, and with the labeled points)
    for name, pl in labels.items():
        pl = pl.long()
        rec = {"pl_acc": (pl == yu_).float().mean().item(), "frac0": (pl == 0).float().mean().item(),
               "pl_only": test_acc(train(P_eval, Xu_, pl, a["eval_steps"]))}
        if a["with_lab"]:
            rec["pl_lab"] = test_acc(train(P_eval, torch.cat([Xl_, Xu_]), torch.cat([yl_, pl]), a["eval_steps"]))
        out["sets"][name] = rec
    out["minutes"] = (time.time() - t_start) / 60
    print(f"n_l={n_l} draw={d}: {out['minutes']:.1f} min", flush=True)
    return out


def snapshot(a, n_l, d):
    """Recompute the labelings of one draw (same seeds as the main run) and keep them for the figure."""
    torch.set_num_threads(1)
    args = {k: getattr(a, k) for k in ("K", "M", "outer", "reps", "st_rounds", "fm_iters")}
    out = {"n_l": n_l, "lab_draw": d, "bilevel": {}, "selftrain": {}}
    labels = compute_labels(n_l, d, args, out)
    Xu, yu, _, _, Xl, yl = make_data(n_l, d)
    snap = {"n_l": n_l, "lab_draw": d, "Xu": Xu.tolist(), "yu": yu.tolist(), "Xl": Xl.tolist(), "yl": yl.tolist(),
            "labels": {k: v.long().tolist() for k, v in labels.items()}}
    path = os.path.join(os.path.dirname(os.path.abspath(a.out)), os.path.basename(SNAP_JSON))
    json.dump(_round(snap), open(path, "w"))
    print("snapshot ->", path)
    return path


def main_runs(a):
    args = {k: getattr(a, k) for k in ("n_eval", "eval_steps", "K", "M", "outer", "reps", "st_rounds", "fm_iters", "with_lab")}
    specs = [{"n_l": n_l, "lab_draw": d, "args": args} for n_l in (1, 5) for d in range(a.lab_draws)]
    t0 = time.time()
    with Pool(a.procs) as pool:
        runs = pool.map(job, specs, chunksize=1)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    meta = dict(args, lab_draws=a.lab_draws, bilevel=BILEVEL, gp_sets=[f"{k}_{s:g}" for k, s in GP_SETS],
                fixmatch=FIXMATCH, minutes=(time.time() - t0) / 60)
    json.dump(_round({"meta": meta, "runs": runs}), open(a.out, "w"))
    print(f"{len(runs)} draws in {meta['minutes']:.1f} min -> {a.out}")


# ----------------------------------------------------------------------------------------
# summary and figures
# ----------------------------------------------------------------------------------------
GROUPS = [("B. bilevel, neural students", [f"{v}" for v in BILEVEL]),
          ("A. exact evidence, GP students", ["gp_nngp_sup", "gp_nngp_icm", "gp_nngp_best", "gp_rbf_icm", "gp_rbf_best"]),
          ("C. coordinate ascent and references", ["sup_ens", "st1", "st8", "fixmatch", "ls", "oracle"])]


def holm(p):
    p = np.asarray(p, float)
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        adj[j] = run
    return adj


def per_draw(runs, name, key):
    """name -> per-draw mean over repeats and evaluation students of (student acc - supervised-only acc)."""
    out = []
    for r in runs:
        names = [k for k in r["sets"] if k == name or k.split("#")[0] == name]
        if not names:
            continue
        diffs = [np.mean(np.array(r["sets"][k][key]) - np.array(r["sup"])) for k in names]
        out.append(float(np.mean(diffs)))
    return np.array(out)


def summary(R, key="pl_only"):
    runs = R["runs"]
    for n_l in (1, 5):
        rs = [r for r in runs if r["n_l"] == n_l]
        sup = np.mean([np.mean(r["sup"]) for r in rs])
        print(f"\n=== {n_l} labeled point(s) per class, {len(rs)} draws, {len(rs[0]['sup'])} fresh students each; "
              f"supervised-only {sup:.3f}; students trained on {'pseudo-labels only' if key == 'pl_only' else 'pseudo-labels + labeled points'}")
        rows = []
        for title, names in GROUPS:
            for n in names:
                sets = [(r, k) for r in rs for k in r["sets"] if k == n or k.split("#")[0] == n]
                if not sets:
                    continue
                d = per_draw(rs, n, key)
                p_t = stats.ttest_1samp(d, 0).pvalue if len(d) > 1 and d.std() > 0 else np.nan
                p_w = stats.wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else np.nan
                rows.append((title, n, np.mean([r["sets"][k]["pl_acc"] for r, k in sets]),
                             np.mean([np.mean(r["sets"][k][key]) for r, k in sets]),
                             np.mean([r["sets"][k]["frac0"] for r, k in sets]), d, p_t, p_w))
        tested = [i for i, row in enumerate(rows) if row[1] != "oracle" and np.isfinite(row[6])]
        adj = np.full(len(rows), np.nan)
        adj[tested] = holm([rows[i][6] for i in tested])
        print("| group | labeling | PL acc | student acc | frac class 0 | − sup (per draw) | draws better | t p | Wilcoxon p | Holm p |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for (title, n, pa, sa, f0, d, pt, pw), h in zip(rows, adj):
            print(f"| {title.split('.')[0]} | {n} | {pa:.3f} | {sa:.3f} | {f0:.2f} | {d.mean():+.3f} ± {d.std(ddof=1):.3f} | "
                  f"{int((d > 0).sum())}/{len(d)} | {pt:.3g} | {pw:.3g} | {h:.3g} |")
        print("\nExact evidence (GP), mean over draws: accuracy of the labelings, and f - f(truth) where f = -2 log evidence + const")
        print("| prior | sup acc | f(sup) - f(truth) | truth beats sup | ICM from sup | sphere + ICM | random + ICM | best found (acc, f - f(truth)) | best found worse than truth | ICM from truth |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for key_gp in R["meta"]["gp_sets"]:
            G = [r["gp"][key_gp] for r in rs]
            m = lambda k: np.mean([g[k] for g in G])  # noqa: E731
            print(f"| {key_gp} | {m('acc_sup'):.3f} | {np.median([g['df_sup'] for g in G]):+.0f} (median) | "
                  f"{sum(g['truth_beats_sup'] for g in G)}/{len(G)} | {m('acc_icm_sup'):.3f} | {m('acc_sphere'):.3f} | "
                  f"{m('acc_random'):.3f} | {m('acc_best'):.3f}, {np.median([g['df_best'] for g in G]):+.0f} | "
                  f"{sum(g['search_gap'] for g in G)}/{len(G)} | {m('acc_truth_icm'):.3f} |")
        print("\nBilevel traces, mean over draws and repeats: PL acc / class-0 share / labeled CE of the students at outer step 0, 10, 20, 40")
        for v in BILEVEL:
            T = [r["bilevel"][k]["trace"] for r in rs for k in r["bilevel"] if k.split("#")[0] == v]
            cols = []
            for s in (0, 10, 20, 40):
                pts = [t[s] for t in T if len(t) > s]
                cols.append(f"{np.mean([p['pl_acc'] for p in pts]):.3f} / {np.mean([p['frac0'] for p in pts]):.2f} / "
                            f"{np.mean([p['lab_ce'] for p in pts]):.2f}")
            sec = np.mean([r["bilevel"][k]["seconds"] for r in rs for k in r["bilevel"] if k.split("#")[0] == v])
            print(f"  {v:14s} " + " | ".join(cols) + f"  ({sec:.0f} s per run)")
        print("\nCoordinate ascent: PL acc per round (mean over draws), share changed in the last round")
        for name in ("st1", "st8"):
            T = [r["selftrain"][name] for r in rs]
            print(f"  {name}: " + " -> ".join(f"{np.mean([t[i]['pl_acc'] for t in T]):.3f}" for i in range(len(T[0])))
                  + f"; changed in the last round {np.mean([t[-1]['changed'] for t in T]):.4f}")


def plots(R, fig_dir=FIG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({
        "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 2.0, "savefig.dpi": 200, "figure.facecolor": "white"})
    runs = R["runs"]
    col = {"B": BLUE, "A": ORANGE, "C": AQUA}
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9), sharey=True)
    for ax, n_l in zip(axes, (1, 5)):
        rs = [r for r in runs if r["n_l"] == n_l]
        lo = 0.45
        ax.plot([lo, 1], [lo, 1], color=INK2, lw=0.8, ls=(0, (4, 3)), zorder=1)
        sup = np.mean([np.mean(r["sup"]) for r in rs])
        ax.axvline(sup, color=INK2, lw=1.0, ls=(0, (1, 2)), zorder=1)
        ax.annotate(f"labeled points only\n{sup:.3f}", (sup, lo + 0.01), xytext=(3, 0), textcoords="offset points",
                    fontsize=6.5, color=INK2, va="bottom")
        for title, names in GROUPS:
            g = title[0]
            for n in names:
                if n == "oracle":
                    continue
                sets = [(r, k) for r in rs for k in r["sets"] if k == n or k.split("#")[0] == n]
                if not sets:
                    continue
                x = np.mean([np.mean(r["sets"][k]["pl_only"]) for r, k in sets])
                y = np.mean([r["sets"][k]["pl_acc"] for r, k in sets])
                ax.scatter([x], [y], s=26, color=col[g], edgecolors="white", linewidths=0.6, zorder=3)
                ax.annotate(n, (x, y), xytext=(4, 2), textcoords="offset points", fontsize=6, color=INK)
        ax.set_xlim(lo, 1.0)
        ax.set_ylim(lo, 1.02)
        ax.set_xlabel("fresh-student test accuracy (mean over draws)")
        ax.set_title(f"{n_l} labeled point{'s' if n_l > 1 else ''} per class", fontsize=8, color=INK, loc="left")
    axes[0].set_ylabel("pseudo-label accuracy on X_u")
    handles = [plt.Line2D([], [], marker="o", ls="", color=col[t[0]], ms=5, label=t) for t, _ in GROUPS]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    p = os.path.join(fig_dir, "toy_moons_evidence.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(p)


def plot_snapshot(S, R, fig_dir=FIG):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    BLUE, ORANGE = "#2a78d6", "#eb6834"
    INK, INK2 = "#0b0b0b", "#52514e"
    plt.rcParams.update({"font.size": 8, "savefig.dpi": 200, "figure.facecolor": "white"})
    Xu, yu, Xl, yl = (np.array(S[k]) for k in ("Xu", "yu", "Xl", "yl"))
    run = next(r for r in R["runs"] if r["n_l"] == S["n_l"] and r["lab_draw"] == S["lab_draw"])
    panels = [("sup_ens", "supervised ensemble"), ("mpl1#0", "B. mpl1 (one student)"), ("ens#0", "B. ens (8 students)"),
              ("elbo_mean#0", "B. elbo_mean"), ("elbo_sum#0", "B. elbo_sum"), ("elbo_lab#0", "B. elbo_lab"),
              ("gp_nngp_best", "A. GP evidence, NNGP prior"), ("gp_rbf_best", "A. GP evidence, RBF prior"),
              ("st8", "C. coordinate ascent, 8 students"), ("fixmatch", "C. FixMatch"), ("ls", "label spreading")]
    panels = [p for p in panels if p[0] in S["labels"]]
    nc = 4
    nr = math.ceil(len(panels) / nc)
    fig, axes = plt.subplots(nr, nc, figsize=(7.6, 1.95 * nr))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (k, title) in zip(axes.ravel(), panels):
        pl = np.array(S["labels"][k])
        wrong = pl != yu
        cols = np.where(pl == 0, BLUE, ORANGE)
        ax.scatter(Xu[~wrong, 0], Xu[~wrong, 1], s=2.5, c=cols[~wrong], lw=0)
        ax.scatter(Xu[wrong, 0], Xu[wrong, 1], s=6, c=cols[wrong], marker="x", lw=0.5)
        for c_, col_ in ((0, BLUE), (1, ORANGE)):
            m = yl == c_
            ax.scatter(Xl[m, 0], Xl[m, 1], s=40, c=col_, edgecolors=INK, linewidths=1.0, zorder=5)
        stu = np.mean(run["sets"][k]["pl_only"]) if k in run["sets"] else float("nan")
        ax.set_title(f"{title}\nPL {(pl == yu).mean():.3f}, students {stu:.3f}", fontsize=7, color=INK, loc="left")
        ax.set_aspect("equal")
    fig.text(0.01, 0.0, f"{S['n_l']} labeled point(s) per class, draw {S['lab_draw']} (large dots).  Colour = pseudo-label, "
             f"x = wrong; students = mean test accuracy of 8 fresh students trained on the labeling; supervised-only "
             f"{np.mean(run['sup']):.3f}.", fontsize=6.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = os.path.join(fig_dir, "toy_moons_evidence_snapshot.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--lab_draws", type=int, default=10)
    ap.add_argument("--n_eval", type=int, default=8)
    ap.add_argument("--eval_steps", type=int, default=3000)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--M", type=int, default=8)
    ap.add_argument("--outer", type=int, default=40)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--st_rounds", type=int, default=5)
    ap.add_argument("--fm_iters", type=int, default=3000)
    ap.add_argument("--with_lab", type=int, default=1)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--out", default=OUT_JSON)
    ap.add_argument("--fig_dir", default=FIG)
    ap.add_argument("--snapshot", default="", help="n_l,draw: recompute and keep the labelings of one draw")
    a = ap.parse_args()
    if a.snapshot:
        path = snapshot(a, *(int(x) for x in a.snapshot.split(",")))
        plot_snapshot(json.load(open(path)), json.load(open(a.out)), a.fig_dir)
        sys.exit(0)
    if not a.summary:
        main_runs(a)
    R = json.load(open(a.out))
    summary(R)
    summary(R, key="pl_lab")
    plots(R, a.fig_dir)
