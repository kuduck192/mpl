#!/usr/bin/env python3
"""Trust-region meta objectives of a Two Moons student (TR-PBPL / TR-EBPL), with exact hypergradients.

Used by checks/toy_two_moons_tr.py, which runs them under the protocols of the MPL toys
(toy_two_moons_meta.py, toy_two_moons_unroll.py, toy_two_moons_anchor.py).

Student: the 2-64-64-2 ReLU MLP of the toys as a function of a flat parameter vector w (layout of
nn.Sequential(Linear, ReLU, Linear, ReLU, Linear).named_parameters()); delta_w(x) = f_w(x)_1 - f_w(x)_0 is
its logit difference, which with two classes carries the prediction p_w(x).

Admissible students of an anchor theta (theory.md, Definitions 1-2, with X = X_u): students within distance r
whose predictions on X_u stay those of the anchor.  The exact fibre {delta_y(X_u) = delta_theta(X_u)} is not
usable numerically here: the tangent kernel K = J J^T on X_u (J = d delta(X_u) / dw, 1000 x 4482) has
numerical rank ~700-800 and a continuum of singular values down to 1e-9 sigma_max, so exact retractions turn
residuals of 1e-3 into parameter steps of order one.  The fibre is therefore enforced by a penalty on the
change of the logits on X_u,
    D_theta(y) = (1/N_u) sum_{x in X_u} (delta_y(x) - delta_theta(x))^2        (no dependence on confidence),
with weight beta (the penalty form F_beta of the review with gamma = inf, in logit space); the RMS logit
change sqrt(D) of every admissible student is logged.

Meta objectives at the anchor:
    opt  (MPL)     L_l(theta)
    pess (TR-PBPL) V(theta) = max_{||eps|| <= r}  Phi(theta, eps),   Phi = L_l(theta + eps) - beta D_theta(theta + eps)
    exp  (TR-EBPL) E(theta) = mean_j L_l(y_j(theta)),
                   y_j(theta) = argmin_y  beta D_theta(y) + 1/2 ||y - theta - r xi_j||^2
                   (proximal retraction of theta + r xi_j; xi_j a fixed set of unit directions in antithetic
                   pairs, uniform on the sphere: the "re-trained random student" of theory.md, Definition 2).
Gradients with respect to the anchor (the fibre moves with theta):
    pess: Danskin, grad V(theta) = d/dtheta Phi(theta, eps*) at fixed eps* (the ball constrains eps, not theta),
          including the path through the anchor's logits in D.  ssl_fmnist.py uses grad L_l(theta + eps*)
          ("fixed eps"); both are logged.
    exp : automatic differentiation through the fixed-point iterations of the proximal retraction, with the
          preconditioner detached (the derivative of the fixed point does not depend on it).
Inner solvers (float64):
    pess: Levenberg-Marquardt steps with the Jacobian at the current point, s = (mu I + (2 beta / N) J_y^T J_y)^-1
          (grad Phi - 2 nu eps), radial projection on the ball, monotone acceptance; zeros of s are exactly the
          KKT points.  Starts: eps = 0, a warm start (persistent adversary) and random soft-tangent directions.
    exp : damped Gauss-Newton steps preconditioned at the anchor, (I + (2 beta / N) J_theta^T J_theta)^-1.
J (N_u x d) is never formed: J v, J^T lam and the Gram K = J J^T are computed layer by layer (Woodbury identity
for the inverses).  Measured on this toy: the labeled gradient lies almost entirely in the row space of J (its
component in the soft tangent space is 1e-3 to 2e-2 of its norm), so the trust-region values stay within
~0.1-2% of L_l unless the fibre is loose (beta <= 10, unlabeled probabilities moving by >= 5%).

  python checks/tr_fiber.py --check      # unit tests and finite-difference gradient checks
"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ----------------------------------------------------------------------------------------
# the student as a function of a flat parameter vector
# ----------------------------------------------------------------------------------------
class FlatMLP:
    """2-64-64-2 ReLU MLP; w has shape (d,) or (M, d)."""

    def __init__(self, width=64):
        h = width
        self.shapes = [(h, 2), (h,), (h, h), (h,), (2, h), (2,)]
        self.sizes = [math.prod(s) for s in self.shapes]
        self.d = sum(self.sizes)

    def split(self, w):
        lead = w.shape[:-1]
        out, i = [], 0
        for s, k in zip(self.shapes, self.sizes):
            out.append(w[..., i:i + k].reshape(*lead, *s))
            i += k
        return out

    def logits(self, w, X):
        W1, b1, W2, b2, W3, b3 = self.split(w)
        if w.dim() == 1:
            h1 = torch.relu(X @ W1.T + b1)
            h2 = torch.relu(h1 @ W2.T + b2)
            return h2 @ W3.T + b3
        h1 = torch.relu(torch.einsum("nk,mjk->mnj", X, W1) + b1[:, None])
        h2 = torch.relu(torch.einsum("mnk,mjk->mnj", h1, W2) + b2[:, None])
        return torch.einsum("mnk,mjk->mnj", h2, W3) + b3[:, None]

    def delta(self, w, X):
        z = self.logits(w, X)
        return z[..., 1] - z[..., 0]

    def parts(self, w, X):
        """Per-input factors of the Jacobian of delta at a single w (d,)."""
        W1, b1, W2, b2, W3, b3 = self.split(w)
        a1 = X @ W1.T + b1
        h1 = torch.relu(a1)
        a2 = h1 @ W2.T + b2
        h2 = torch.relu(a2)
        g2 = (W3[1] - W3[0]) * (a2 > 0)
        g1 = (g2 @ W2) * (a1 > 0)
        return {"X": X, "h1": h1, "h2": h2, "g1": g1, "g2": g2}

    def jvp(self, P, v):
        """J v for v (d,) -> (N,), or (M, d) -> (M, N)."""
        V1, c1, V2, c2, V3, c3 = self.split(v)
        X, h1, h2, g1, g2 = P["X"], P["h1"], P["h2"], P["g1"], P["g2"]
        if v.dim() == 1:
            return (((X @ V1.T + c1) * g1).sum(-1) + ((h1 @ V2.T + c2) * g2).sum(-1)
                    + h2 @ (V3[1] - V3[0]) + (c3[1] - c3[0]))
        return ((torch.einsum("nk,mjk->mnj", X, V1) + c1[:, None]) * g1).sum(-1) \
            + ((torch.einsum("nk,mjk->mnj", h1, V2) + c2[:, None]) * g2).sum(-1) \
            + torch.einsum("nk,mk->mn", h2, V3[:, 1] - V3[:, 0]) + (c3[:, 1] - c3[:, 0])[:, None]

    def vjp(self, P, lam):
        """J^T lam for lam (N,) -> (d,), or (M, N) -> (M, d)."""
        X, h1, h2, g1, g2 = P["X"], P["h1"], P["h2"], P["g1"], P["g2"]
        if lam.dim() == 1:
            s = lam.sum()
            q = lam @ h2
            parts = [(g1 * lam[:, None]).T @ X, g1.T @ lam, (g2 * lam[:, None]).T @ h1, g2.T @ lam,
                     torch.stack([-q, q]), torch.stack([-s, s])]
            return torch.cat([p.reshape(-1) for p in parts])
        s = lam.sum(1)
        q = lam @ h2
        parts = [torch.einsum("mn,nj,nk->mjk", lam, g1, X), lam @ g1, torch.einsum("mn,nj,nk->mjk", lam, g2, h1),
                 lam @ g2, torch.stack([-q, q], 1), torch.stack([-s, s], 1)]
        return torch.cat([p.reshape(lam.shape[0], -1) for p in parts], 1)

    def gram(self, P):
        X, h1, h2, g1, g2 = P["X"], P["h1"], P["h2"], P["g1"], P["g2"]
        return (g1 @ g1.T) * (X @ X.T + 1) + (g2 @ g2.T) * (h1 @ h1.T + 1) + 2 * (h2 @ h2.T + 1)


def flat_params(net, dtype=torch.float64):
    return torch.cat([p.detach().reshape(-1) for p in net.parameters()]).to(dtype)


def lam_max(K, iters=30):
    v = torch.ones(K.shape[0], dtype=K.dtype, device=K.device) / math.sqrt(K.shape[0])
    lam = 0.0
    for _ in range(iters):
        w = K @ v
        lam = float(v @ w)
        v = w / w.norm()
    return lam


class AnchorLA:
    """Linear algebra of the tangent kernel K = J J^T (J = d delta(X_u) / dw) at an anchor w, by an
    eigendecomposition K = U diag(lam) U^T, so that for any mu > 0 and c > 0
        (mu I + c J^T J)^-1 h = (h - J^T U diag(1 / (lam + mu / c)) U^T J h) / mu        (Woodbury).
    With mu = 1 this is the soft projector onto the directions that leave delta(X_u) unchanged."""

    def __init__(self, f, Xu, w):
        self.f = f
        self.P = f.parts(w.detach(), Xu)
        lam, self.U = torch.linalg.eigh(f.gram(self.P))
        self.lam = lam.clamp_min(0)
        self.lam_max = self.lam[-1].item()

    def apply(self, h, mu, c):
        Jh = self.f.jvp(self.P, h)
        coef = (Jh @ self.U) / (self.lam + mu / c)
        return (h - self.f.vjp(self.P, coef @ self.U.T)) / mu


def lab_loss(f, w, Xl, yl):
    """Mean CE on the labeled points; w (d,) -> scalar, (M, d) -> (M,)."""
    z = f.logits(w, Xl)
    if w.dim() == 1:
        return F.cross_entropy(z, yl)
    return F.cross_entropy(z.reshape(-1, 2), yl.repeat(w.shape[0]), reduction="none").view(w.shape[0], -1).mean(1)


def lab_loss_grad(f, w, Xl, yl):
    with torch.enable_grad():
        w = w.detach().requires_grad_(True)
        L = lab_loss(f, w, Xl, yl)
        return L.item(), torch.autograd.grad(L, w)[0]


def cosine(a, b):
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-300)).item()


# ----------------------------------------------------------------------------------------
# TR-PBPL
# ----------------------------------------------------------------------------------------
class CurLA:
    """(mu I + c J_y^T J_y)^-1 at the current point y, by a Cholesky factor of K_y + (mu / c) I (Woodbury)."""

    def __init__(self, f, Xu, y, c):
        self.f, self.c = f, c
        self.P = f.parts(y.detach(), Xu)
        self.K = f.gram(self.P)
        self.eye = torch.eye(self.K.shape[0], dtype=self.K.dtype, device=self.K.device)
        self.mu, self.L = None, None

    def set_mu(self, mu):
        if mu != self.mu:
            self.mu = mu
            if self.c > 0:
                self.L = torch.linalg.cholesky(self.K + (mu / self.c) * self.eye)

    def apply(self, h):
        if self.c == 0:  # no fibre penalty (beta = 0): M = mu I
            return h / self.mu
        Jh = self.f.jvp(self.P, h)
        return (h - self.f.vjp(self.P, torch.cholesky_solve(Jh[:, None], self.L)[:, 0])) / self.mu


def pess(f, Xu, Xl, yl, theta, r, beta, warm=None, starts=("warm", "zero"), n_random=0, seed=0, T=60, tol=1e-6,
         LA=None):
    """V(theta) = max_{||eps|| <= r} Phi(eps),  Phi(eps) = L_l(theta + eps) - beta D_theta(theta + eps), the
    maximiser and the Danskin gradient d/dtheta Phi(theta, eps*) at fixed eps* (the ball constrains eps, not
    theta; the path through the anchor's logits in D is included).

    Starts: "warm" (eps of the previous call: a persistent adversary), "zero" (eps = 0, from which the first
    steps follow the first-order maximiser), and n_random random soft-tangent directions of norm r / 3 (for
    maxima driven by curvature).  Each start runs up to T Levenberg-Marquardt steps with the Jacobian at the
    current point,
        s = M^-1 (grad Phi - 2 nu eps),   M = mu I + (2 beta / N) J_y^T J_y,
    nu = 0 inside the ball and, on the sphere, such that <eps, s> = 0; eps <- radial projection of eps + s,
    accepted if Phi increases (mu / 3), else mu * 4.  The stiff directions of the penalty get a Gauss-Newton
    step, the free ones a gradient step of length 1 / mu.  Zeros of s are exactly the KKT points; the
    relative residual kkt (|grad Phi - 2 nu eps| / |grad Phi| on the sphere, |grad Phi| / |grad L_l(theta)|
    inside) is logged and stops the iteration below tol.  The best start is returned.  LA is unused (kept for
    the interface)."""
    N = Xu.shape[0]
    c = 2 * beta / N
    theta = theta.detach()
    target = f.delta(theta, Xu)
    L0, g0 = lab_loss_grad(f, theta, Xl, yl)
    g0n = max(g0.norm().item(), 1e-300)

    def phi(e):
        y = theta + e
        return lab_loss(f, y, Xl, yl) - beta * (f.delta(y, Xu) - target).pow(2).mean()

    def phi_grad(e):
        with torch.enable_grad():
            e = e.detach().requires_grad_(True)
            v = phi(e)
            return v.item(), torch.autograd.grad(v, e)[0]

    def to_ball(e):
        n = e.norm()
        return e * (r / n) if n > r else e

    cand = []
    if "warm" in starts and warm is not None:
        cand.append(("warm", to_ball(warm.detach().to(theta))))
    if "zero" in starts:
        cand.append(("zero", torch.zeros_like(theta)))
    if n_random:
        la0 = CurLA(f, Xu, theta, c)
        la0.set_mu(1.0)
        gen = torch.Generator().manual_seed(seed)
        for j in range(n_random):
            u = la0.apply(torch.randn(f.d, generator=gen, dtype=torch.float64).to(theta))
            cand.append((f"rand{j}", (r / 3) * u / u.norm()))
    best, n_chol = None, 0
    for name, e in cand:
        val, g = phi_grad(e)
        mu = max(g.norm().item(), 1e-12) / (0.1 * r)
        kkt, it = float("nan"), 0
        for it in range(1, T + 1):
            on = e.norm().item() >= r * (1 - 1e-9)
            if on:
                nu_e = max(0.0, (g @ e).item() / (2 * (e @ e).item()))
                kkt = (g - 2 * nu_e * e).norm().item() / max(g.norm().item(), 1e-300)
            else:
                kkt = g.norm().item() / g0n
            if kkt <= tol:
                break
            la = CurLA(f, Xu, theta + e, c)
            moved = False
            for _ in range(25):
                la.set_mu(mu)
                n_chol += 1
                Mg = la.apply(g)
                if on:
                    Me = la.apply(e)
                    nu = max(0.0, (e @ Mg).item() / (2 * (e @ Me).item()))
                    s = Mg - 2 * nu * Me
                else:
                    s = Mg
                e_new = to_ball(e + s)
                v_new = phi(e_new).item()
                if v_new > val:
                    e, moved = e_new, True
                    val, g = phi_grad(e)
                    mu = max(mu / 3, 1e-14)
                    break
                mu *= 4
            if not moved:
                break
        if best is None or val > best["V"]:
            best = {"V": val, "eps": e, "start": name, "iters": it, "kkt": kkt}
    e = best["eps"]
    with torch.enable_grad():
        th = theta.clone().requires_grad_(True)
        y = th + e
        Phi = lab_loss(f, y, Xl, yl) - beta * (f.delta(y, Xu) - f.delta(th, Xu)).pow(2).mean()
        v = torch.autograd.grad(Phi, th)[0]
    with torch.no_grad():
        y = theta + e
        res = f.delta(y, Xu) - target
        _, g_fixed = lab_loss_grad(f, y, Xl, yl)
        info = {"V": best["V"], "Ll_adv": lab_loss(f, y, Xl, yl).item(), "Ll_anchor": L0, "eps_norm": e.norm().item(),
                "rms_dlogit": res.pow(2).mean().sqrt().item(), "max_dlogit": res.abs().max().item(),
                "max_dp": (torch.sigmoid(f.delta(y, Xu)) - torch.sigmoid(target)).abs().max().item(),
                "kkt": best["kkt"], "iters": best["iters"], "start": best["start"], "n_steps": n_chol,
                "cos_v_fixed": cosine(v, g_fixed), "cos_v_opt": cosine(v, g0), "v_norm": v.norm().item(),
                "gopt_norm": g0.norm().item()}
    return v, e, info


# ----------------------------------------------------------------------------------------
# TR-EBPL
# ----------------------------------------------------------------------------------------
def unit_directions(d, M, seed, dtype=torch.float64, device="cpu"):
    """M unit directions (M even: antithetic pairs), uniform on the sphere of R^d."""
    g = torch.Generator().manual_seed(seed)
    xi = torch.randn(M // 2, d, generator=g, dtype=torch.float64)
    xi = xi / xi.norm(dim=1, keepdim=True)
    return torch.cat([xi, -xi]).to(dtype=dtype, device=device)


def expv(f, Xu, Xl, yl, theta, r, beta, xis, nc=8, need_grad=True):
    """E(theta) = mean_j L_l(y_j(theta)) and its gradient (autograd, the fibre moving with theta):
        u_j = r S_theta xi_j / |S_theta xi_j|,   S_theta = (I + (2 beta / N) J_theta^T J_theta)^-1
              (random direction of norm r in the soft tangent space at theta; antithetic pairs),
        y_j = argmin_y beta D_theta(y) + 1/2 |y - theta - u_j|^2   (proximal retraction onto the fibre),
    by nc damped Gauss-Newton steps y <- y - t (I + (2 beta / N) J_theta^T J_theta)^-1 G, G the gradient of
    the retraction objective, t halved until it decreases (preconditioner detached: it does not change the
    fixed point nor, at convergence, its derivative)."""
    N = Xu.shape[0]
    c = 2 * beta / N
    with torch.enable_grad():
        th = theta.detach().clone().requires_grad_(need_grad)
        P = f.parts(th, Xu)
        K = f.gram(P)
        eye = torch.eye(N, dtype=K.dtype, device=K.device)
        Lc = torch.linalg.cholesky(K + eye / c)
        Jx = f.jvp(P, xis)
        u = xis - f.vjp(P, torch.cholesky_solve(Jx.T, Lc).T)
        u = r * u / u.norm(dim=1, keepdim=True)
        z0 = th + u
        target = f.delta(th, Xu)
        Pd = {k: v.detach() for k, v in P.items()}
        Ld = Lc.detach()

        def S(h):
            return h - f.vjp(Pd, torch.cholesky_solve(f.jvp(Pd, h).T, Ld).T)

        def q(y):
            return beta * (f.delta(y, Xu) - target).pow(2).mean(1) + 0.5 * (y - z0).pow(2).sum(1)

        y = z0
        qs = [q(y).detach()]
        for _ in range(nc):
            yk = y if need_grad else y.detach().requires_grad_(True)
            qk = q(yk)
            G = torch.autograd.grad(qk.sum(), yk, create_graph=need_grad)[0]
            step = S(G)
            t = torch.ones(xis.shape[0], 1, dtype=y.dtype, device=y.device)
            q_old = qk.detach()
            for _ in range(20):
                q_new = q((yk - t * step).detach())
                bad = (q_new > q_old)[:, None]
                if not bad.any():
                    break
                t = torch.where(bad, 0.5 * t, t)
            y = yk - t * step
            if not need_grad:
                y = y.detach()
            qs.append(q(y).detach())
        losses = lab_loss(f, y, Xl, yl)
        E = losses.mean()
        v = torch.autograd.grad(E, th)[0] if need_grad else None
    with torch.no_grad():
        yd = y.detach()
        res = f.delta(yd, Xu) - target.detach()
        L0, g0 = lab_loss_grad(f, theta, Xl, yl)
    with torch.enable_grad():
        yy = yd.clone().requires_grad_(True)
        Gend = torch.autograd.grad(q(yy).sum(), yy)[0]
    info = {"V": E.item(), "Ll_anchor": L0, "loss_sd": losses.std().item(), "Ll_max_sample": losses.max().item(),
            "rms_dlogit": res.pow(2).mean(1).sqrt().mean().item(), "max_dlogit": res.abs().max().item(),
            "eps_norm": (yd - theta).norm(dim=1).mean().item(), "fp_res": Gend.norm(dim=1).max().item(),
            "q_drop": (qs[0] - qs[-1]).max().item(), "q_last_drop": (qs[-2] - qs[-1]).abs().max().item()}
    if v is not None:
        info.update({"cos_v_opt": cosine(v, g0), "v_norm": v.norm().item(), "gopt_norm": g0.norm().item()})
    return v, yd - theta, info


# ----------------------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------------------
def checks(betas=(1e3, 1e4, 1e5), radii=(0.3, 1.0, 3.0)):
    from toy_two_moons_meta import make_data, mlp, tensors
    from torch.func import functional_call, jacrev, vmap

    torch.set_default_dtype(torch.float64)
    f = FlatMLP()
    Xu, yu, Xt, yt, Xl, yl = make_data(1, 1)
    Xu_, _, Xl_ = [a.double() for a in tensors(Xu, Xt, Xl)]
    yl_ = torch.tensor(yl)
    torch.manual_seed(1000)
    net = mlp().double()
    w = flat_params(net)
    names = [n for n, _ in net.named_parameters()]
    z_ref = functional_call(net, dict(zip(names, f.split(w))), (Xu_,))
    print(f"[forward] max |logits - nn.Sequential| = {(f.logits(w, Xu_) - z_ref).abs().max().item():.1e}")
    J = vmap(jacrev(lambda ww, x: f.delta(ww, x[None])[0]), in_dims=(None, 0))(w, Xu_)
    P = f.parts(w, Xu_)
    v = torch.randn(f.d)
    lam = torch.randn(len(Xu_))
    print(f"[jvp] rel err {((f.jvp(P, v) - J @ v).norm() / (J @ v).norm()).item():.1e}   "
          f"[vjp] rel err {((f.vjp(P, lam) - J.T @ lam).norm() / (J.T @ lam).norm()).item():.1e}   "
          f"[gram] rel err {((f.gram(P) - J @ J.T).norm() / (J @ J.T).norm()).item():.1e}")
    V2, L2 = torch.randn(3, f.d), torch.randn(3, len(Xu_))
    print(f"[batched] jvp {(f.jvp(P, V2) - V2 @ J.T).abs().max().item():.1e}, vjp {(f.vjp(P, L2) - L2 @ J).abs().max().item():.1e}, "
          f"logits {(f.logits(V2, Xu_)[1] - f.logits(V2[1], Xu_)).abs().max().item():.1e}")
    c = 2 * 1e4 / len(Xu_)
    LA = AnchorLA(f, Xu_, w)
    h = torch.randn(f.d)
    for mu in (1.0, 1e-3):
        ref = torch.linalg.solve(mu * torch.eye(f.d) + c * J.T @ J, h)
        print(f"[(mu I + c J^T J)^-1, mu={mu:g}] rel err {((LA.apply(h, mu, c) - ref).norm() / ref.norm()).item():.1e}")
    truth = F.one_hot(torch.tensor(yu), 2).double()

    def train(w0, t, steps, lr=0.1):
        w_ = w0.clone()
        for _ in range(steps):
            w_ = w_.detach().requires_grad_(True)
            Lu = -(t * F.log_softmax(f.logits(w_, Xu_), 1)).sum(1).mean()
            w_ = w_ - lr * torch.autograd.grad(Lu, w_)[0]
        return w_.detach()

    anchors = {"init": w, "20 steps on uniform t": train(w, torch.full_like(truth, 0.5), 20),
               "1000 steps on truth": train(w, truth, 1000)}
    gen = torch.Generator().manual_seed(5)
    xis = unit_directions(f.d, 16, seed=7)
    hh = 1e-5
    for name, th in anchors.items():
        L0, g0 = lab_loss_grad(f, th, Xl_, yl_)
        LA = AnchorLA(f, Xu_, th)
        print(f"\n=== anchor: {name}; L_l = {L0:.3e}, |grad L_l| = {g0.norm():.2e}, lambda_max(K) = {LA.lam_max:.2e}")
        for beta in betas:
            for r in radii:
                t0 = time.time()
                v, e, info = pess(f, Xu_, Xl_, yl_, th, r, beta, n_random=2)
                tp = time.time() - t0
                per = []
                for st in ("zero", "rand"):
                    _, _, ii = pess(f, Xu_, Xl_, yl_, th, r, beta, starts=(st,) if st == "zero" else (), n_random=int(st == "rand"),
                                    seed=11)
                    per.append(f"{st} {ii['V'] - L0:+.3e} (kkt {ii['kkt']:.0e})")
                dd = torch.randn(f.d, generator=gen)
                dd = dd / dd.norm()
                vp = pess(f, Xu_, Xl_, yl_, th + hh * dd, r, beta, warm=e, starts=("warm",))[2]["V"]
                vm = pess(f, Xu_, Xl_, yl_, th - hh * dd, r, beta, warm=e, starts=("warm",))[2]["V"]
                fd = (vp - vm) / (2 * hh)
                gfix = lab_loss_grad(f, th + e, Xl_, yl_)[1]
                print(f"  pess beta={beta:.0e} r={r:g}: V-L_l {info['V'] - L0:+.3e}  L_l(adv)-L_l {info['Ll_adv'] - L0:+.3e}  "
                      f"|eps| {info['eps_norm']:.3f}  rms dlogit {info['rms_dlogit']:.1e} (max {info['max_dlogit']:.1e}, max dp {info['max_dp']:.1e})  "
                      f"kkt {info['kkt']:.1e} it {info['iters']} ({info['start']}, {info['n_steps']} trial steps)  cos(v,opt) {info['cos_v_opt']:.3f} "
                      f"cos(v,fixed) {info['cos_v_fixed']:.3f}  {tp:.1f}s")
                print(f"      single starts: {'; '.join(per)}")
                print(f"      FD {fd:+.6e}  Danskin {(v @ dd).item():+.6e} (rel {abs(fd - (v @ dd).item()) / max(abs(fd), 1e-300):.1e})  "
                      f"fixed-eps {(gfix @ dd).item():+.6e}  opt {(g0 @ dd).item():+.6e}")
                t0 = time.time()
                v, e, info = expv(f, Xu_, Xl_, yl_, th, r, beta, xis)
                te = time.time() - t0
                Ep = expv(f, Xu_, Xl_, yl_, th + hh * dd, r, beta, xis, need_grad=False)[2]["V"]
                Em = expv(f, Xu_, Xl_, yl_, th - hh * dd, r, beta, xis, need_grad=False)[2]["V"]
                fdv = (Ep - Em) / (2 * hh)
                print(f"  exp  beta={beta:.0e} r={r:g}: E-L_l {info['V'] - L0:+.3e}  sd {info['loss_sd']:.1e}  |eps| {info['eps_norm']:.3f}  "
                      f"rms dlogit {info['rms_dlogit']:.1e} (max {info['max_dlogit']:.1e})  retraction: grad {info['fp_res']:.1e}, "
                      f"q drop {info['q_drop']:.1e}, last {info['q_last_drop']:.1e}  cos(v,opt) {info['cos_v_opt']:.3f}  "
                      f"FD {fdv:+.6e} autograd {(v @ dd).item():+.6e} (rel {abs(fdv - (v @ dd).item()) / max(abs(fdv), 1e-300):.1e})  {te:.1f}s")
    # end to end: F(t) = Phi(theta_K(t)) through K unrolled student steps, gradient in the pseudo-label logits
    print("\n=== end to end through K = 5 unrolled student steps (soft pseudo-labels t = softmax(z))")
    K_, eta = 5, 0.1
    z = 0.3 * torch.randn(len(yu), 2, generator=gen)

    def unroll(z_, create):
        t = F.softmax(z_, 1)
        th = w.clone().requires_grad_(True) if create else w.clone()
        for _ in range(K_):
            with torch.enable_grad():
                th_ = th if create else th.detach().requires_grad_(True)
                Lu = -(t * F.log_softmax(f.logits(th_, Xu_), 1)).sum(1).mean()
                gth = torch.autograd.grad(Lu, th_, create_graph=create)[0]
            th = th_ - eta * gth
        return th

    for mode in ("pess", "exp"):
        for r in radii[:2]:
            zz = z.clone().requires_grad_(True)
            thK = unroll(zz, True)
            if mode == "pess":
                v, e, _ = pess(f, Xu_, Xl_, yl_, thK.detach(), r, 1e4, n_random=2)
                val = lambda z_: pess(f, Xu_, Xl_, yl_, unroll(z_, False).detach(), r, 1e4, warm=e, starts=("warm",))[2]["V"]  # noqa: E731
            else:
                v, e, _ = expv(f, Xu_, Xl_, yl_, thK.detach(), r, 1e4, xis)
                val = lambda z_: expv(f, Xu_, Xl_, yl_, unroll(z_, False).detach(), r, 1e4, xis, need_grad=False)[2]["V"]  # noqa: E731
            gz = torch.autograd.grad(thK, zz, grad_outputs=v)[0]
            dz = torch.randn(z.shape, generator=gen)
            dz = dz / dz.norm()
            fdz = (val(z + 1e-5 * dz) - val(z - 1e-5 * dz)) / 2e-5
            an = (gz * dz).sum().item()
            print(f"  {mode} r={r:g}: FD {fdz:+.6e}  back-propagated {an:+.6e}  (rel {abs(fdz - an) / max(abs(fdz), 1e-300):.1e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--betas", default="1e3,1e4,1e5")
    ap.add_argument("--radii", default="0.3,1,3")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    if a.check:
        torch.set_num_threads(a.threads)
        checks([float(b) for b in a.betas.split(",")], [float(r) for r in a.radii.split(",")])
