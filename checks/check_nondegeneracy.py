"""Numerical check of Proposition N (non-degeneracy of fixed points).

Idealized setting of the theory: soft pseudo-labels, full batch, teacher and
student see the same unlabeled inputs, no augmentation.  One-step bilevel
pseudo-labeling with feedback vector v:

    theta+ = theta - eta * grad_theta L_u(p_phi(X_u), theta)
    phi+   = phi   - eta_phi * (eta/N_u) J_u(phi)^T diag(p_theta)^-1 J_u(theta) v

  opt  (MPL)  : v = grad L_l(theta+)
  pess (PBPL) : v = grad L_l(theta+ + eps*),  eps* = argmax over the fiber of
                theta+ inside B(0, r)
  exp  (EBPL) : v = E grad L_l(R(theta+ + xi)), xi ~ N(0, s^2 I), R = retraction
                onto the fiber

At a supervised-only pair (w, w) with grad L_l(w) = 0 the student does not move
(theta+ = w) and
  opt : teacher gradient = 0 exactly            -> fixed point (Prop. 4)
  pess: teacher gradient = r * T (I - Pi) H u + O(r^2) != 0
  exp : teacher gradient = O(s^2) != 0
where T = (eta/N_u) J_u^T diag(p)^-1 J_u, Pi = projector onto ker J_u(w),
H = Hessian of L_l at w, u = top eigenvector of Pi H Pi.

Model: one-hidden-layer tanh MLP, C = 3 classes, soft labels on X_l so that
the labeled infimum is attained.  float64."""
import math

import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)

D_IN, HID, C = 4, 16, 3
N_L, N_U = 6, 25
ETA = 1.0


def unflat(w):
    i = 0
    W1 = w[i:i + D_IN * HID].view(HID, D_IN); i += D_IN * HID
    b1 = w[i:i + HID]; i += HID
    W2 = w[i:i + HID * C].view(C, HID); i += HID * C
    b2 = w[i:i + C]
    return W1, b1, W2, b2


NPAR = D_IN * HID + HID + HID * C + C


def logits(w, X):
    W1, b1, W2, b2 = unflat(w)
    return torch.tanh(X @ W1.T + b1) @ W2.T + b2


def probs(w, X):
    return torch.softmax(logits(w, X), 1)


def kl(P, w, X):  # mean_x KL(P(x) || p_w(x))
    logp = torch.log_softmax(logits(w, X), 1)
    return (P * (torch.log(P) - logp)).sum(1).mean()


# data: labeled soft labels from a random "true" network
w_true = torch.randn(NPAR) * 1.5
Xl = torch.randn(N_L, D_IN)
Xu = torch.randn(N_U, D_IN)
Yl_hard = probs(w_true, Xl).argmax(1)
Yl = torch.full((N_L, C), 0.1)
Yl[torch.arange(N_L), Yl_hard] = 0.8


def L_l(w):
    return kl(Yl, w, Xl)


def grad(f, w):
    w = w.detach().requires_grad_(True)
    return torch.autograd.grad(f(w), w)[0]


# 1) supervised-only solution w: grad L_l(w) = 0
w = torch.randn(NPAR) * 0.5
w.requires_grad_(True)
opt = torch.optim.LBFGS([w], lr=1, max_iter=5000, tolerance_grad=1e-14, tolerance_change=1e-16,
                        history_size=50, line_search_fn="strong_wolfe")


def closure():
    opt.zero_grad()
    l = L_l(w)
    l.backward()
    return l


for _ in range(5):
    opt.step(closure)
w = w.detach()
print(f"supervised-only w: L_l = {L_l(w).item():.3e}, ||grad L_l|| = {grad(L_l, w).norm().item():.2e}")

# 2) geometry at w
P_u = probs(w, Xu).detach()
J = torch.autograd.functional.jacobian(lambda z: probs(z, Xu).reshape(-1), w)  # (N_U*C) x d
U, S, Vh = torch.linalg.svd(J, full_matrices=True)
rank = int((S > 1e-10 * S[0]).sum())
Vk = Vh[rank:].T                      # basis of ker J_u(w)
Pi = Vk @ Vk.T
H = torch.autograd.functional.hessian(L_l, w)
M = Pi @ H @ Pi
evals, evecs = torch.linalg.eigh(M)
lam, u = evals[-1], evecs[:, -1]
Tmat = (ETA / N_U) * J.T @ torch.diag(1.0 / P_u.reshape(-1)) @ J  # teacher map v -> grad_phi
print(f"d = {NPAR}, rank J_u = {rank} (N_u(C-1) = {N_U * (C - 1)}), dim ker J_u = {NPAR - rank}")
print(f"lambda_max(Pi H Pi) = {lam.item():.3e},  ||(I-Pi) H u|| = {((torch.eye(NPAR) - Pi) @ H @ u).norm().item():.3e}")
pred = (Tmat @ (H @ u)).norm().item()
print(f"predicted first-order slope ||T H u|| = {pred:.3e}")


def retract(z, iters=60):
    """Project z onto the fiber {p_z(Xu) = P_u} by Gauss-Newton steps."""
    z = z.clone()
    for _ in range(iters):
        r = (probs(z, Xu) - P_u).reshape(-1)
        if r.norm() < 1e-13:
            break
        Jz = torch.autograd.functional.jacobian(lambda q: probs(q, Xu).reshape(-1), z, vectorize=True)
        z = z - torch.linalg.pinv(Jz, rtol=1e-10) @ r
    return z


def pess_eps(r, steps=400):
    """max L_l(w+eps) over the fiber of w inside B(0, r): projected (retracted)
    gradient ascent started from +-r u, best of the two."""
    best, best_val = None, -1e9
    for s in (+1.0, -1.0):
        z = retract(w + s * r * u)
        for t in range(steps):
            g = grad(L_l, z)
            Jz = torch.autograd.functional.jacobian(lambda q: probs(q, Xu).reshape(-1), z, vectorize=True)
            g_t = g - torch.linalg.pinv(Jz, rtol=1e-10) @ (Jz @ g)  # tangent part of the gradient
            z = retract(z + 0.05 * r * g_t / (g_t.norm() + 1e-15))
            e = z - w
            if e.norm() > r:
                z = retract(w + e * (r / e.norm()))
        val = L_l(z).item()
        if val > best_val:
            best, best_val = z, val
    return best - w, best_val


print("\n   r      ||g_T opt||    ||g_T pess||   ||g_T pess||/r   V_r - L_l(w)    fiber residual")
g_opt = Tmat @ grad(L_l, w)
for r in [0.02, 0.05, 0.1, 0.2]:
    eps, val = pess_eps(r)
    v = grad(L_l, w + eps)
    g_pess = Tmat @ v
    res = (probs(w + eps, Xu) - P_u).abs().max().item()
    print(f"{r:6.3f}   {g_opt.norm().item():.3e}     {g_pess.norm().item():.3e}     {g_pess.norm().item() / r:.3e}"
          f"      {val - L_l(w).item():.3e}      {res:.1e}")

print("\n   s      ||E g_T exp||   ||E g_T exp||/s^2   (Monte Carlo, 400 antithetic pairs)")
for s in [0.025, 0.05, 0.1]:
    acc = torch.zeros(NPAR)
    n = 400
    for _ in range(n):
        xi = torch.randn(NPAR) * s
        for sgn in (1.0, -1.0):  # antithetic pairs cancel the O(s) noise
            acc += grad(L_l, retract(w + sgn * xi))
    g_exp = Tmat @ (acc / (2 * n))
    print(f"{s:6.3f}   {g_exp.norm().item():.3e}       {g_exp.norm().item() / s ** 2:.3e}")


# ---------------------------------------------------------------------------
# Practical inner solver used in ssl_fmnist.py: persistent eps, normalized
# projected ascent on  L_l(w+eps) - beta * KL(p_w(Xu) || p_{w+eps}(Xu)).
# Does it find the fiber maximizer (direction +-u, value V_r)?
# ---------------------------------------------------------------------------
def D_u(z):
    return kl(P_u, z, Xu)


print("\npractical solver (r = 0.1, 600 steps, alpha = 0.2, from a random eps of norm 1e-3):")
print("   beta    |cos(eps,u)|   L_l(w+eps)-L_l(w)   exact V_r-L_l(w)   KL to anchor")
r = 0.1
_, vr_exact = pess_eps(r)
for beta in [10.0, 100.0, 1000.0]:
    torch.manual_seed(1)
    e = torch.randn(NPAR)
    e = 1e-3 * e / e.norm()
    for t in range(600):
        g = grad(L_l, w + e) - beta * grad(D_u, w + e)
        e = e + 0.2 * r * g / (g.norm() + 1e-15)
        if e.norm() > r:
            e = e * (r / e.norm())
    cos = abs((e @ u) / e.norm()).item()
    print(f"{beta:7.0f}     {cos:.4f}         {(L_l(w + e) - L_l(w)).item():.3e}          "
          f"{vr_exact - L_l(w).item():.3e}         {D_u(w + e).item():.1e}")
    v = grad(L_l, w + e)
    print(f"          teacher gradient ||A v|| / r = {(Tmat @ v).norm().item() / r:.3e}  (exact maximizer: see table above)")
