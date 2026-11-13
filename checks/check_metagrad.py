"""Numerical check: the JVP-based teacher loss used in ssl_fmnist.py has the same
gradient in phi as the exact meta-objective  phi -> L_l(theta'(phi) + eps)
with theta'(phi) = theta - eta * grad_theta mean_i CE(t_phi(x_i), f_theta(x_i)),
eps held fixed (Danskin).  float64, CPU, BatchNorm in batch-statistics mode."""
import os
import sys

import torch
import torch.nn.functional as F
from torch.func import functional_call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import ssl_fmnist as S  # noqa: E402

torch.manual_seed(0)
torch.set_default_dtype(torch.float64)


class A:  # minimal args for Feedback
    feedback = "opt"
    radius = 0.5
    beta = 10.0
    alpha = 0.5
    k_proj = 2
    eps_reset = 0


def run(mode):
    A.feedback = mode
    teacher, student = S.ConvNet(4).double(), S.ConvNet(4).double()
    xu = torch.randn(6, 1, 28, 28)
    xl = torch.randn(5, 1, 28, 28)
    yl = torch.randint(0, 10, (5,))
    eta = 0.7
    fb = S.Feedback(A, student)
    if mode == "pess":  # give the adversary a non-trivial state
        fb.eps = {k: 0.05 * torch.randn_like(v) for k, v in S.pdict(student).items()}
    eps_fixed = None if mode == "opt" else {k: v.clone() for k, v in fb.eps.items()}

    # --- implementation (as in ssl_fmnist.main) -----------------------------
    zw = teacher(xu)
    t = F.softmax(zw, 1)
    with S.bn_frozen_stats(student):
        s_logits = student(xu)
    l_stu = S.soft_ce(s_logits, t.detach()).mean()
    g_stu = S.grads_of(l_stu, dict(student.named_parameters()))
    theta = S.pdict(student)
    theta_p = {n: theta[n] - eta * g_stu[n] for n in theta}
    c, _ = fb.feedback(theta, theta_p, xu, xl, yl)
    l_meta = (t * (c * eta)).sum(1).mean()
    g_impl = torch.autograd.grad(l_meta, list(teacher.parameters()))

    # --- exact meta-gradient by double backprop -----------------------------
    zw2 = teacher(xu)
    t2 = F.softmax(zw2, 1)
    th = {n: p.detach().clone().requires_grad_(True) for n, p in student.named_parameters()}
    with S.bn_frozen_stats(student):
        l_stu2 = S.soft_ce(functional_call(student, th, (xu,)), t2).mean()
    g2 = torch.autograd.grad(l_stu2, list(th.values()), create_graph=True)
    th_p = {n: th[n] - eta * g for n, g in zip(th, g2)}
    if eps_fixed is not None:
        th_p = {n: th_p[n] + eps_fixed[n] for n in th_p}
    with S.bn_frozen_stats(student):
        l_l = F.cross_entropy(functional_call(student, th_p, (xl,)), yl)
    g_exact = torch.autograd.grad(l_l, list(teacher.parameters()))

    num = sum(((a - b) ** 2).sum() for a, b in zip(g_impl, g_exact)).sqrt()
    den = sum((b ** 2).sum() for b in g_exact).sqrt()
    print(f"{mode:5s}  ||g_impl - g_exact|| / ||g_exact|| = {(num / den).item():.2e}   (||g_exact|| = {den.item():.3e})")


if __name__ == "__main__":
    for m in ["opt", "pess"]:
        run(m)
