#!/usr/bin/env python3
"""Semi-supervised FashionMNIST: FixMatch / FlexMatch baselines and
trust-region (pessimistic / expected) bilevel pseudo-labeling (PBPL / EBPL).

Single file so that it can be shipped to a Kaggle script kernel as is.

Deployed model = "teacher" network phi, trained with
    CE on labeled data + lam_u * (FixMatch or FlexMatch) + lam_m * L_meta.
L_meta is the bilevel pseudo-labeling feedback.  It is obtained from an
auxiliary label-free *student* theta (two-network mode, `--bilevel two`) or
from a one-step look-ahead of the deployed network itself (`--bilevel self`):

  theta'   = theta - eta * grad_theta CE(t_phi(x_u), f_theta(x_u))   (one student step)
  eps      : admissible perturbation of the student, i.e. it keeps the
             student's own predictions on the unlabeled inputs (fiber of the
             anchor theta'), ||eps|| <= r
               opt  : eps = 0                                (= meta pseudo labels)
               pess : eps = persistent projected ascent on
                      L_l(theta'+eps) - beta * KL(p_theta'(x_u) || p_{theta'+eps}(x_u))
               exp  : eps = random direction on the sphere of radius r,
                      re-fitted to the fiber by K descent steps on the KL term
               sam  : eps = r * grad L_l(theta') / ||.||     (eps reset every step)
  v        = grad L_l(theta' + eps)                       (labeled gradient at
             the pessimistic / random admissible student)
  L_meta   = mean_i < t_phi(x_i) , stopgrad( J_{f_theta}(x_i) v ) >
             whose gradient in phi equals (up to the factor eta/B) the
             meta-gradient d/dphi L_l(theta'(phi) + eps).

Everything (data, augmentation) lives on the GPU.
"""
import argparse
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jvp

MEAN, STD = 0.2860, 0.3530
NUM_CLASSES = 10


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def _read_idx(path):
    with open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    arr = np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim)
    return arr.reshape(dims)


def find_data_dir(hint):
    cands = []
    if hint:
        cands.append(hint)
    cands += ["/kaggle/input", "./data", os.path.expanduser("~/data")]
    for c in cands:
        if not os.path.isdir(c):
            continue
        for root, _, files in os.walk(c):
            if "train-images-idx3-ubyte" in files and "t10k-images-idx3-ubyte" in files:
                return root
    return None


def load_fmnist(data_dir):
    d = find_data_dir(data_dir)
    if d is not None:
        xtr = _read_idx(os.path.join(d, "train-images-idx3-ubyte"))
        ytr = _read_idx(os.path.join(d, "train-labels-idx1-ubyte"))
        xte = _read_idx(os.path.join(d, "t10k-images-idx3-ubyte"))
        yte = _read_idx(os.path.join(d, "t10k-labels-idx1-ubyte"))
    else:  # fall back to torchvision download
        import torchvision
        root = data_dir or "./data"
        tr = torchvision.datasets.FashionMNIST(root, train=True, download=True)
        te = torchvision.datasets.FashionMNIST(root, train=False, download=True)
        xtr, ytr = tr.data.numpy(), tr.targets.numpy()
        xte, yte = te.data.numpy(), te.targets.numpy()
    return (np.ascontiguousarray(xtr), ytr.astype(np.int64),
            np.ascontiguousarray(xte), yte.astype(np.int64))


def make_split(ytr, n_per_class, n_val, seed):
    """Balanced labeled set, held-out validation set (never trained on),
    unlabeled pool = everything except validation (labeled included)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(ytr))
    val_idx = perm[:n_val]
    rest = perm[n_val:]
    lab = []
    for c in range(NUM_CLASSES):
        lab.extend(rest[ytr[rest] == c][:n_per_class].tolist())
    lab = np.array(sorted(lab))
    unl = np.array(sorted(rest.tolist()))
    return lab, unl, val_idx


# ----------------------------------------------------------------------------
# GPU augmentation (weak = crop/flip, strong = RandAugment(2) + Cutout)
# ----------------------------------------------------------------------------
def to_float(xb):
    return xb.float().div(255.0).unsqueeze(1)


def normalize(x):
    return (x - MEAN) / STD


def rand_crop_flip(x, pad=4, flip=True):
    B, C, H, W = x.shape
    xp = F.pad(x, (pad, pad, pad, pad))
    ox = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    oy = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    rows = oy[:, None] + torch.arange(H, device=x.device)
    cols = ox[:, None] + torch.arange(W, device=x.device)
    tmp = torch.gather(xp, 2, rows[:, None, :, None].expand(B, C, H, xp.shape[3]))
    out = torch.gather(tmp, 3, cols[:, None, None, :].expand(B, C, H, W))
    if flip:
        m = torch.rand(B, device=x.device) < 0.5
        out = torch.where(m[:, None, None, None], out.flip(3), out)
    return out


def _affine(x, theta):
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def _rand_sign(n, device):
    return torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)


def _op_identity(x, m):
    return x


def _op_autocontrast(x, m):
    lo = x.amin(dim=(1, 2, 3), keepdim=True)
    hi = x.amax(dim=(1, 2, 3), keepdim=True)
    return torch.where(hi > lo, (x - lo) / (hi - lo + 1e-8), x)


def _op_brightness(x, m):
    f = 1.0 + _rand_sign(len(x), x.device) * 0.9 * m
    return (x * f[:, None, None, None]).clamp(0, 1)


def _op_contrast(x, m):
    f = 1.0 + _rand_sign(len(x), x.device) * 0.9 * m
    mu = x.mean(dim=(1, 2, 3), keepdim=True)
    return (mu + (x - mu) * f[:, None, None, None]).clamp(0, 1)


_SMOOTH = None


def _op_sharpness(x, m):
    global _SMOOTH
    if _SMOOTH is None or _SMOOTH.device != x.device:
        k = torch.tensor([[1., 1., 1.], [1., 5., 1.], [1., 1., 1.]]) / 13.0
        _SMOOTH = k.view(1, 1, 3, 3).to(x.device)
    blur = F.conv2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), _SMOOTH)
    f = 1.0 + _rand_sign(len(x), x.device) * 0.9 * m
    return (blur + (x - blur) * f[:, None, None, None]).clamp(0, 1)


def _op_equalize(x, m):
    n = len(x)
    q = (x.view(n, -1) * 255).round().long().clamp(0, 255)
    hist = torch.zeros(n, 256, device=x.device).scatter_add_(1, q, torch.ones_like(q, dtype=torch.float))
    cdf = hist.cumsum(1)
    cdf_min = torch.where(hist > 0, cdf, torch.full_like(cdf, float("inf"))).amin(1, keepdim=True)
    tot = cdf[:, -1:]
    lut = ((cdf - cdf_min) / (tot - cdf_min).clamp_min(1)).clamp(0, 1)
    return lut.gather(1, q).view_as(x)


def _op_posterize(x, m):
    bits = (8 - (4 * m).round()).long().clamp(4, 8)
    shift = (8 - bits)[:, None, None, None]
    q = (x * 255).round().long()
    q = (q >> shift) << shift
    return q.float() / 255.0


def _op_solarize(x, m):
    th = (1.0 - m)[:, None, None, None]
    return torch.where(x < th, x, 1.0 - x)


def _theta_eye(n, device):
    t = torch.zeros(n, 2, 3, device=device)
    t[:, 0, 0] = 1
    t[:, 1, 1] = 1
    return t


def _op_rotate(x, m):
    a = _rand_sign(len(x), x.device) * 30.0 * m * math.pi / 180.0
    t = torch.zeros(len(x), 2, 3, device=x.device)
    t[:, 0, 0] = torch.cos(a)
    t[:, 0, 1] = -torch.sin(a)
    t[:, 1, 0] = torch.sin(a)
    t[:, 1, 1] = torch.cos(a)
    return _affine(x, t)


def _op_shear_x(x, m):
    t = _theta_eye(len(x), x.device)
    t[:, 0, 1] = _rand_sign(len(x), x.device) * 0.3 * m
    return _affine(x, t)


def _op_shear_y(x, m):
    t = _theta_eye(len(x), x.device)
    t[:, 1, 0] = _rand_sign(len(x), x.device) * 0.3 * m
    return _affine(x, t)


def _op_translate_x(x, m):
    t = _theta_eye(len(x), x.device)
    t[:, 0, 2] = _rand_sign(len(x), x.device) * 0.3 * m * 2.0  # normalized coords span 2
    return _affine(x, t)


def _op_translate_y(x, m):
    t = _theta_eye(len(x), x.device)
    t[:, 1, 2] = _rand_sign(len(x), x.device) * 0.3 * m * 2.0
    return _affine(x, t)


RA_OPS = [_op_identity, _op_autocontrast, _op_brightness, _op_contrast, _op_equalize,
          _op_posterize, _op_rotate, _op_sharpness, _op_shear_x, _op_shear_y,
          _op_solarize, _op_translate_x, _op_translate_y]


def rand_augment(x, n_ops=2, p_apply=0.5):
    B = len(x)
    x = x.clone()
    for _ in range(n_ops):
        op_id = torch.randint(0, len(RA_OPS), (B,), device=x.device)
        apply = torch.rand(B, device=x.device) < p_apply
        mag = torch.randint(1, 10, (B,), device=x.device).float() / 10.0
        for k, op in enumerate(RA_OPS):
            idx = torch.nonzero((op_id == k) & apply, as_tuple=False).flatten()
            if idx.numel() == 0 or k == 0:
                continue
            x[idx] = op(x[idx], mag[idx])
    return x


def cutout(x, frac=0.5, fill=0.5):
    B, C, H, W = x.shape
    s = int(frac * H)
    cy = torch.randint(0, H, (B,), device=x.device)
    cx = torch.randint(0, W, (B,), device=x.device)
    ys = torch.arange(H, device=x.device)[None, :]
    xs = torch.arange(W, device=x.device)[None, :]
    my = ((ys >= (cy[:, None] - s // 2)) & (ys < (cy[:, None] + s - s // 2)))
    mx = ((xs >= (cx[:, None] - s // 2)) & (xs < (cx[:, None] + s - s // 2)))
    mask = (my[:, :, None] & mx[:, None, :])[:, None]
    return torch.where(mask, torch.full_like(x, fill), x)


def weak_aug(xb):
    return normalize(rand_crop_flip(to_float(xb)))


def strong_aug(xb):
    x = rand_crop_flip(to_float(xb))
    x = rand_augment(x)
    x = cutout(x)
    return normalize(x)


def clean(xb):
    return normalize(to_float(xb))


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------
class ConvNet(nn.Module):
    """Small VGG-style CNN for 28x28 inputs (width w: 32 -> ~0.16M params)."""

    def __init__(self, w=32, num_classes=NUM_CLASSES):
        super().__init__()

        def block(cin, cout, k=3):
            return [nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                    nn.BatchNorm2d(cout, momentum=0.01),
                    nn.LeakyReLU(0.1)]

        layers = []
        layers += block(1, w) + block(w, w) + [nn.MaxPool2d(2)]
        layers += block(w, 2 * w) + block(2 * w, 2 * w) + [nn.MaxPool2d(2)]
        layers += block(2 * w, 4 * w) + block(4 * w, 4 * w, k=1)
        self.features = nn.Sequential(*layers)
        self.fc = nn.Linear(4 * w, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu", a=0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.features(x)
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.fc(h)


class bn_frozen_stats:
    """Use batch statistics but do not update running statistics (auxiliary passes)."""

    def __init__(self, model):
        self.bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]

    def __enter__(self):
        self.prev = [m.track_running_stats for m in self.bns]
        for m in self.bns:
            m.track_running_stats = False
        return self

    def __exit__(self, *a):
        for m, p in zip(self.bns, self.prev):
            m.track_running_stats = p


class EMA:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        self.decay = decay
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
        for be, b in zip(self.model.buffers(), model.buffers()):
            be.copy_(b)


def make_sgd(model, lr, wd, momentum=0.9):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if (p.ndim <= 1) else decay).append(p)
    return torch.optim.SGD([{"params": decay, "weight_decay": wd},
                            {"params": no_decay, "weight_decay": 0.0}],
                           lr=lr, momentum=momentum, nesterov=True)


def cosine_lr(base, k, total):
    return base * math.cos(7.0 * math.pi * k / (16.0 * total))


def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr


# ----------------------------------------------------------------------------
# helpers for parameter dicts
# ----------------------------------------------------------------------------
def pdict(model):
    return {n: p.detach() for n, p in model.named_parameters()}


def dnorm(d):
    return torch.sqrt(sum((t.float() ** 2).sum() for t in d.values()))


def dadd(a, b, s=1.0):
    return {k: a[k] + s * b[k] for k in a}


def dscale(a, s):
    return {k: a[k] * s for k in a}


def dzeros(a):
    return {k: torch.zeros_like(v) for k, v in a.items()}


def ddot(a, b):
    return sum((a[k] * b[k]).sum() for k in a)


def soft_ce(logits, target):
    return -(target * F.log_softmax(logits, 1)).sum(1)


def kl_anchor(p_anchor, logits):
    """KL(p_anchor || softmax(logits)) per row."""
    logp = F.log_softmax(logits, 1)
    return (p_anchor * (torch.log(p_anchor.clamp_min(1e-8)) - logp)).sum(1)


def grads_of(loss, params_dict):
    g = torch.autograd.grad(loss, list(params_dict.values()), allow_unused=True)
    return {k: (gi if gi is not None else torch.zeros_like(v)) for (k, v), gi in zip(params_dict.items(), g)}


def leaf(d):
    return {k: v.detach().clone().requires_grad_(True) for k, v in d.items()}


# ----------------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, X, Y, bs=2500):
    model.eval()
    correct, n = 0, 0
    preds = []
    for i in range(0, len(X), bs):
        logits = model(clean(X[i:i + bs]))
        p = logits.argmax(1)
        preds.append(p)
        correct += (p == Y[i:i + bs]).sum().item()
        n += len(p)
    model.train()
    return correct / max(n, 1), torch.cat(preds)


# ----------------------------------------------------------------------------
# FlexMatch state
# ----------------------------------------------------------------------------
class FlexState:
    def __init__(self, n_unl, device, tau):
        self.selected = torch.full((n_unl,), -1, dtype=torch.long, device=device)
        self.tau = tau
        self.n = n_unl

    def thresholds(self):
        counts = torch.bincount(self.selected + 1, minlength=NUM_CLASSES + 1).float()
        unused, cls = counts[0], counts[1:]
        denom = torch.maximum(cls.max(), unused)  # threshold warm-up
        beta = cls / denom.clamp_min(1)
        return self.tau * beta / (2.0 - beta)

    def mask_and_update(self, conf, pl, idx):
        th = self.thresholds()
        mask = (conf >= th[pl]).float()
        sel = conf >= self.tau
        if sel.any():
            self.selected[idx[sel]] = pl[sel]
        return mask


# ----------------------------------------------------------------------------
# bilevel feedback
# ----------------------------------------------------------------------------
class Feedback:
    """Computes the meta feedback c_i = J_{f_theta}(x_i) v for a student network
    (two-network mode) or for the deployed network itself (self mode)."""

    def __init__(self, args, net):
        self.a = args
        self.net = net
        self.eps = None  # persistent adversary (pess)
        self.stats = {}

    def _f(self, params, x):
        with bn_frozen_stats(self.net):
            return functional_call(self.net, params, (x,))

    def feedback(self, theta, theta_p, xs_in, xl, yl, diag=False):
        """theta: student params before its step, theta_p: after one SGD step on
        the pseudo-labels.  Returns c (B_u x C) and diagnostics.  With diag=True
        the optimistic (MPL) feedback at the same anchor is also computed and
        compared (labeled loss of the anchor, ||c_opt||, cos(c, c_opt))."""
        a = self.a
        st = {}
        mode = a.feedback
        if mode == "opt":
            eps = dzeros(theta_p)
        elif mode == "sam":
            p = leaf(theta_p)
            l0 = F.cross_entropy(self._f(p, xl), yl)
            g0 = grads_of(l0, p)
            eps = dscale(g0, a.radius / (dnorm(g0) + 1e-12))
        elif mode == "pess":
            if self.eps is None or a.eps_reset:
                self.eps = dzeros(theta_p)
            eps = self.eps
        elif mode == "exp":
            eps = self._sample_fiber_noise(theta_p, xs_in, st)
        else:
            raise ValueError(mode)

        p = leaf(dadd(theta_p, eps))
        loss_l = F.cross_entropy(self._f(p, xl), yl)
        v = grads_of(loss_l, p)
        st["lab_loss_adv"] = loss_l.item()

        if mode == "pess":
            with torch.no_grad():
                anchor = F.softmax(self._f(theta_p, xs_in), 1)
            D = kl_anchor(anchor, self._f(p, xs_in)).mean()
            gD = grads_of(D, p)
            st["fiber_kl"] = D.item()
            g = dadd(v, gD, -a.beta)
            gn = dnorm(g)
            new = dadd(self.eps, g, a.alpha * a.radius / (gn + 1e-12))
            en = dnorm(new)
            self.eps = dscale(new, min(1.0, a.radius / (en.item() + 1e-12)))
            st["eps_norm"] = en.item()

        # meta feedback through the student's one-step update: JVP at theta
        _, c = jvp(lambda q: self._f(q, xs_in), (theta,), (v,))
        st["v_norm"] = dnorm(v).item()
        st["c_norm"] = c.norm().item() / math.sqrt(c.shape[0])
        if diag:
            if mode == "opt":
                st["lab_loss_anchor"] = loss_l.item()
            else:
                p0 = leaf(theta_p)
                l0 = F.cross_entropy(self._f(p0, xl), yl)
                v0 = grads_of(l0, p0)
                _, c0 = jvp(lambda q: self._f(q, xs_in), (theta,), (v0,))
                cc, c0c = c - c.mean(1, keepdim=True), c0 - c0.mean(1, keepdim=True)
                st["lab_loss_anchor"] = l0.item()
                st["c_opt_norm"] = c0.norm().item() / math.sqrt(c0.shape[0])
                st["cos_c_copt"] = ((cc * c0c).sum() / (cc.norm() * c0c.norm() + 1e-12)).item()
        self.stats = st
        return c.detach(), st

    def _sample_fiber_noise(self, theta_p, xs_in, st):
        """Random admissible student: isotropic direction on the sphere of radius r,
        re-fitted to the anchor's predictions (fiber) by K descent steps on the
        KL to the anchor, then rescaled to radius r."""
        a = self.a
        xi = {k: torch.randn_like(v) for k, v in theta_p.items()}
        xi = dscale(xi, a.radius / (dnorm(xi) + 1e-12))
        with torch.no_grad():
            anchor = F.softmax(self._f(theta_p, xs_in), 1)
        for _ in range(a.k_proj):
            p = leaf(dadd(theta_p, xi))
            D = kl_anchor(anchor, self._f(p, xs_in)).mean()
            gD = grads_of(D, p)
            # Rayleigh-quotient (Barzilai-Borwein-like) step size for 0.5 xi^T G xi
            step = (2.0 * D.item()) / (ddot(gD, gD).item() + 1e-12)
            xi = {k: xi[k] - step * gD[k] for k in xi}
            st["fiber_kl"] = D.item()
        xi = dscale(xi, a.radius / (dnorm(xi) + 1e-12))
        return xi


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="run")
    ap.add_argument("--out_dir", default="./results")
    ap.add_argument("--data_dir", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_per_class", type=int, default=4)
    ap.add_argument("--n_val", type=int, default=5000)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--bl", type=int, default=64)
    ap.add_argument("--mu", type=int, default=7)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--base", default="fixmatch", choices=["sup", "fixmatch", "flexmatch"])
    ap.add_argument("--lam_u", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.95)
    # bilevel pseudo-labeling regularizer
    ap.add_argument("--bilevel", default="none", choices=["none", "two", "self"])
    ap.add_argument("--feedback", default="pess", choices=["opt", "pess", "exp", "sam"])
    ap.add_argument("--lam_m", type=float, default=0.3)
    ap.add_argument("--meta_norm", type=int, default=1)
    ap.add_argument("--meta_warmup", type=int, default=0, help="iters before L_meta is switched on")
    ap.add_argument("--T_s", type=float, default=1.0, help="temperature of teacher soft pseudo-labels")
    ap.add_argument("--stu_view", default="strong", choices=["strong", "weak", "clean"])
    ap.add_argument("--stu_lr", type=float, default=-1.0)
    ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--k_proj", type=int, default=2)
    ap.add_argument("--eps_reset", type=int, default=0)
    # one-network labeled-loss regularizers (controls): SAM / fiber-SAM (teacher-anchored PBPL)
    ap.add_argument("--lab_reg", default="none", choices=["none", "sam", "fibersam"])
    # diagnostics / degeneracy experiment
    ap.add_argument("--warm_sup_iters", type=int, default=0,
                    help="supervised warm start of the teacher; student copied from it")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--max_minutes", type=float, default=1e9)
    ap.add_argument("--eval_max", type=int, default=0, help="subsample eval sets (smoke tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    t_start = time.time()

    xtr, ytr, xte, yte = load_fmnist(args.data_dir)
    lab, unl, val = make_split(ytr, args.n_per_class, args.n_val, args.seed)
    Xtr = torch.from_numpy(xtr).to(dev)
    Ytr = torch.from_numpy(ytr).to(dev)
    Xte = torch.from_numpy(xte).to(dev)
    Yte = torch.from_numpy(yte).to(dev)
    lab_t = torch.from_numpy(lab).to(dev)
    unl_t = torch.from_numpy(unl).to(dev)
    val_t = torch.from_numpy(val).to(dev)
    Xl, Yl = Xtr[lab_t], Ytr[lab_t]
    Xu, Yu_true = Xtr[unl_t], Ytr[unl_t]  # true labels only for diagnostics
    Xv, Yv = Xtr[val_t], Ytr[val_t]
    nl, nu = len(Xl), len(Xu)
    bu = args.bl * args.mu
    Xu_ev, Yu_ev = Xu, Yu_true
    if args.eval_max > 0:
        Xte, Yte = Xte[:args.eval_max], Yte[:args.eval_max]
        Xv, Yv = Xv[:args.eval_max], Yv[:args.eval_max]
        Xu_ev, Yu_ev = Xu[:args.eval_max], Yu_true[:args.eval_max]

    teacher = ConvNet(args.width).to(dev).to(memory_format=torch.channels_last)
    t_opt = make_sgd(teacher, args.lr, args.wd)
    t_ema = EMA(teacher, args.ema)
    student = s_opt = s_ema = None
    if args.bilevel == "two":
        student = ConvNet(args.width).to(dev).to(memory_format=torch.channels_last)
        s_opt = make_sgd(student, args.stu_lr if args.stu_lr > 0 else args.lr, args.wd)
        s_ema = EMA(student, args.ema)
    fb = None
    if args.bilevel == "two":
        fb = Feedback(args, student)
    elif args.bilevel == "self":
        fb = Feedback(args, teacher)
    flex = FlexState(nu, dev, args.tau) if args.base == "flexmatch" else None
    lab_eps = None  # persistent eps for fibersam

    # optional supervised warm start (degeneracy diagnostic)
    if args.warm_sup_iters > 0:
        for k in range(args.warm_sup_iters):
            set_lr(t_opt, args.lr)
            il = torch.randint(0, nl, (args.bl,), device=dev)
            loss = F.cross_entropy(teacher(weak_aug(Xl[il])), Yl[il])
            t_opt.zero_grad(set_to_none=True)
            loss.backward()
            t_opt.step()
        t_ema = EMA(teacher, args.ema)
        if student is not None:
            student.load_state_dict(teacher.state_dict())
            s_ema = EMA(student, args.ema)
        t_opt = make_sgd(teacher, args.lr, args.wd)

    hist = []
    run_stats = {}
    perm_u = torch.randperm(nu, device=dev)
    pu = 0
    n_params = sum(p.numel() for p in teacher.parameters())
    print(f"[{args.name}] labeled={nl} unlabeled={nu} val={len(Xv)} params={n_params} dev={dev}", flush=True)

    def log_eval(k):
        acc_te_ema, _ = evaluate(t_ema.model, Xte, Yte)
        acc_te, _ = evaluate(teacher, Xte, Yte)
        acc_val_ema, _ = evaluate(t_ema.model, Xv, Yv)
        acc_unl_ema, pred_u = evaluate(t_ema.model, Xu_ev, Yu_ev)
        rec = {"iter": k, "test_ema": acc_te_ema, "test": acc_te, "val_ema": acc_val_ema,
               "unl_ema": acc_unl_ema, "minutes": (time.time() - t_start) / 60}
        if student is not None:
            rec["stu_test_ema"], _ = evaluate(s_ema.model, Xte, Yte)
            rec["stu_test"], _ = evaluate(student, Xte, Yte)
            rec["stu_val_ema"], _ = evaluate(s_ema.model, Xv, Yv)
        rec.update({k2: float(np.mean(v)) for k2, v in run_stats.items() if len(v)})
        run_stats.clear()
        hist.append(rec)
        print(json.dumps({"name": args.name, **{kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                                 for kk, vv in rec.items()}}), flush=True)

    def push(key, val):
        run_stats.setdefault(key, []).append(float(val))

    stopped_early = False
    for k in range(args.iters):
        lr_k = cosine_lr(args.lr, k, args.iters)
        set_lr(t_opt, lr_k)
        il = torch.randint(0, nl, (args.bl,), device=dev)
        xl_w, yl = weak_aug(Xl[il]), Yl[il]

        if args.base == "sup":
            loss = F.cross_entropy(teacher(xl_w), yl)
            t_opt.zero_grad(set_to_none=True)
            loss.backward()
            t_opt.step()
            t_ema.update(teacher)
        else:
            if pu + bu > nu:
                perm_u = torch.randperm(nu, device=dev)
                pu = 0
            iu = perm_u[pu:pu + bu]
            pu += bu
            xu_raw = Xu[iu]
            xu_w, xu_s = weak_aug(xu_raw), strong_aug(xu_raw)

            logits = teacher(torch.cat([xl_w, xu_w, xu_s]).contiguous(memory_format=torch.channels_last))
            zl, zw, zs = logits.split([args.bl, bu, bu])
            l_sup = F.cross_entropy(zl, yl)
            with torch.no_grad():
                q = F.softmax(zw.detach(), 1)
                conf, pl = q.max(1)
                if flex is not None:
                    mask = flex.mask_and_update(conf, pl, iu)
                else:
                    mask = (conf >= args.tau).float()
            l_fm = (F.cross_entropy(zs, pl, reduction="none") * mask).mean()
            total = l_sup + args.lam_u * l_fm
            push("mask", mask.mean().item())
            push("pl_acc_masked", ((pl == Yu_true[iu]).float() * mask).sum().item() / max(mask.sum().item(), 1))
            push("pl_acc_all", (pl == Yu_true[iu]).float().mean().item())
            push("ent_u", -(q * torch.log(q.clamp_min(1e-8))).sum(1).mean().item())

            # ---- bilevel pseudo-labeling regularizer --------------------------------
            if fb is not None:
                t_soft = F.softmax(zw / args.T_s, 1)
                if args.stu_view == "strong":
                    xs_in = xu_s
                elif args.stu_view == "weak":
                    xs_in = xu_w
                else:
                    xs_in = clean(xu_raw)
                if args.bilevel == "two":
                    eta = cosine_lr(args.stu_lr if args.stu_lr > 0 else args.lr, k, args.iters)
                    set_lr(s_opt, eta)
                    s_logits = student(xs_in)
                    l_stu = soft_ce(s_logits, t_soft.detach()).mean()
                    sp = dict(student.named_parameters())
                    g_stu = grads_of(l_stu, sp)
                    theta = pdict(student)
                    push("stu_loss", l_stu.item())
                else:  # self: one-step look-ahead of the deployed network
                    eta = lr_k
                    theta = pdict(teacher)
                    pl_ = leaf(theta)
                    with bn_frozen_stats(teacher):
                        l_look = soft_ce(functional_call(teacher, pl_, (xs_in,)), t_soft.detach()).mean()
                    g_stu = grads_of(l_look, pl_)
                    push("stu_loss", l_look.item())
                theta_p = {n: theta[n] - eta * g_stu[n] for n in theta}
                c, st = fb.feedback(theta, theta_p, xs_in, xl_w, yl, diag=(k % args.log_every == 0))
                for kk, vv in st.items():
                    push(kk, vv)
                if args.meta_norm:
                    c = c - c.mean(1, keepdim=True)
                    c = c / (c.pow(2).mean().sqrt() + 1e-12)
                else:
                    c = c * eta
                l_meta = (t_soft * c).sum(1).mean()
                lam_m = args.lam_m if k >= args.meta_warmup else 0.0
                total = total + lam_m * l_meta
                push("meta", l_meta.item())
                if args.bilevel == "two":
                    with torch.no_grad():
                        push("stu_pl_agree", (s_logits.argmax(1) == pl).float().mean().item())

            # ---- one-network labeled-loss regularizers (controls) --------------------
            if args.lab_reg != "none":
                if args.lab_reg == "sam":
                    tp = leaf(pdict(teacher))
                    with bn_frozen_stats(teacher):
                        l0 = F.cross_entropy(functional_call(teacher, tp, (xl_w,)), yl)
                    g0 = grads_of(l0, tp)
                    eps_l = dscale(g0, args.radius / (dnorm(g0) + 1e-12))
                else:
                    if lab_eps is None:
                        lab_eps = dzeros(pdict(teacher))
                    eps_l = lab_eps
                pe = leaf(dadd(pdict(teacher), eps_l))
                with bn_frozen_stats(teacher):
                    l_adv = F.cross_entropy(functional_call(teacher, pe, (xl_w,)), yl)
                g_adv = grads_of(l_adv, pe)
                push("lab_loss_adv", l_adv.item())
                if args.lab_reg == "fibersam":
                    with torch.no_grad(), bn_frozen_stats(teacher):
                        anchor = F.softmax(functional_call(teacher, pdict(teacher), (xu_w,)), 1)
                    with bn_frozen_stats(teacher):
                        D = kl_anchor(anchor, functional_call(teacher, pe, (xu_w,))).mean()
                    gD = grads_of(D, pe)
                    g = dadd(g_adv, gD, -args.beta)
                    new = dadd(lab_eps, g, args.alpha * args.radius / (dnorm(g) + 1e-12))
                    en = dnorm(new).item()
                    lab_eps = dscale(new, min(1.0, args.radius / (en + 1e-12)))
                    push("fiber_kl", D.item())
                # replace the labeled CE gradient by the gradient at the perturbed point
                total = total - l_sup  # labeled term handled through g_adv below
                t_opt.zero_grad(set_to_none=True)
                total.backward()
                for n, p in teacher.named_parameters():
                    p.grad = g_adv[n] if p.grad is None else p.grad + g_adv[n]
                t_opt.step()
            else:
                t_opt.zero_grad(set_to_none=True)
                total.backward()
                t_opt.step()
            t_ema.update(teacher)

            if args.bilevel == "two":
                for n, p in student.named_parameters():
                    p.grad = g_stu[n]
                s_opt.step()
                s_ema.update(student)

        if (k + 1) % args.eval_every == 0 or k + 1 == args.iters:
            log_eval(k + 1)
        if (time.time() - t_start) / 60 > args.max_minutes:
            print(f"[{args.name}] stopping early at iter {k + 1} (time budget)", flush=True)
            log_eval(k + 1)
            stopped_early = True
            break

    final = hist[-1] if hist else {}
    out = {"args": vars(args), "final": final, "hist": hist, "stopped_early": stopped_early,
           "n_params": n_params, "minutes": (time.time() - t_start) / 60}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{args.name}.json"), "w") as f:
        json.dump(out, f)
    print(f"[{args.name}] done in {out['minutes']:.1f} min; final={json.dumps(final)}", flush=True)
    return out


if __name__ == "__main__":
    main()
