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
AMP = False          # fp16 autocast inside every network forward (set in main, CUDA only)
GRAD_SCALE = 1024.0  # static loss scale for torch.autograd.grad calls under AMP


def amp_ctx():
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=AMP)


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


_GEO = {"rotate": 6, "shear_x": 8, "shear_y": 9, "translate_x": 11, "translate_y": 12}
_COLOR = [(1, _op_autocontrast), (2, _op_brightness), (3, _op_contrast), (4, _op_equalize),
          (5, _op_posterize), (7, _op_sharpness), (10, _op_solarize)]


def rand_augment(x, n_ops=2, p_apply=0.5):
    """RandAugment as in FixMatch (2 ops, each applied w.p. 0.5, magnitude in {0.1..0.9}).
    Fully batched without host synchronisation: all geometric ops of a layer are one
    per-sample affine warp; colour ops are evaluated on the batch and selected."""
    B, dev = len(x), x.device
    for _ in range(n_ops):
        op_id = torch.randint(0, len(RA_OPS), (B,), device=dev)
        op_id = torch.where(torch.rand(B, device=dev) < p_apply, op_id, torch.zeros_like(op_id))
        m = torch.randint(1, 10, (B,), device=dev).float() / 10.0
        sgn = _rand_sign(B, dev)
        t = _theta_eye(B, dev)
        ang = sgn * 30.0 * m * math.pi / 180.0
        is_rot = op_id == _GEO["rotate"]
        t[:, 0, 0] = torch.where(is_rot, torch.cos(ang), t[:, 0, 0])
        t[:, 0, 1] = torch.where(is_rot, -torch.sin(ang), t[:, 0, 1])
        t[:, 1, 0] = torch.where(is_rot, torch.sin(ang), t[:, 1, 0])
        t[:, 1, 1] = torch.where(is_rot, torch.cos(ang), t[:, 1, 1])
        t[:, 0, 1] = torch.where(op_id == _GEO["shear_x"], sgn * 0.3 * m, t[:, 0, 1])
        t[:, 1, 0] = torch.where(op_id == _GEO["shear_y"], sgn * 0.3 * m, t[:, 1, 0])
        t[:, 0, 2] = torch.where(op_id == _GEO["translate_x"], sgn * 0.6 * m, t[:, 0, 2])
        t[:, 1, 2] = torch.where(op_id == _GEO["translate_y"], sgn * 0.6 * m, t[:, 1, 2])
        is_geo = is_rot | (op_id == _GEO["shear_x"]) | (op_id == _GEO["shear_y"]) | \
            (op_id == _GEO["translate_x"]) | (op_id == _GEO["translate_y"])
        x = torch.where(is_geo[:, None, None, None], _affine(x, t), x)
        for k, op in _COLOR:
            x = torch.where((op_id == k)[:, None, None, None], op(x, m), x)
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
        with amp_ctx():
            h = self.features(x)
            h = F.adaptive_avg_pool2d(h, 1).flatten(1)
            out = self.fc(h)
        return out.float() if out.dtype == torch.float16 else out


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
    sc = GRAD_SCALE if AMP else 1.0
    g = torch.autograd.grad(loss * sc, list(params_dict.values()), allow_unused=True)
    return {k: (gi / sc if gi is not None else torch.zeros_like(v)) for (k, v), gi in zip(params_dict.items(), g)}


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
        counts = torch.zeros(NUM_CLASSES + 1, device=self.selected.device).scatter_add_(
            0, self.selected + 1, torch.ones_like(self.selected, dtype=torch.float))
        unused, cls = counts[0], counts[1:]
        denom = torch.maximum(cls.max(), unused)  # threshold warm-up
        beta = cls / denom.clamp_min(1)
        return self.tau * beta / (2.0 - beta)

    def mask_and_update(self, conf, pl, idx):
        th = self.thresholds()
        mask = (conf >= th[pl]).float()
        sel = conf >= self.tau
        self.selected[idx] = torch.where(sel, pl, self.selected[idx])
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

    def feedback(self, theta, theta_p, xs_in, xl, yl, diag=False, w=None, mode=None):
        """theta: student params before its step, theta_p: after one SGD step on
        the pseudo-labels.  Returns c (B_u x C) and diagnostics.  With diag=True
        the optimistic (MPL) feedback at the same anchor is also computed and
        compared (labeled loss of the anchor, ||c_opt||, cos(c, c_opt))."""
        a = self.a
        st = {}
        mode = mode or a.feedback
        self.w = w
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
            eps = self._sample_fiber_noise(theta_p, xs_in, st, diag)
        else:
            raise ValueError(mode)

        if mode == "pess":
            # persistent adversary: one normalised ascent step on L_l, then K retraction
            # steps back to the fiber of the anchor (descent on the KL to the anchor's
            # predictions on the student's inputs), then projection on the ball B(0, r).
            p = leaf(dadd(theta_p, eps))
            g = grads_of(F.cross_entropy(self._f(p, xl), yl), p)
            new = dadd(eps, g, a.alpha * a.radius / (dnorm(g) + 1e-12))
            if a.pess_solver == "penalty":  # (first version, kept for ablation)
                with torch.no_grad():
                    anchor = F.softmax(self._f(theta_p, xs_in), 1)
                q = leaf(dadd(theta_p, eps))
                gD = grads_of(kl_anchor(anchor, self._f(q, xs_in)).mean(), q)
                new = dadd(new, gD, -a.beta * a.alpha * a.radius / (dnorm(g) + 1e-12))
            else:
                new = self._retract(theta_p, new, xs_in, a.k_proj, st, diag)
            en = dnorm(new)
            self.eps = dscale(new, torch.clamp(a.radius / (en + 1e-12), max=1.0))
            eps = self.eps
            if diag:
                st["eps_norm"] = min(en.item(), a.radius)

        p = leaf(dadd(theta_p, eps))
        loss_l = F.cross_entropy(self._f(p, xl), yl)
        v = grads_of(loss_l, p)

        # meta feedback through the student's one-step update: JVP at theta
        _, c = jvp(lambda q: self._f(q, xs_in), (theta,), (v,))
        if diag:
            st["lab_loss_adv"] = loss_l.item()
            st["v_norm"] = dnorm(v).item()
            st["c_norm"] = c.norm().item() / math.sqrt(c.shape[0])
            if mode == "opt":
                st["lab_loss_anchor"] = loss_l.item()
            else:
                p0 = leaf(theta_p)
                l0 = F.cross_entropy(self._f(p0, xl), yl)
                v0 = grads_of(l0, p0)
                _, c0 = jvp(lambda q: self._f(q, xs_in), (theta,), (v0,))
                cc, c0c = c - c.mean(1, keepdim=True), c0 - c0.mean(1, keepdim=True)
                if self.w is not None:
                    cc, c0c = cc * self.w[:, None], c0c * self.w[:, None]
                st["lab_loss_anchor"] = l0.item()
                st["c_opt_norm"] = c0.norm().item() / math.sqrt(c0.shape[0])
                st["cos_c_copt"] = ((cc * c0c).sum() / (cc.norm() * c0c.norm() + 1e-12)).item()
        self.stats = st
        return c.detach(), st

    def _wmean(self, x):
        w = getattr(self, "w", None)
        return x.mean() if w is None else (x * w).sum() / w.sum().clamp_min(1.0)

    def _retract(self, theta_p, eps, xs_in, K, st, diag=False):
        """K descent steps on D(eps) = KL(p_anchor(x) || p_{anchor+eps}(x)) over the
        student's inputs, with the Rayleigh / Barzilai-Borwein step 2D/||grad D||^2
        (exact for one eigen-direction of the quadratic model 0.5 eps^T G eps)."""
        with torch.no_grad():
            anchor = F.softmax(self._f(theta_p, xs_in), 1)
        for _ in range(K):
            p = leaf(dadd(theta_p, eps))
            D = self._wmean(kl_anchor(anchor, self._f(p, xs_in)))
            gD = grads_of(D, p)
            step = (2.0 * D.detach()) / (ddot(gD, gD) + 1e-20)
            eps = {k: eps[k] - step * gD[k] for k in eps}
            if diag:
                st["fiber_kl_before"] = D.item()
        if diag:
            with torch.no_grad():
                st["fiber_kl"] = self._wmean(kl_anchor(anchor, self._f(dadd(theta_p, eps), xs_in))).item()
        return eps

    def _sample_fiber_noise(self, theta_p, xs_in, st, diag=False):
        """Random admissible student: isotropic perturbation of norm r, re-fitted to the
        anchor's predictions (fiber) by K retraction steps (law of R(theta + xi))."""
        a = self.a
        xi = {k: torch.randn_like(v) for k, v in theta_p.items()}
        xi = dscale(xi, a.radius / (dnorm(xi) + 1e-12))
        xi = self._retract(theta_p, xi, xs_in, a.k_proj, st, diag)
        if diag:
            st["eps_norm"] = dnorm(xi).item()
        return xi


# ----------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------
def check_amp(args, dev):
    """Compare the meta feedback computed with and without fp16 autocast on one batch."""
    global AMP
    xtr, ytr, _, _ = load_fmnist(args.data_dir)
    X = torch.from_numpy(xtr[:2000]).to(dev)
    Y = torch.from_numpy(ytr[:2000]).to(dev)
    torch.manual_seed(0)
    teacher, student = ConvNet(args.width).to(dev), ConvNet(args.width).to(dev)
    opt = make_sgd(teacher, 0.03, 5e-4)
    for _ in range(200):  # make the networks non-trivial
        i = torch.randint(0, 2000, (128,), device=dev)
        loss = F.cross_entropy(teacher(weak_aug(X[i])), Y[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
    student.load_state_dict(teacher.state_dict())
    xl, yl = weak_aug(X[:64]), Y[:64]
    xs = strong_aug(X[64:512])
    res = {}
    for amp in [False, True]:
        AMP = amp
        fb = Feedback(args, student)
        torch.manual_seed(1)
        t = F.softmax(teacher(xs), 1).detach()
        l_stu = soft_ce(student(xs), t).mean()
        g = grads_of(l_stu, dict(student.named_parameters()))
        theta = pdict(student)
        theta_p = {n: theta[n] - 0.03 * g[n] for n in theta}
        c, _ = fb.feedback(theta, theta_p, xs, xl, yl)
        res[amp] = (c, g)
    c0, g0 = res[False]
    c1, g1 = res[True]
    rel_c = ((c1 - c0).norm() / c0.norm()).item()
    rel_g = (dnorm(dadd(g1, g0, -1.0)) / dnorm(g0)).item()
    cos = (F.cosine_similarity((c1 - c1.mean(1, keepdim=True)).flatten(), (c0 - c0.mean(1, keepdim=True)).flatten(), 0)).item()
    print(json.dumps({"check_amp": args.feedback, "rel_err_c": rel_c, "cos_c": cos, "rel_err_grad_student": rel_g}), flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{args.name}.json"), "w") as f:
        json.dump({"args": vars(args), "final": {"rel_err_c": rel_c, "cos_c": cos, "rel_err_grad_student": rel_g}, "hist": []}, f)


def ft_copy(src, Xl, Yl, lr, iters, bl, wd, bn="train", ema=0.0):
    """Finetune a copy of `src` on the labeled loss only (weak views, nesterov SGD, cosine lr,
    fp16 as in the main training).  bn = "frozen" keeps every BatchNorm layer in eval mode (the
    running statistics of the main training are used and kept); "train" re-estimates them on the
    labeled batches.  Returns the finetuned copy and the EMA of its trajectory (None if ema = 0)."""
    m = copy.deepcopy(src)
    for p in m.parameters():
        p.requires_grad_(True)
    m.train()
    if bn == "frozen":
        for mod in m.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                mod.eval()
    m_ema = EMA(m, ema) if ema > 0 else None
    opt = make_sgd(m, lr, wd)
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    for it in range(iters):
        set_lr(opt, cosine_lr(lr, it, iters))
        il = torch.randint(0, len(Xl), (bl,), device=Xl.device)
        loss = F.cross_entropy(m(weak_aug(Xl[il])), Yl[il])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if m_ema is not None:
            m_ema.update(m)
    return m, m_ema


def finetune_students(pool, Xl, Yl, Xv, Yv, Xte, Yte, args):
    """Diagnostic: finetune a copy of each student's EMA on the labeled set (weak views, cosine
    lr) and report test / val accuracy of the members and of their ensemble, before and after."""
    def probs(m, X, bs=2500):
        m.eval()
        with torch.no_grad():
            p = torch.cat([F.softmax(m(clean(X[i:i + bs])), 1) for i in range(0, len(X), bs)])
        return p

    res = {"before": {}, "after": {}}
    models = [ej.model for (sj, oj, ej, fj) in pool]
    for tag in ("before", "after"):
        if tag == "after":
            models = [ft_copy(m, Xl, Yl, args.stu_ft_lr, args.stu_ft_iters, args.bl, args.wd)[0] for m in models]
        pt = [probs(m, Xte) for m in models]
        pv = [probs(m, Xv) for m in models]
        res[tag] = {"test": [(p.argmax(1) == Yte).float().mean().item() for p in pt],
                    "val": [(p.argmax(1) == Yv).float().mean().item() for p in pv],
                    "ens_test": ((sum(pt) / len(pt)).argmax(1) == Yte).float().mean().item(),
                    "ens_val": ((sum(pv) / len(pv)).argmax(1) == Yv).float().mean().item()}
    return res


def finetune_deployed(models, Xl, Yl, Xv, Yv, Xte, Yte, args):
    """End-of-run labeled finetune applied identically to every deployed EMA model (the teacher of
    every method, and the students of the two-network methods), so that a finetuned student is
    compared with finetuned baselines.  For each model and each (lr, BN mode) of the grid: test /
    val accuracy of the finetuned weights and of the EMA of the finetune trajectory; the analysis
    selects the configuration on val.  Every model and configuration sees the same random batches."""
    t0 = time.time()
    lrs = [float(x) for x in args.ft_lrs.split(",") if x]
    bns = [x for x in args.ft_bn.split(",") if x]
    res = {}
    for name, src in models.items():
        r = {"before": {"test": evaluate(src, Xte, Yte)[0], "val": evaluate(src, Xv, Yv)[0]}}
        for lr in lrs:
            for bn in bns:
                torch.manual_seed(1_000_003 + args.seed)
                m, me = ft_copy(src, Xl, Yl, lr, args.ft_iters, args.bl, args.wd, bn, args.ft_ema)
                c = {"test": evaluate(m, Xte, Yte)[0], "val": evaluate(m, Xv, Yv)[0]}
                if me is not None:
                    c.update(test_ema=evaluate(me.model, Xte, Yte)[0], val_ema=evaluate(me.model, Xv, Yv)[0])
                r[f"lr{lr:g}_{bn}"] = c
        res[name] = r
    res["minutes"] = (time.time() - t0) / 60
    return res


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
    ap.add_argument("--base", default="fixmatch",
                    choices=["sup", "fixmatch", "flexmatch", "pseudolabel", "meanteacher"])
    ap.add_argument("--lam_u", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.95)
    # bilevel pseudo-labeling regularizer
    ap.add_argument("--bilevel", default="none", choices=["none", "two", "self"])
    ap.add_argument("--feedback", default="pess", choices=["opt", "pess", "exp", "sam"])
    ap.add_argument("--lam_m", type=float, default=0.3)
    ap.add_argument("--meta_norm", type=int, default=2, help="0 exact scale, 1 per-batch RMS, 2 running RMS")
    ap.add_argument("--meta_rms_decay", type=float, default=0.99)
    ap.add_argument("--stu_excl_lab", type=int, default=1, help="student path never sees labeled images")
    ap.add_argument("--meta_warmup", type=int, default=0, help="iters before L_meta is switched on")
    ap.add_argument("--T_s", type=float, default=1.0, help="temperature of teacher soft pseudo-labels")
    ap.add_argument("--stu_view", default="strong", choices=["strong", "weak", "clean"])
    ap.add_argument("--stu_lr", type=float, default=-1.0)
    ap.add_argument("--stu_bs", type=int, default=0, help="unlabeled batch of the student path (0 = all)")
    # student pool (two-network): k held-out students trained on the same pseudo-labels; the
    # optimistic feedback of each member is aggregated by the members' labeled loss after their
    # step: min = optimistic over the pool, mean = expected, max = pessimistic, cvar = mean of
    # the pool_m worst members
    ap.add_argument("--n_stu", type=int, default=1, help="number of held-out students (two-network)")
    ap.add_argument("--pool_agg", default="mean", choices=["min", "mean", "max", "cvar"])
    ap.add_argument("--pool_m", type=int, default=2, help="cvar: number of worst members averaged")
    ap.add_argument("--pool_aug", default="same", choices=["same", "indep"],
                    help="members see the same strong view of X_u, or independent strong views")
    ap.add_argument("--pool_bag", type=float, default=1.0,
                    help="fraction of X_u each member trains on (fixed random subset per member)")
    ap.add_argument("--stu_ft_iters", type=int, default=0,
                    help="after training, finetune each student (EMA) on the labeled set and evaluate "
                         "the members and their ensemble (diagnostic; the teacher is the deployed model)")
    ap.add_argument("--stu_ft_lr", type=float, default=0.01)
    ap.add_argument("--ft_iters", type=int, default=0,
                    help="after training, finetune copies of the deployed EMA models (teacher and "
                         "students) on the labeled loss only; every (lr, BN mode) of the grid is evaluated")
    ap.add_argument("--ft_lrs", default="0.001,0.003,0.01", help="finetune learning rates (grid)")
    ap.add_argument("--ft_bn", default="frozen,train",
                    help="finetune BatchNorm modes: frozen keeps the running statistics of the main "
                         "training, train re-estimates them on the labeled batches")
    ap.add_argument("--ft_ema", type=float, default=0.99, help="EMA decay over the finetune trajectory")
    ap.add_argument("--radius", type=float, default=0.3)
    ap.add_argument("--beta", type=float, default=10.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--k_proj", type=int, default=2, help="retraction steps (pess and exp)")
    ap.add_argument("--pess_solver", default="retract", choices=["retract", "penalty"])
    ap.add_argument("--eps_reset", type=int, default=0)
    # one-network labeled-loss regularizers (controls): SAM / fiber-SAM (teacher-anchored PBPL)
    ap.add_argument("--lab_reg", default="none", choices=["none", "sam", "fibersam", "samfull"],
                    help="sam: SAM on L_l only; samfull: SAM on L_l + lam_u L_u (one perturbation)")
    ap.add_argument("--pl_src", default="point", choices=["point", "ema", "adv"],
                    help="hypothesis the pseudo-labels are read from: the live network (point), the "
                         "EMA network (ema), or the SAM-perturbed network (adv, needs --lab_reg sam/samfull)")
    ap.add_argument("--lab_radius", type=float, default=-1.0,
                    help="radius of --lab_reg (default: --radius); lets SAM on L_l run with the "
                         "bilevel feedback at its own trust-region radius")
    # diagnostics / degeneracy experiment
    ap.add_argument("--warm_sup_iters", type=int, default=0,
                    help="supervised warm start of the teacher; student copied from it")
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=200)
    ap.add_argument("--max_minutes", type=float, default=1e9)
    ap.add_argument("--eval_max", type=int, default=0, help="subsample eval sets (smoke tests)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", type=int, default=1, help="fp16 autocast (CUDA only)")
    ap.add_argument("--channels_last", type=int, default=1)
    ap.add_argument("--check_amp", type=int, default=0)
    args = ap.parse_args(argv)
    global AMP
    AMP = bool(args.amp) and args.device.startswith("cuda") and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=AMP)
    if args.check_amp:
        return check_amp(args, torch.device(args.device))
    if args.lab_reg == "samfull":
        assert args.base in ("fixmatch", "pseudolabel") and args.bilevel == "none", \
            "--lab_reg samfull: FixMatch or Pseudo-Label base, no bilevel term"
    if args.base == "sup":
        assert args.lab_reg in ("none", "sam") and args.pl_src == "point", "sup: only --lab_reg sam"
    if args.pl_src != "point":
        assert args.base in ("fixmatch", "flexmatch", "pseudolabel"), "--pl_src needs a pseudo-label base"
        assert args.pl_src != "adv" or args.lab_reg in ("sam", "samfull"), "--pl_src adv needs a SAM perturbation"

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

    mf = torch.channels_last if args.channels_last else torch.contiguous_format
    teacher = ConvNet(args.width).to(dev).to(memory_format=mf)
    t_opt = make_sgd(teacher, args.lr, args.wd)
    t_ema = EMA(teacher, args.ema)
    student = s_opt = s_ema = None
    if args.bilevel == "two":
        student = ConvNet(args.width).to(dev).to(memory_format=mf)
        s_opt = make_sgd(student, args.stu_lr if args.stu_lr > 0 else args.lr, args.wd)
        s_ema = EMA(student, args.ema)
    fb = None
    if args.bilevel == "two":
        fb = Feedback(args, student)
    elif args.bilevel == "self":
        fb = Feedback(args, teacher)
    # student pool: member 0 is `student`; members differ by their random initialisation and,
    # optionally, by their strong views and a fixed subset of X_u
    pool, bag = [], None
    if args.bilevel == "two":
        pool.append((student, s_opt, s_ema, fb))
        for _ in range(1, args.n_stu):
            sj = ConvNet(args.width).to(dev).to(memory_format=mf)
            pool.append((sj, make_sgd(sj, args.stu_lr if args.stu_lr > 0 else args.lr, args.wd),
                         EMA(sj, args.ema), Feedback(args, sj)))
    if args.n_stu > 1:
        assert args.bilevel == "two" and args.warm_sup_iters == 0, "student pool: two-network, no warm start"
        if args.pool_bag < 1.0:
            gb = torch.Generator().manual_seed(10_000 + args.seed)
            bag = (torch.rand(args.n_stu, nu, generator=gb) < args.pool_bag).float().to(dev)
    pool_prev_worst = [-1]
    flex = FlexState(nu, dev, args.tau) if args.base == "flexmatch" else None
    # labeled examples are part of the unlabeled pool (standard for FixMatch); the
    # student path excludes them so that its labeled loss is a held-out signal and the
    # fiber does not pin the labeled predictions
    is_lab_u = torch.zeros(nu, dtype=torch.bool, device=dev)
    if args.stu_excl_lab:
        is_lab_u[torch.searchsorted(unl_t, lab_t)] = True
    meta_rms = None
    lab_eps = None  # persistent eps for fibersam
    lab_fb = Feedback(args, teacher) if args.lab_reg == "fibersam" else None
    lab_r = args.lab_radius if args.lab_radius > 0 else args.radius  # radius of --lab_reg
    teacher0 = None

    # optional supervised warm start (degeneracy diagnostic)
    if args.warm_sup_iters > 0:
        for k in range(args.warm_sup_iters):
            set_lr(t_opt, args.lr)
            il = torch.randint(0, nl, (args.bl,), device=dev)
            loss = F.cross_entropy(teacher(weak_aug(Xl[il])), Yl[il])
            t_opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(t_opt)
            scaler.update()
        t_ema = EMA(teacher, args.ema)
        if student is not None:
            student.load_state_dict(teacher.state_dict())
            s_ema = EMA(student, args.ema)
            s_opt = make_sgd(student, args.stu_lr if args.stu_lr > 0 else args.lr, args.wd)
        t_opt = make_sgd(teacher, args.lr, args.wd)
        teacher0 = copy.deepcopy(teacher).eval()  # reference supervised-only model

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
        if teacher0 is not None:  # degeneracy diagnostic: how far did the teacher move?
            _, pred_now = evaluate(teacher, Xu_ev, Yu_ev)
            _, pred_0 = evaluate(teacher0, Xu_ev, Yu_ev)
            rec["pl_changed"] = (pred_now != pred_0).float().mean().item()
            rec["unl_raw"] = (pred_now == Yu_ev).float().mean().item()
            rec["drift"] = math.sqrt(sum(((p - p0) ** 2).sum().item() for p, p0 in
                                         zip(teacher.parameters(), teacher0.parameters())))
        if student is not None:
            rec["stu_test_ema"], _ = evaluate(s_ema.model, Xte, Yte)
            rec["stu_test"], _ = evaluate(student, Xte, Yte)
            rec["stu_val_ema"], _ = evaluate(s_ema.model, Xv, Yv)
        if len(pool) > 1:
            pool_eval(rec)
        rec.update({k2: float(np.mean(v)) for k2, v in run_stats.items() if len(v)})
        run_stats.clear()
        hist.append(rec)
        print(json.dumps({"name": args.name, **{kk: (round(vv, 4) if isinstance(vv, float) else vv)
                                                 for kk, vv in rec.items()}}), flush=True)

    def push(key, val):
        run_stats.setdefault(key, []).append(float(val))

    def log_pl(pl_, mask_, iu_, suffix=""):
        """Mask rate and accuracy (masked / all) of the pseudo-labels pl_ of the batch iu_."""
        push("mask" + suffix, mask_.mean().item())
        push("pl_acc_masked" + suffix, ((pl_ == Yu_true[iu_]).float() * mask_).sum().item() / max(mask_.sum().item(), 1))
        push("pl_acc_all" + suffix, (pl_ == Yu_true[iu_]).float().mean().item())

    def manual_step(g):
        """Teacher step with the gradient dict g computed outside the scaler (SAM-type passes).
        GradScaler.step needs a scale, which only scaler.scale() creates: initialize it first."""
        scaler.scale(torch.zeros((), device=dev))
        t_opt.zero_grad(set_to_none=True)
        sc = scaler.get_scale() if AMP else 1.0
        for n, p in teacher.named_parameters():
            p.grad = g[n] * sc
        scaler.step(t_opt)
        scaler.update()

    def lab_sam_eps(xl, yl_):
        """SAM perturbation of the labeled loss at the current teacher, norm lab_r."""
        tp = leaf(pdict(teacher))
        with bn_frozen_stats(teacher):
            l0 = F.cross_entropy(functional_call(teacher, tp, (xl,)), yl_)
        g0 = grads_of(l0, tp)
        return dscale(g0, lab_r / (dnorm(g0) + 1e-12))

    def wcos(a, b):
        return ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()

    def pool_feedback(eta, xs_in, xu_raw, iu, nb, w_s, t_soft, xl, yl, log_now):
        """One step of every pool member on the pseudo-labels, then the optimistic feedback of
        each member at its own look-ahead (held-out labeled loss), aggregated by that loss.
        Returns the aggregated, centred and masked feedback and each member's gradient."""
        K = len(pool)
        thetas, tps, gs, xss, ws = [], [], [], [], []
        for j, (sj, oj, ej, fj) in enumerate(pool):
            set_lr(oj, eta)
            xj = xs_in if (j == 0 or args.pool_aug == "same") else strong_aug(xu_raw)[:nb]
            wj = w_s if bag is None else w_s * bag[j][iu[:nb]]
            lj = (soft_ce(sj(xj), t_soft.detach()) * wj).sum() / wj.sum().clamp_min(1.0)
            gj = grads_of(lj, dict(sj.named_parameters()))
            th = pdict(sj)
            thetas.append(th), gs.append(gj), xss.append(xj), ws.append(wj)
            tps.append({n: th[n] - eta * gj[n] for n in th})
            if log_now:
                push("stu_loss", lj.item())
        with torch.no_grad():
            Ll = torch.stack([F.cross_entropy(pool[j][3]._f(tps[j], xl), yl).float() for j in range(K)])
        order = torch.argsort(Ll).tolist()  # ascending labeled loss: order[0] best, order[-1] worst
        wts = [0.0] * K
        if args.pool_agg == "min":
            wts[order[0]] = 1.0
        elif args.pool_agg == "max":
            wts[order[-1]] = 1.0
        elif args.pool_agg == "mean":
            wts = [1.0 / K] * K
        else:
            m = min(max(args.pool_m, 1), K)
            for j in order[-m:]:
                wts[j] = 1.0 / m
        need = range(K) if log_now else [j for j in range(K) if wts[j] > 0]
        cs = {}
        for j in need:
            cj, _ = pool[j][3].feedback(thetas[j], tps[j], xss[j], xl, yl, diag=False, w=ws[j], mode="opt")
            cs[j] = (cj - cj.mean(1, keepdim=True)) * ws[j][:, None]
        c = sum(wts[j] * cs[j] for j in range(K) if wts[j] > 0)
        st = {}
        if log_now:
            ll = Ll.tolist()
            st.update(pool_ll_min=min(ll), pool_ll_mean=float(np.mean(ll)), pool_ll_max=max(ll),
                      pool_ll_spread=(max(ll) - min(ll)) / max(float(np.mean(ll)), 1e-12),
                      lab_loss_anchor=float(np.mean(ll)))
            cm = sum(cs[j] for j in range(K)) / K
            m = min(max(args.pool_m, 1), K)
            ccv = sum(cs[j] for j in order[-m:]) / m
            cmax, cmin = cs[order[-1]], cs[order[0]]
            st["pool_cos_pair"] = float(np.mean([wcos(cs[i], cs[j]) for i in range(K) for j in range(i + 1, K)]))
            st["pool_cos_max_mean"] = wcos(cmax, cm)
            st["pool_cos_min_mean"] = wcos(cmin, cm)
            st["pool_cos_max_min"] = wcos(cmax, cmin)
            st["pool_cos_cvar_mean"] = wcos(ccv, cm)
            mem = float(np.mean([cs[j].norm().item() for j in range(K)]))
            st["pool_norm_mean_rel"] = cm.norm().item() / max(mem, 1e-12)  # 1 if members agree, 1/sqrt(K) if orthogonal
            st["c_norm"] = c.norm().item() / math.sqrt(c.shape[0])
            st["pool_worst_switch"] = float(order[-1] != pool_prev_worst[0])
            pool_prev_worst[0] = order[-1]
            for j in range(K):
                st[f"pool_worst_is_{j}"] = float(order[-1] == j)
            # reference: the same member and direction v on a second strong view of the batch
            xb = strong_aug(xu_raw)[:nb]
            cb, _ = pool[0][3].feedback(thetas[0], tps[0], xb, xl, yl, diag=False, w=ws[0], mode="opt")
            cb = (cb - cb.mean(1, keepdim=True)) * ws[0][:, None]
            st["pool_cos_aug_ref"] = wcos(cs[0], cb)
        return c, st, gs

    @torch.no_grad()
    def predict_proba(model, X, bs=2500):
        model.eval()
        out = torch.cat([F.softmax(model(clean(X[i:i + bs])), 1) for i in range(0, len(X), bs)])
        model.train()
        return out

    def pool_eval(rec):
        accs, pte, pv = [], 0, []
        for (sj, oj, ej, fj) in pool:
            pr = predict_proba(ej.model, Xte)
            accs.append((pr.argmax(1) == Yte).float().mean().item())
            pte = pte + pr
            pv.append(predict_proba(ej.model, Xv).argmax(1))
        lab = [F.cross_entropy(predict_proba(ej.model, Xl).clamp_min(1e-8).log(), Yl, reduction="mean").item()
               for (sj, oj, ej, fj) in pool]
        rec["stu_test_ema"] = float(np.mean(accs))
        rec["pool_test_ema_min"], rec["pool_test_ema_max"] = min(accs), max(accs)
        rec["pool_ens_test"] = ((pte / len(pool)).argmax(1) == Yte).float().mean().item()
        rec["pool_dis_val"] = float(np.mean([(pv[i] != pv[j]).float().mean().item()
                                             for i in range(len(pv)) for j in range(i + 1, len(pv))]))
        rec["pool_lab_loss_min"], rec["pool_lab_loss_max"] = min(lab), max(lab)

    stopped_early = False
    for k in range(args.iters):
        lr_k = cosine_lr(args.lr, k, args.iters)
        set_lr(t_opt, lr_k)
        il = torch.randint(0, nl, (args.bl,), device=dev)
        xl_w, yl = weak_aug(Xl[il]), Yl[il]

        if args.base == "sup" and args.lab_reg == "sam":
            # SAM on the labeled loss without any unlabeled data (control for --lab_reg sam):
            # ascent and perturbed passes as in the SSL bases, the plain pass only updates BN stats
            pe = leaf(dadd(pdict(teacher), lab_sam_eps(xl_w, yl)))
            with bn_frozen_stats(teacher):
                l_adv = F.cross_entropy(functional_call(teacher, pe, (xl_w,)), yl)
            g_adv = grads_of(l_adv, pe)
            with torch.no_grad():
                teacher(xl_w)  # running statistics at theta, as in the plain supervised step
            manual_step(g_adv)
            t_ema.update(teacher)
            if k % args.log_every == 0:
                push("lab_loss_adv", l_adv.item())
        elif args.base == "sup":
            loss = F.cross_entropy(teacher(xl_w), yl)
            t_opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(t_opt)
            scaler.update()
            t_ema.update(teacher)
        else:
            if pu + bu > nu:
                perm_u = torch.randperm(nu, device=dev)
                pu = 0
            iu = perm_u[pu:pu + bu]
            pu += bu
            xu_raw = Xu[iu]
            xu_w = weak_aug(xu_raw)
            if args.base in ("fixmatch", "flexmatch"):
                xu_s = strong_aug(xu_raw)
            elif args.base == "meanteacher":
                xu_s = weak_aug(xu_raw)  # second, independent weak view
            else:  # pseudolabel: target and prediction on the same weak view
                xu_s = None
            parts = [xl_w, xu_w] + ([xu_s] if xu_s is not None else [])
            x_cat = torch.cat(parts).contiguous(memory_format=mf)
            logits = teacher(x_cat)
            if xu_s is None:
                zl, zw = logits.split([args.bl, bu])
                zs = zw
            else:
                zl, zw, zs = logits.split([args.bl, bu, bu])
            l_sup = F.cross_entropy(zl, yl)
            ramp = min(1.0, k / (0.4 * args.iters)) if args.base in ("meanteacher", "pseudolabel") else 1.0
            log_now = (k % args.log_every == 0)
            # SAM perturbation of the labeled loss, needed before the pseudo-labels when they are
            # read off the perturbed hypothesis (--pl_src adv with --lab_reg sam)
            eps_pre = lab_sam_eps(xl_w, yl) if (args.pl_src == "adv" and args.lab_reg == "sam") else None
            if args.base == "meanteacher":
                with torch.no_grad(), bn_frozen_stats(t_ema.model):
                    t_ema.model.train()
                    q = F.softmax(t_ema.model(xu_w), 1)
                    t_ema.model.eval()
                    conf, pl = q.max(1)
                    mask = torch.ones_like(conf)
                l_fm = ((F.softmax(zs, 1) - q) ** 2).mean()
            else:
                with torch.no_grad():
                    q = F.softmax(zw.detach(), 1)
                    conf, pl = q.max(1)
                    if args.pl_src != "point":
                        # Y_u-step of the ELBO under another q: the point pseudo-labels of the
                        # live network are kept for the logs, the used ones are read off the EMA
                        # network or the SAM-perturbed one, on the same batch (same BN statistics)
                        pl_point, mask_point = pl, (conf >= args.tau).float()
                        if args.pl_src == "ema":
                            with bn_frozen_stats(t_ema.model):
                                t_ema.model.train()
                                zw_q = t_ema.model(x_cat)[args.bl:args.bl + bu]
                                t_ema.model.eval()
                            q = F.softmax(zw_q.float(), 1)
                        elif eps_pre is not None:
                            with bn_frozen_stats(teacher):
                                zw_q = functional_call(teacher, dadd(pdict(teacher), eps_pre), (x_cat,))[args.bl:args.bl + bu]
                            q = F.softmax(zw_q.float(), 1)
                        conf, pl = q.max(1)  # samfull + adv: still the point labels (adv ones below)
                    if flex is not None:
                        mask = flex.mask_and_update(conf, pl, iu)
                    else:
                        mask = (conf >= args.tau).float()
                l_fm = (F.cross_entropy(zs, pl, reduction="none") * mask).mean()
            total = l_sup + args.lam_u * ramp * l_fm
            if log_now:
                late_pl_log = args.lab_reg == "samfull" and args.pl_src == "adv"  # logged after the ascent
                if not late_pl_log:
                    log_pl(pl, mask, iu)
                if args.pl_src != "point":
                    log_pl(pl_point, mask_point, iu, "_point")
                    if not late_pl_log:
                        push("pl_agree", (pl == pl_point).float().mean().item())
                push("ent_u", -(q * torch.log(q.clamp_min(1e-8))).sum(1).mean().item())

            # ---- bilevel pseudo-labeling regularizer --------------------------------
            if fb is not None:
                nb = args.stu_bs if 0 < args.stu_bs < bu else bu  # unlabeled examples used by the student path
                w_s = (~is_lab_u[iu[:nb]]).float()
                t_soft = F.softmax(zw[:nb] / args.T_s, 1)
                if args.stu_view == "strong":
                    xs_in = (xu_s if xu_s is not None else strong_aug(xu_raw))[:nb]
                elif args.stu_view == "weak":
                    xs_in = xu_w[:nb]
                else:
                    xs_in = clean(xu_raw[:nb])
                pool_gs = None
                if len(pool) > 1:
                    eta = cosine_lr(args.stu_lr if args.stu_lr > 0 else args.lr, k, args.iters)
                    c, st, pool_gs = pool_feedback(eta, xs_in, xu_raw, iu, nb, w_s, t_soft, xl_w, yl, log_now)
                    g_stu = pool_gs[0]
                elif args.bilevel == "two":
                    eta = cosine_lr(args.stu_lr if args.stu_lr > 0 else args.lr, k, args.iters)
                    set_lr(s_opt, eta)
                    s_logits = student(xs_in)
                    l_stu = (soft_ce(s_logits, t_soft.detach()) * w_s).sum() / w_s.sum().clamp_min(1.0)
                    sp = dict(student.named_parameters())
                    g_stu = grads_of(l_stu, sp)
                    theta = pdict(student)
                    if log_now:
                        push("stu_loss", l_stu.item())
                else:  # self: one-step look-ahead of the deployed network
                    eta = lr_k
                    theta = pdict(teacher)
                    pl_ = leaf(theta)
                    with bn_frozen_stats(teacher):
                        l_look = (soft_ce(functional_call(teacher, pl_, (xs_in,)), t_soft.detach()) * w_s).sum() \
                            / w_s.sum().clamp_min(1.0)
                    g_stu = grads_of(l_look, pl_)
                    if log_now:
                        push("stu_loss", l_look.item())
                if pool_gs is None:
                    theta_p = {n: theta[n] - eta * g_stu[n] for n in theta}
                    c, st = fb.feedback(theta, theta_p, xs_in, xl_w, yl, diag=log_now, w=w_s)
                for kk, vv in st.items():
                    push(kk, vv)
                c = (c - c.mean(1, keepdim=True)) * w_s[:, None]  # labeled examples carry no feedback
                if args.meta_norm == 1:      # per-batch RMS
                    c = c / (c.pow(2).sum() / (w_s.sum().clamp_min(1.0) * c.shape[1])).sqrt().clamp_min(1e-12)
                elif args.meta_norm == 2:    # running RMS (keeps slow changes of the feedback size)
                    rms = (c.pow(2).sum() / (w_s.sum().clamp_min(1.0) * c.shape[1])).sqrt()
                    meta_rms = rms if meta_rms is None else args.meta_rms_decay * meta_rms + (1 - args.meta_rms_decay) * rms
                    c = c / meta_rms.clamp_min(1e-12)
                else:                        # exact meta-gradient scale (eta / B per example)
                    c = c * eta
                l_meta = (t_soft * c).sum(1).mean()
                lam_m = args.lam_m if k >= args.meta_warmup else 0.0
                total = total + lam_m * l_meta
                if log_now:
                    push("meta", l_meta.item())
                    if args.bilevel == "two" and pool_gs is None:
                        with torch.no_grad():
                            push("stu_pl_agree", (s_logits.argmax(1) == pl[:nb]).float().mean().item())

            # ---- SAM on the whole likelihood L_l + lam_u L_u (sharpness bound of the ELBO) ----
            if args.lab_reg == "samfull":
                # one perturbation along the gradient of the summed loss at theta (point pseudo-
                # labels), then the gradient of the same sum at theta + eps; with --pl_src adv the
                # pseudo-labels and mask of that second pass are read off the perturbed network
                # (the alternating solution of min over Y_u of max over the ball)
                g0 = grads_of(total, dict(teacher.named_parameters()))
                g0n = dnorm(g0)
                eps = dscale(g0, lab_r / (g0n + 1e-12)) if torch.isfinite(g0n) else dzeros(g0)  # fp16 overflow: plain step
                pe = leaf(dadd(pdict(teacher), eps))
                with bn_frozen_stats(teacher):
                    zp = functional_call(teacher, pe, (x_cat,))
                zl_p, zw_p = zp[:args.bl], zp[args.bl:args.bl + bu]
                zs_p = zp[args.bl + bu:] if xu_s is not None else zw_p
                if args.pl_src == "adv":
                    with torch.no_grad():
                        q_p = F.softmax(zw_p.detach().float(), 1)
                        conf_p, pl_p = q_p.max(1)
                        mask_p = (conf_p >= args.tau).float()
                else:
                    pl_p, mask_p = pl, mask
                l_adv = F.cross_entropy(zl_p, yl) + args.lam_u * ramp * (
                    F.cross_entropy(zs_p, pl_p, reduction="none") * mask_p).mean()
                g_adv = grads_of(l_adv, pe)
                manual_step(g_adv)
                if log_now:
                    push("loss_center", total.item())
                    push("loss_adv", l_adv.item())
                    push("lab_loss_adv", F.cross_entropy(zl_p.detach().float(), yl).item())
                    if args.pl_src == "adv":
                        log_pl(pl_p, mask_p, iu)
                        push("pl_agree", (pl_p == pl).float().mean().item())
            # ---- one-network labeled-loss regularizers (controls) --------------------
            elif args.lab_reg != "none":
                if args.lab_reg == "sam":
                    eps_l = eps_pre if eps_pre is not None else lab_sam_eps(xl_w, yl)
                else:
                    if lab_eps is None:
                        lab_eps = dzeros(pdict(teacher))
                    eps_l = lab_eps
                pe = leaf(dadd(pdict(teacher), eps_l))
                with bn_frozen_stats(teacher):
                    l_adv = F.cross_entropy(functional_call(teacher, pe, (xl_w,)), yl)
                g_adv = grads_of(l_adv, pe)
                if log_now:
                    push("lab_loss_adv", l_adv.item())
                if args.lab_reg == "fibersam":
                    # teacher-anchored PBPL: same ascent-then-retract adversary, fiber of the
                    # deployed network's own predictions on the weak unlabeled views
                    new = dadd(lab_eps, g_adv, args.alpha * lab_r / (dnorm(g_adv) + 1e-12))
                    st_l = {}
                    lab_fb.w = (~is_lab_u[iu]).float()
                    new = lab_fb._retract(pdict(teacher), new, xu_w, args.k_proj, st_l, log_now)
                    en = dnorm(new)
                    lab_eps = dscale(new, torch.clamp(lab_r / (en + 1e-12), max=1.0))
                    for kk, vv in st_l.items():
                        push(kk, vv)
                # replace the labeled CE gradient by the gradient at the perturbed point
                total = total - l_sup  # labeled term handled through g_adv below
                t_opt.zero_grad(set_to_none=True)
                scaler.scale(total).backward()
                sc = scaler.get_scale() if AMP else 1.0
                for n, p in teacher.named_parameters():
                    p.grad = g_adv[n] * sc if p.grad is None else p.grad + g_adv[n] * sc
                scaler.step(t_opt)
                scaler.update()
            else:
                t_opt.zero_grad(set_to_none=True)
                scaler.scale(total).backward()
                scaler.step(t_opt)
                scaler.update()
            t_ema.update(teacher)

            if args.bilevel == "two":
                for (sj, oj, ej, fj), gj in zip(pool, pool_gs if pool_gs is not None else [g_stu]):
                    ok = torch.stack([torch.isfinite(g).all() for g in gj.values()]).all()
                    if bool(ok):
                        for n, p in sj.named_parameters():
                            p.grad = gj[n]
                        oj.step()
                    else:
                        push("stu_nonfinite", 1.0)
                    ej.update(sj)

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
    if pool and args.stu_ft_iters > 0:
        out["stu_ft"] = finetune_students(pool, Xl, Yl, Xv, Yv, Xte, Yte, args)
        print(f"[{args.name}] student finetune: {json.dumps(out['stu_ft'])}", flush=True)
    if args.ft_iters > 0:
        dep = {"teacher": t_ema.model}
        dep.update({("student" if j == 0 else f"student{j}"): ej.model for j, (sj, oj, ej, fj) in enumerate(pool)})
        out["ft"] = finetune_deployed(dep, Xl, Yl, Xv, Yv, Xte, Yte, args)
        print(f"[{args.name}] finetune ({out['ft']['minutes']:.1f} min): "
              + json.dumps({n: {c: round(v.get("test_ema", v["test"]), 4) for c, v in r.items()}
                            for n, r in out["ft"].items() if n != "minutes"}), flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, f"{args.name}.json"), "w") as f:
        json.dump(out, f)
    print(f"[{args.name}] done in {out['minutes']:.1f} min; final={json.dumps(final)}", flush=True)
    return out


if __name__ == "__main__":
    main()
