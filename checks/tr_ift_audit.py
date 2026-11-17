"""Audit of the TR-PBPL solver along the IFT mirror-descent runs of toy_two_moons_tr.py (Sec. 14 protocol).

At IFT the run solves V(theta) = max_eps Phi(eps) warm-started from the previous adversary with T_warm = 10
Levenberg-Marquardt steps, and its KKT residual stays high (median 2.2).  This script reruns the IFT part
of job14 for a few students and, at every meta call, also solves the same problem much harder (warm + zero
+ 4 random starts, 400 steps, tol 1e-8) without touching the run's adversary.  It records

    V_run - L_l, V_long - L_l, cos(v_run, v_long), cos(v_run, grad L_l), cos(v_long, grad L_l), kkt,

so that the question "is TR-PBPL ~ MPL at IFT because the solver stops early?" has a direct answer.
With --strong 1 it instead runs the whole IFT trajectory with the strong solver (T_cold = T_warm = 200,
warm + zero starts at every call, 2 random starts on cold calls) and records the pseudo-label accuracies,
to compare with the stored run of the same students.

    sbatch slurm_tools/tr_ift_audit.sbatch      (both runs on the A5000 node, then:)
    python checks/tr_ift_audit.py --summary checks/tr_ift_audit.json checks/tr_ift_strong.json
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import toy_two_moons_tr as TR  # noqa: E402
from tr_fiber import cosine, lab_loss_grad, pess  # noqa: E402

AUDIT = []
_orig_call = TR.Meta.__call__


def audited_call(self, theta, cold=False, need_grad=True, T=None, refresh=1):
    v, val, info = _orig_call(self, theta, cold=cold, need_grad=need_grad, T=T, refresh=refresh)
    if self.mode == "pess" and self.cfg.get("audit"):
        th = theta.detach().to(self.dev, torch.float64)
        t0 = time.time()
        L0, g0 = lab_loss_grad(self.f, th, self.Xl, self.yl)
        v2, _, i2 = pess(self.f, self.Xu, self.Xl, self.yl, th, self.cfg["radius"], self.cfg["fbeta"], warm=self.eps,
                         starts=("warm", "zero"), n_random=4, seed=10_000 + self.n_eval, T=400, tol=1e-8)
        AUDIT.append({"call": self.n_eval, "cold": bool(cold), "Ll": L0, "V_run": val, "V_long": i2["V"],
                      "kkt_run": info.get("kkt"), "kkt_long": i2["kkt"], "start_long": i2["start"],
                      "iters_long": i2["iters"], "eps_run": info.get("eps_norm"), "eps_long": i2["eps_norm"],
                      "cos_run_long": cosine(v.to(v2), v2), "cos_run_opt": cosine(v.to(g0), g0),
                      "cos_long_opt": cosine(v2, g0), "seconds": time.time() - t0})
    return v, val, info


TR.Meta.__call__ = audited_call


def job(spec):
    AUDIT.clear()
    out = TR.job14(spec)
    ift = out.get("IFT", {})
    keep = {k: ift.get(k) for k in ("pl_acc_static", "pl_acc_opt", "pl_acc_from_truth", "F_opt", "F_true", "Ll_opt")}
    keep["stu_opt"] = ift.get("train_opt", {}).get("stu_acc")
    return {"n_l": spec["n_l"], "lab_draw": spec["lab_draw"], "init": spec["init"], "ift": keep, "audit": list(AUDIT),
            "seconds": out.get("seconds"), "tr_seconds": out.get("tr_seconds"), "n_eval": out.get("n_eval")}


def summary(path_audit, path_strong, stored_dir):
    """Print the audit table, and the strong-solver IFT results next to the run and MPL for the same students."""
    import glob

    import numpy as np
    q = lambda x, p: float(np.percentile(x, p))  # noqa: E731
    A = json.load(open(path_audit))
    for n_l in (1, 5):
        E = [e for r in A["runs"] if r["n_l"] == n_l for e in r["audit"]]
        if not E:
            continue
        g = lambda k: np.array([e[k] for e in E], float)  # noqa: E731
        Ll, Vr, Vl = g("Ll"), g("V_run"), g("V_long")
        rel_r, rel_l = (Vr - Ll) / Ll, (Vl - Ll) / Ll
        warm = ~g("cold").astype(bool)
        print(f"n_l={n_l}: {len(E)} meta calls ({warm.sum()} warm) over {len(set((r['lab_draw'], r['init']) for r in A['runs'] if r['n_l'] == n_l))} students")
        print(f"  (V - L_l) / L_l   run median {np.median(rel_r):+.2e} | long median {np.median(rel_l):+.2e}; "
              f"share V_long > V_run (rel 1e-6): {np.mean(Vl > Vr * (1 + 1e-6)):.2f}; (V_run - L_l) / (V_long - L_l) "
              f"median {np.median((Vr - Ll) / np.maximum(Vl - Ll, 1e-300)):.2f}")
        print(f"  kkt              run median {np.nanmedian(g('kkt_run')):.1e} | long median {np.nanmedian(g('kkt_long')):.1e}")
        for nm, x in (("cos(v_run, v_long)", g("cos_run_long")), ("cos(v_run, grad L_l)", g("cos_run_opt")),
                      ("cos(v_long, grad L_l)", g("cos_long_opt"))):
            print(f"  {nm:22s} median {np.median(x):.4f} | 10% {q(x, 10):.4f} | min {x.min():.4f}"
                  f" | warm calls: median {np.median(x[warm]):.4f}, 10% {q(x[warm], 10):.4f}")
    if not path_strong or not os.path.exists(path_strong):
        return
    St = {(r["n_l"], r["lab_draw"], r["init"]): r["ift"] for r in json.load(open(path_strong))["runs"]}
    Au = {(r["n_l"], r["lab_draw"], r["init"]): r["ift"] for r in A["runs"]}
    stored = {}
    for tag in ("opt", "pess_r1_b10000"):
        for p in glob.glob(os.path.join(stored_dir, f"sec14_{tag}_s*.json")):
            for r in json.load(open(p))["runs"]:
                if "IFT" in r:
                    stored.setdefault(tag, {})[(r["n_l"], r["lab_draw"], r["init"])] = r["IFT"]
    print("\nIFT, same students: PL acc opt from uniform t / student on it")
    print("  student (n_l, draw, init) | MPL (cluster) | TR-PBPL run (cluster) | TR-PBPL run (audit rerun) | TR-PBPL strong solver")
    for k in sorted(St):
        cells = []
        for src in (stored.get("opt", {}).get(k), stored.get("pess_r1_b10000", {}).get(k), Au.get(k), St[k]):
            if src is None:
                cells.append("   -    ")
                continue
            stu = src["train_opt"]["stu_acc"] if "train_opt" in src else src.get("stu_opt")
            cells.append(f"{src['pl_acc_opt']:.3f} / {stu:.3f}")
        print(f"  {str(k):26s}| " + " | ".join(cells))
    # paired differences in PL acc (points) over these students
    import numpy as np
    M = stored.get("opt", {})
    Rn = stored.get("pess_r1_b10000", {})
    ks = [k for k in sorted(St) if k in M and k in Rn]
    d_run = np.array([100 * (Rn[k]["pl_acc_opt"] - M[k]["pl_acc_opt"]) for k in ks])
    d_str = np.array([100 * (St[k]["pl_acc_opt"] - M[k]["pl_acc_opt"]) for k in ks])
    print(f"  PL acc - MPL (points), {len(ks)} students: run {d_run.mean():+.2f} (mean |d| {np.abs(d_run).mean():.2f}) | "
          f"strong solver {d_str.mean():+.2f} (mean |d| {np.abs(d_str).mean():.2f}); strong closer to MPL than run for "
          f"{int((np.abs(d_str) < np.abs(d_run)).sum())}/{len(ks)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", nargs=2, metavar=("AUDIT", "STRONG"), help="print the tables of two finished runs")
    ap.add_argument("--stored", default="results/toy-two-moons-tr")
    ap.add_argument("--strong", type=int, default=0)
    ap.add_argument("--students", default="1:0:0,1:1:4,1:2:0,1:3:4,5:0:0,5:1:4,5:2:0,5:3:4",
                    help="n_l:lab_draw:init, students present in the stored rerun shards")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--tr_device", default="cpu", help="device of the trust-region linear algebra (cuda on the A5000 node)")
    ap.add_argument("--M", type=int, default=30, help="mirror-descent steps (30 in the protocol)")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.summary:
        return summary(*a.summary, a.stored)
    cfg = {"mode": "pess", "radius": 1.0, "fbeta": 1e4, "M_exp": 16, "nc": 8, "n_random": 1, "T_cold": 40,
           "T_warm": 10, "tr_device": a.tr_device, "warm_only": True, "audit": not a.strong}
    if a.strong:
        cfg.update(n_random=2, T_cold=200, T_warm=200, warm_only=False)
    specs = []
    for s in a.students.split(","):
        n_l, d, i = (int(x) for x in s.split(":"))
        specs.append(dict(cfg, n_l=n_l, lab_draw=d, init=i, eta=0.1, M=a.M, beta=0.3, beta_md=0.3, steps=1000, Ks=[],
                          ift=True, keep_pl=False))
    t0 = time.time()
    with mp.get_context("spawn").Pool(min(a.procs, len(specs))) as pool:
        runs = pool.map(job, specs)
    json.dump({"cfg": cfg, "minutes": (time.time() - t0) / 60, "runs": runs}, open(a.out, "w"))
    print(f"wrote {a.out} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
