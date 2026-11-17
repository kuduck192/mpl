#!/usr/bin/env python3
"""Summary, paired tables and figures of checks/toy_two_moons_tr.py (TR-PBPL / TR-EBPL under the MPL protocols).

  python checks/toy_two_moons_tr_summary.py [results/toy-two-moons-tr]

Reads every shard results/toy-two-moons-tr/<protocol>_<config>_s<i>of<n>.json, merges the shards of a
configuration and compares each trust-region configuration with MPL (mode opt, the original code of the
three toys, rerun on the same machine) student by student: mean paired difference, wins / losses (|d| >
0.001), paired t-test and Wilcoxon p.  Also checks that the MPL rerun reproduces the stored results of the
original toys (results/toy-two-moons/*.json), and prints the diagnostics of the trust-region values along the
runs: cos(grad Phi, grad L_l) at the same student, relative gap (Phi - L_l) / L_l, RMS change of the
student's logits on X_u at the adversarial / random students, KKT residual of the TR-PBPL ascent.
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_DIR = os.path.join(ROOT, "results", "toy-two-moons-tr")
OLD = os.path.join(ROOT, "results", "toy-two-moons")
FIG = os.path.join(ROOT, "docs", "figures")
KS14 = ["1", "2", "5", "10", "20", "50", "100", "IFT"]
KS15 = ["1", "2", "5", "10", "20", "50", "IFT"]
LAMS = ["0", "0.1", "1", "10"]
ORDER = ["opt", "pess_b10000_r1", "pess_b10_r1", "exp_b10000_r1", "exp_b10_r1"]
LABEL = {"opt": "MPL", "pess_b10000_r1": "TR-PBPL b=1e4", "pess_b10_r1": "TR-PBPL b=10",
         "exp_b10000_r1": "TR-EBPL b=1e4", "exp_b10_r1": "TR-EBPL b=10"}


def tag_of(cfg):
    return "opt" if cfg["mode"] == "opt" else f"{cfg['mode']}_b{cfg['fbeta']:g}_r{cfg['radius']:g}"


def load(d):
    R = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(d, "*.json"))):
        D = json.load(open(p))
        bad = [r for r in D["runs"] if "error" in r]
        if bad:
            print(f"{os.path.basename(p)}: {len(bad)} failed jobs skipped ({bad[0]['error'][:80]})")
        R[(D["meta"]["protocol"], tag_of(D["meta"]["cfg"]))] += [r for r in D["runs"] if "error" not in r]
    return R


def tags(R, proto):
    present = {t for (p, t) in R if p == proto}
    return [t for t in ORDER if t in present] + sorted(present - set(ORDER))


def lab(t):
    return LABEL.get(t, t)


def paired(a, b):
    """a, b: dict key -> value.  Mean difference a - b, wins / losses, t-test and Wilcoxon p."""
    ks = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in ks], float)
    if len(d) == 0:
        return None
    pt = stats.ttest_1samp(d, 0).pvalue if len(d) > 1 and d.std() > 0 else float("nan")
    pw = stats.wilcoxon(d).pvalue if np.any(np.abs(d) > 0) and len(d) > 5 else float("nan")
    return {"n": len(d), "mean": d.mean(), "sd": d.std(ddof=1) if len(d) > 1 else 0.0,
            "win": int((d > 0.001).sum()), "loss": int((d < -0.001).sum()), "p_t": pt, "p_w": pw}


def fmt_pair(s, scale=100):
    if s is None:
        return "-"
    return f"{scale * s['mean']:+.2f} ({s['win']}/{s['loss']}, p {s['p_w']:.2g})"


def med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def q(xs, p):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.percentile(xs, p)) if xs else float("nan")


# ----------------------------------------------------------------------------------------
# sec13
# ----------------------------------------------------------------------------------------
def sec13(R):
    print("\n" + "=" * 100 + "\nSec. 13 protocol: pseudo-labels = argmin of the first-order signal c = J_f(X_u) v")
    T = tags(R, "sec13")
    for n_l in (1, 5):
        print(f"\n--- {n_l} labeled point(s) per class (means over random students; paired differences x100 vs MPL)")
        print(f"{'config':16s} | dyn: student / PL start -> end | frozen: student / PL | statics: one student / avg of 64 | "
              f"d student dyn | d PL dyn | d student frozen | d PL frozen")
        base = {}
        for t in T:
            runs = [r for r in R[("sec13", t)] if r.get("n_l") == n_l]
            dyn = {(r["lab_draw"], r["init"]): r for r in runs if r.get("rule") == "dynamic"}
            frz = {(r["lab_draw"], r["init"]): r for r in runs if r.get("rule") == "frozen"}
            st = [r for r in runs if "single_accs" in r]
            if not dyn:
                continue
            m = {"dyn_s": {k: r["hist"][-1]["stu_acc"] for k, r in dyn.items()},
                 "dyn_p": {k: r["hist"][-1]["pl_acc"] for k, r in dyn.items()},
                 "frz_s": {k: r["hist"][-1]["stu_acc"] for k, r in frz.items()},
                 "frz_p": {k: r["hist"][-1]["pl_acc"] for k, r in frz.items()}}
            if t == "opt":
                base = m
            p0 = np.mean([r["hist"][0]["pl_acc"] for r in dyn.values()])
            single = np.mean([a for s in st for a in s["single_accs"]]) if st else float("nan")
            avg = np.mean([s["avg_acc"] for s in st]) if st else float("nan")
            line = (f"{lab(t):16s} | {np.mean(list(m['dyn_s'].values())):.3f} / {p0:.3f} -> {np.mean(list(m['dyn_p'].values())):.3f} | "
                    f"{np.mean(list(m['frz_s'].values())):.3f} / {np.mean(list(m['frz_p'].values())):.3f} | {single:.3f} / {avg:.3f}")
            if t != "opt" and base:
                line += " | " + " | ".join(fmt_pair(paired(m[k], base[k])) for k in ("dyn_s", "dyn_p", "frz_s", "frz_p"))
            print(line)
        for t in T:
            if t == "opt":
                continue
            H = [h for r in R[("sec13", t)] if r.get("n_l") == n_l and r.get("rule") == "dynamic" for h in r["hist"]
                 if "tr_cos_v_opt" in h]
            if H:
                gap = [(h["tr_V"] - h["tr_Ll_anchor"]) / max(h["tr_Ll_anchor"], 1e-300) for h in H]
                print(f"  {lab(t):16s} along dynamic runs ({len(H)} logged students): cos(v, grad L_l) median {med([h['tr_cos_v_opt'] for h in H]):.4f} "
                      f"(5% {q([h['tr_cos_v_opt'] for h in H], 5):.4f}); (Phi - L_l) / L_l median {med(gap):+.2e} (95% {q(gap, 95):+.2e}); "
                      f"rms dlogit median {med([h.get('tr_rms_dlogit') for h in H]):.1e}; kkt median {med([h.get('tr_kkt') for h in H]):.1e}")


# ----------------------------------------------------------------------------------------
# sec14
# ----------------------------------------------------------------------------------------
def rec14(r, K):
    return r["IFT"] if K == "IFT" else r["K"].get(K)


def sec14(R):
    print("\n" + "=" * 100 + "\nSec. 14 protocol: free pseudo-labels, K unrolled steps and IFT (medians over students)")
    T = tags(R, "sec14")
    for n_l in (1, 5):
        print(f"\n--- {n_l} labeled point(s) per class")
        by = {t: {(r["lab_draw"], r["init"]): r for r in R[("sec14", t)] if r["n_l"] == n_l} for t in T}
        for K in KS14:
            print(f"  K = {K}")
            base = None
            for t in T:
                S = {k: rec14(r, K) for k, r in by[t].items() if rec14(r, K) is not None}
                if not S:
                    continue
                X = list(S.values())
                m = {"pl_opt": {k: x["pl_acc_opt"] for k, x in S.items()},
                     "stu_opt": {k: x["train_opt"]["stu_acc"] for k, x in S.items()},
                     "pl_tr": {k: x["pl_acc_from_truth"] for k, x in S.items()},
                     "pl_st": {k: x["pl_acc_static"] for k, x in S.items()}}
                if t == "opt":
                    base = m
                beat = np.mean([x["F_opt"] < x["F_true"] for x in X])
                line = (f"    {lab(t):16s} PL acc static {med(m['pl_st'].values()):.3f} | opt {med(m['pl_opt'].values()):.3f} | "
                        f"from truth {med(m['pl_tr'].values()):.3f} || student on opt {med(m['stu_opt'].values()):.3f} | "
                        f"F true {med([x['F_true'] for x in X]):.2e} opt {med([x['F_opt'] for x in X]):.2e} (opt < true: {beat:.2f})")
                if t != "opt" and base:
                    line += (f" || d PL opt {fmt_pair(paired(m['pl_opt'], base['pl_opt']))}, d student {fmt_pair(paired(m['stu_opt'], base['stu_opt']))}, "
                             f"d PL from truth {fmt_pair(paired(m['pl_tr'], base['pl_tr']))}")
                print(line)
        for t in T:
            if t == "opt":
                continue
            E = [e for r in by[t].values() for K in KS14 if rec14(r, K) for key in ("trace_uniform", "trace_truth")
                 for e in rec14(r, K)[key] if "cos_v_opt" in e]
            if E:
                gap = [(e["F"] - e["Ll"]) / max(e["Ll"], 1e-300) for e in E if "Ll" in e]
                print(f"  {lab(t):16s} along all mirror-descent steps ({len(E)} students): cos(v, grad L_l) median {med([e['cos_v_opt'] for e in E]):.4f} "
                      f"(5% {q([e['cos_v_opt'] for e in E], 5):.4f}, min {min(e['cos_v_opt'] for e in E):.3f}); (Phi - L_l) / L_l median {med(gap):+.2e} "
                      f"(95% {q(gap, 95):+.2e}); rms dlogit median {med([e.get('rms_dlogit') for e in E]):.1e}; kkt median {med([e.get('kkt') for e in E]):.1e}")


# ----------------------------------------------------------------------------------------
# sec15
# ----------------------------------------------------------------------------------------
def sec15(R):
    print("\n" + "=" * 100 + "\nSec. 15 protocol: pseudo-labels anchored to a FixMatch teacher, t* = argmin <t, unit(g)> + lam KL(t || q)")
    T = tags(R, "sec15")
    for n_l in (1, 5):
        print(f"\n--- {n_l} labeled point(s) per class (means; PL acc - teacher, corrections / corruptions, student - student on q; "
              f"paired d x100 vs MPL)")
        by = {t: {(r["lab_draw"], r["init"]): r for r in R[("sec15", t)] if r["n_l"] == n_l} for t in T}
        for K in KS15:
            for lam in LAMS:
                base, cells = None, []
                for t in T:
                    S = {k: (r, r["K"][K]["lam"][lam]) for k, r in by[t].items() if K in r["K"] and lam in r["K"][K]["lam"]}
                    if not S:
                        continue
                    dpl = {k: x["pl_acc"] - r["teacher_pl_acc"] for k, (r, x) in S.items()}
                    dst = {k: x["train"]["stu_acc"] - r["train_teacher"]["stu_acc"] for k, (r, x) in S.items()}
                    pl = {k: x["pl_acc"] for k, (r, x) in S.items()}
                    st = {k: x["train"]["stu_acc"] for k, (r, x) in S.items()}
                    cell = (f"{lab(t)}: {100 * np.mean(list(dpl.values())):+.2f} ({100 * np.mean([x['corrections'] for _, x in S.values()]):.1f}/"
                            f"{100 * np.mean([x['corruptions'] for _, x in S.values()]):.1f}) stu {100 * np.mean(list(dst.values())):+.2f}")
                    if t == "opt":
                        base = (pl, st)
                    elif base:
                        cell += f" [d PL {fmt_pair(paired(pl, base[0]))}, d stu {fmt_pair(paired(st, base[1]))}]"
                    cells.append(cell)
                print(f"  K={K:>3s} lam={lam:>3s} | " + " | ".join(cells))


# ----------------------------------------------------------------------------------------
# reproduction of the stored MPL results
# ----------------------------------------------------------------------------------------
def reproduction(R):
    print("\n" + "=" * 100 + "\nMPL rerun (mode opt) vs the stored results of the original toys")
    checks = []
    if ("sec13", "opt") in R and os.path.exists(os.path.join(OLD, "results.json")):
        old = {(r["n_l"], r["lab_draw"], r["init"], r["rule"]): r for r in json.load(open(os.path.join(OLD, "results.json")))["runs"]}
        new = {(r["n_l"], r["lab_draw"], r["init"], r["rule"]): r for r in R[("sec13", "opt")] if "rule" in r}
        d = [abs(new[k]["hist"][-1]["stu_acc"] - old[k]["hist"][-1]["stu_acc"]) for k in new if k in old]
        e = [abs(new[k]["hist"][-1]["pl_acc"] - old[k]["hist"][-1]["pl_acc"]) for k in new if k in old]
        checks.append(f"sec13: {len(d)} runs, max |d student acc| {max(d):.4f}, max |d PL acc| {max(e):.4f}, identical {np.mean(np.array(d) == 0):.2f}")
    if ("sec14", "opt") in R and os.path.exists(os.path.join(OLD, "unroll.json")):
        old = {(r["n_l"], r["lab_draw"], r["init"]): r for r in json.load(open(os.path.join(OLD, "unroll.json")))["runs"]}
        d = []
        for r in R[("sec14", "opt")]:
            o = old.get((r["n_l"], r["lab_draw"], r["init"]))
            if o:
                d += [abs(rec14(r, K)["pl_acc_opt"] - rec14(o, K)["pl_acc_opt"]) for K in KS14]
        checks.append(f"sec14: {len(d)} (student, K) cells, max |d PL acc opt| {max(d):.4f}, identical {np.mean(np.array(d) == 0):.2f}")
    if ("sec15", "opt") in R and os.path.exists(os.path.join(OLD, "anchor.json")):
        old = {(r["n_l"], r["lab_draw"], r["init"]): r for r in json.load(open(os.path.join(OLD, "anchor.json")))["runs"]}
        d = []
        for r in R[("sec15", "opt")]:
            o = old.get((r["n_l"], r["lab_draw"], r["init"]))
            if o:
                d += [abs(r["K"][K]["lam"][lam]["pl_acc"] - o["K"][K]["lam"][lam]["pl_acc"]) for K in KS15 for lam in LAMS]
        checks.append(f"sec15: {len(d)} (student, K, lam) cells, max |d PL acc| {max(d):.4f}, identical {np.mean(np.array(d) == 0):.2f}")
    for c in checks:
        print("  " + c)


# ----------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------
def figures(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
    INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({
        "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6, "savefig.dpi": 200,
        "figure.facecolor": "white"})
    STY = {"opt": (SERIES[0], "-"), "pess_b10000_r1": (SERIES[1], "-"), "pess_b10_r1": (SERIES[1], (0, (4, 2))),
           "exp_b10000_r1": (SERIES[2], "-"), "exp_b10_r1": (SERIES[2], (0, (4, 2)))}
    os.makedirs(FIG, exist_ok=True)
    paths = []
    xpos = [1, 2, 5, 10, 20, 50, 100, 400]

    # ---- 1. sec14: pseudo-label and student accuracy against K ----------------------------------
    T = tags(R, "sec14")
    if T:
        fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.0))
        for row, n_l in enumerate((1, 5)):
            for j, (title, f) in enumerate((("PL acc., from uniform t", lambda x: x["pl_acc_opt"]),
                                            ("student on them (test)", lambda x: x["train_opt"]["stu_acc"]),
                                            ("PL acc., from the truth", lambda x: x["pl_acc_from_truth"]))):
                ax = axes[row, j]
                for t in T:
                    rs = [r for r in R[("sec14", t)] if r["n_l"] == n_l]
                    ys = [med([f(rec14(r, K)) for r in rs if rec14(r, K)]) for K in KS14]
                    col, ls = STY.get(t, (INK2, ":"))
                    ax.plot(xpos[:-1], ys[:-1], color=col, ls=ls, marker="o", ms=2.5)
                    ax.plot(xpos[-1:], ys[-1:], color=col, marker="o", ms=4, ls="")
                ax.set_xscale("log")
                ax.set_xticks(xpos)
                ax.set_xticklabels([str(k) for k in KS14])
                ax.minorticks_off()
                ax.axvspan(150, 600, color=GRID, alpha=0.35, lw=0)
                ax.set_xlabel("unrolled student steps K (IFT = converged)")
                ax.set_title(f"{'abcdef'[3 * row + j]}. {title}, {n_l} lab./class", fontsize=7.3, loc="left", color=INK)
        handles = [Line2D([], [], color=STY[t][0], ls=STY[t][1], marker="o", ms=3, label=lab(t)) for t in T if t in STY]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, bbox_to_anchor=(0.5, -0.01))
        fig.text(0.01, 0.045, "Medians over the 40 random students of the Sec. 14 protocol per row (5 labeled draws x 8 inits); "
                 "30 mirror-descent steps on the soft pseudo-labels against L_l (MPL) or the trust-region value (TR) of the "
                 "student.\nWhere a line is not visible it lies under another one.", fontsize=6.3, color=INK2)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        paths.append(os.path.join(FIG, "toy_tr_unroll.png"))
        fig.savefig(paths[-1], bbox_inches="tight")
        plt.close(fig)

    # ---- 2. sec15: anchored pseudo-labels, accuracy against the teacher -------------------------
    T = tags(R, "sec15")
    if T:
        fig, axes = plt.subplots(2, 3, figsize=(8.6, 5.0))
        xs = np.arange(len(LAMS) + 1)
        for row, n_l in enumerate((1, 5)):
            for j, K in enumerate(("1", "50", "IFT")):
                ax = axes[row, j]
                for t in T:
                    rs = [r for r in R[("sec15", t)] if r["n_l"] == n_l and K in r["K"]]
                    if not rs:
                        continue
                    ys = [100 * np.mean([r["K"][K]["lam"][lam]["pl_acc"] - r["teacher_pl_acc"] for r in rs]) for lam in LAMS] + [0.0]
                    col, ls = STY.get(t, (INK2, ":"))
                    ax.plot(xs, ys, color=col, ls=ls, marker="o", ms=2.5)
                ax.axhline(0, color=INK2, lw=0.8)
                ax.set_xticks(xs)
                ax.set_xticklabels(LAMS + ["inf\n(teacher)"])
                ax.set_xlabel("KL weight lam")
                ax.set_ylabel("PL acc. - teacher (points)" if j == 0 else "")
                ax.set_title(f"{'abcdef'[3 * row + j]}. K = {K}, {n_l} lab./class", fontsize=7.3, loc="left", color=INK)
        handles = [Line2D([], [], color=STY[t][0], ls=STY[t][1], marker="o", ms=3, label=lab(t)) for t in T if t in STY]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, bbox_to_anchor=(0.5, -0.01))
        n_st = len([r for r in R[("sec15", T[0])] if r["n_l"] == 1])
        fig.text(0.01, 0.045, f"Means over {n_st} random students per row (5 labeled draws x {n_st // 5} inits, one FixMatch teacher per "
                 "draw). Above 0: the meta signal corrects more teacher errors than it creates.\nWhere a line is not visible it "
                 "lies under another one.", fontsize=6.3, color=INK2)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        paths.append(os.path.join(FIG, "toy_tr_anchor.png"))
        fig.savefig(paths[-1], bbox_inches="tight")
        plt.close(fig)

    # ---- 3. pseudo-label accuracy vs student accuracy (sec14, K = 20 and IFT), tight fibre -------
    T = [t for t in ("opt", "pess_b10000_r1", "exp_b10000_r1") if ("sec14", t) in R]
    if T:
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.6), sharex=True, sharey=True)
        for row, n_l in enumerate((1, 5)):
            for j, K in enumerate(("20", "IFT")):
                ax = axes[row, j]
                ax.plot([0.38, 1], [0.38, 1], color=INK2, lw=0.8, ls=(0, (4, 3)))
                for t, mk in zip(T, ("o", "o", "o")):
                    rs = [r for r in R[("sec14", t)] if r["n_l"] == n_l and rec14(r, K)]
                    x = [rec14(r, K)["train_opt"]["stu_acc"] for r in rs]
                    y = [rec14(r, K)["pl_acc_opt"] for r in rs]
                    col = STY[t][0]
                    if t == "opt":
                        ax.scatter(x, y, s=26, facecolors="white", edgecolors=col, linewidths=1.1, zorder=2)
                    else:
                        ax.scatter(x, y, s=10, color=col, edgecolors="white", linewidths=0.3, zorder=3)
                ax.set_xlim(0.38, 1.0)
                ax.set_ylim(0.38, 1.0)
                ax.set_title(f"{'abcd'[2 * row + j]}. K = {K}, {n_l} lab./class", fontsize=7.5, loc="left", color=INK)
                if row == 1:
                    ax.set_xlabel("student test accuracy")
                if j == 0:
                    ax.set_ylabel("pseudo-label accuracy (optimised from uniform t)")
        handles = [Line2D([], [], marker="o", ls="", mfc="white", mec=STY["opt"][0], ms=5, label="MPL")] + \
                  [Line2D([], [], marker="o", ls="", color=STY[t][0], ms=4, label=lab(t)) for t in T if t != "opt"]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=7, bbox_to_anchor=(0.5, -0.01))
        fig.text(0.01, 0.035, "One point per random student (Sec. 14 protocol).  Dashed: student = pseudo-labels.", fontsize=6.3, color=INK2)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        paths.append(os.path.join(FIG, "toy_tr_pl_vs_student.png"))
        fig.savefig(paths[-1], bbox_inches="tight")
        plt.close(fig)

    # ---- 4. how far the TR signal is from MPL's: cos and relative gap along the sec14 runs ------
    T = [t for t in tags(R, "sec14") if t != "opt"]
    if T:
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.9))
        for i, t in enumerate(T):
            E = [e for r in R[("sec14", t)] for K in KS14 if rec14(r, K) for key in ("trace_uniform", "trace_truth")
                 for e in rec14(r, K)[key] if "cos_v_opt" in e]
            c = np.array([e["cos_v_opt"] for e in E])
            g = np.array([(e["F"] - e["Ll"]) / max(e["Ll"], 1e-300) for e in E])
            col, ls = STY.get(t, (INK2, ":"))
            for ax, vals in ((axes[0], 1 - c), (axes[1], np.abs(g))):
                v = np.sort(np.clip(vals, 1e-12, None))
                ax.plot(v, np.linspace(0, 1, len(v)), color=col, ls=ls, label=lab(t))
        axes[0].set_xscale("log")
        axes[0].set_xlim(1e-6, 2)
        axes[0].set_xlabel("1 - cos(grad of the TR value, grad L_l) at the same student")
        axes[1].set_xscale("log")
        axes[1].set_xlim(1e-6, 1e2)
        axes[1].set_xlabel("|TR value - L_l| / L_l")
        for ax, ttl in zip(axes, ("a. direction of the meta signal", "b. size of the trust-region correction")):
            ax.set_ylabel("fraction of students visited")
            ax.set_title(ttl, fontsize=7.5, loc="left", color=INK)
        axes[0].legend(fontsize=6.5, loc="lower right")
        fig.text(0.01, -0.02, "Empirical distributions over every student visited by the Sec. 14 optimisations (all K, IFT, "
                 "both starts).", fontsize=6.3, color=INK2)
        fig.tight_layout()
        paths.append(os.path.join(FIG, "toy_tr_signal.png"))
        fig.savefig(paths[-1], bbox_inches="tight")
        plt.close(fig)
    print("\n" + "\n".join(paths))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else IN_DIR
    R = load(d)
    print("configurations:", ", ".join(f"{p}/{t} ({len(v)} runs)" for (p, t), v in sorted(R.items())))
    reproduction(R)
    sec13(R)
    sec14(R)
    sec15(R)
    figures(R)


if __name__ == "__main__":
    main()
