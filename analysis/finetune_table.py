#!/usr/bin/env python3
"""Deployed models with and without the end-of-run labeled finetune (runs with --ft_iters, 40
labels, seeds 1-10).  Per run: the teacher EMA, the student EMA (two-network runs) and both after
the same labeled finetune (EMA of the finetune trajectory).  The finetune configuration (lr x BN
mode) is selected per run and per model on the validation split (--cfg val, default) or fixed for
every run (--cfg lr0.003_frozen, ...).

Fair comparisons: a finetuned model is compared with the finetuned FixMatch (same finetune, same
seeds); a model without finetune with the FixMatch EMA of the same runs.  Paired t-test, Wilcoxon
signed-rank, Holm over the methods of each column.  Two-network runs without finetune (the
original MPL / PBPL / EBPL rows) are listed for their teacher and student EMA.

  python analysis/finetune_table.py results/fmnist-ssl-*
  python analysis/finetune_table.py results/fmnist-ssl-* --cfg lr0.003_frozen --tex
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats

FT_METHODS = ["fixmatchft", "labsamft", "fibersamft", "mplft", "pbplft", "ebplft", "lsmpl", "lspbpl", "lsebpl",
              "fsmpl", "fspbpl", "fsebpl"]
OLD_TWO = ["mpl", "pbpl", "ebpl"]
LABEL = {"fixmatchft": "FixMatch", "labsamft": "FixMatch + SAM on $L_l$",
         "mplft": "MPL", "pbplft": "PBPL", "ebplft": "EBPL",
         "mpl": "MPL (no finetune run)", "pbpl": "PBPL (no finetune run)", "ebpl": "EBPL (no finetune run)",
         "lsmpl": "MPL + SAM on $L_l$", "lspbpl": "PBPL + SAM on $L_l$", "lsebpl": "EBPL + SAM on $L_l$",
         "fibersamft": "FixMatch + fiber-SAM", "fsmpl": "MPL + fiber-SAM", "fspbpl": "PBPL + fiber-SAM",
         "fsebpl": "EBPL + fiber-SAM"}
COLS = [("t_ema", "teacher EMA"), ("s_ema", "student EMA"), ("t_ft", "teacher ft"), ("s_ft", "student ft")]


def pick(ftm, cfg):
    """Test accuracy (%) of the EMA of the finetune for the configuration selected on val (or
    the fixed one), and that configuration."""
    cfgs = {c: v for c, v in ftm.items() if c != "before"}
    c = max(cfgs, key=lambda k: cfgs[k]["val_ema"]) if cfg == "val" else cfg
    return 100 * cfgs[c]["test_ema"], c


def load(dirs, cfg, npc=4, seeds=range(1, 11)):
    R = defaultdict(dict)
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            if "final" not in r or "args" not in r or not r.get("hist"):
                continue
            m = re.match(r"(.+)_(\d+)_s(\d+)$", r["args"]["name"])
            if not m or int(m.group(2)) != npc or int(m.group(3)) not in seeds:
                continue
            meth, sd = m.group(1), int(m.group(3))
            if meth not in FT_METHODS + OLD_TWO:
                continue
            if r.get("stopped_early") or r["final"].get("iter", 0) < r["args"].get("iters", 0):
                continue
            fin = r["final"]
            x = {"t_ema": 100 * fin["test_ema"], "min": r.get("minutes", np.nan)}
            if "stu_test_ema" in fin:
                x["s_ema"] = 100 * fin["stu_test_ema"]
            ft = r.get("ft")
            if ft:
                x["ft_min"] = ft.get("minutes", np.nan)
                x["t_ft"], x["t_cfg"] = pick(ft["teacher"], cfg)
                if "student" in ft:
                    x["s_ft"], x["s_cfg"] = pick(ft["student"], cfg)
            R[meth][sd] = x
    return R


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty_like(p), 0.0
    for i, j in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - i) * p[j]))
        adj[j] = run
    return adj


def fmt_p(p):
    return "—" if not np.isfinite(p) else (f"{p:.3f}" if p < 0.12 else f"{p:.2f}")


def compare(R, meths, col, ref, ref_col, min_n):
    """Paired differences meth[col] - ref[ref_col] over common seeds, Holm over the methods."""
    rows = []
    for m in meths:
        if m == ref:
            continue
        common = sorted(s for s in set(R[m]) & set(R[ref]) if col in R[m][s] and ref_col in R[ref][s])
        if len(common) < 2:
            continue
        x = np.array([R[m][s][col] for s in common])
        y = np.array([R[ref][s][ref_col] for s in common])
        d = x - y
        p = stats.ttest_rel(x, y).pvalue
        w = stats.wilcoxon(x, y).pvalue if len(common) >= 5 and np.any(d != 0) else np.nan
        rows.append([m, common, d, p, w, np.nan])
    tested = [i for i, r in enumerate(rows) if len(r[1]) >= min_n]
    if tested:
        for i, a in zip(tested, holm([rows[i][3] for i in tested])):
            rows[i][5] = a
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--cfg", default="val", help="'val' (per-run selection) or a fixed config, e.g. lr0.003_frozen")
    ap.add_argument("--ref", default="fixmatchft")
    ap.add_argument("--min_n", type=int, default=5)
    ap.add_argument("--tex", action="store_true", help="also print LaTeX rows")
    args = ap.parse_args()
    R = load(args.dirs, args.cfg)
    meths = [m for m in FT_METHODS + OLD_TWO if m in R]
    if args.ref not in R:
        raise SystemExit(f"no runs of the reference {args.ref}")

    def ms(m, col):
        v = np.array([x[col] for x in R[m].values() if col in x])
        return (f"{v.mean():.2f} ± {v.std(ddof=1):.2f}" if len(v) > 1 else "—"), len(v)

    print(f"Deployed models, 40 labels, seeds 1-10, final iterate; finetune config: {args.cfg}\n")
    print("| method | n | teacher EMA | student EMA | teacher ft | student ft | min/run (train + ft) |")
    print("|---|---|---|---|---|---|---|")
    for m in meths:
        cells = [ms(m, c)[0] for c, _ in COLS]
        tm = np.nanmean([x["min"] for x in R[m].values()])
        fms = [x["ft_min"] for x in R[m].values() if np.isfinite(x.get("ft_min", np.nan))]
        fm = np.mean(fms) if fms else np.nan
        print(f"| {m} | {len(R[m])} | " + " | ".join(cells) + f" | {tm:.1f}" + (f" + {fm:.1f}" if np.isfinite(fm) else "") + " |")

    # fair references: without finetune vs FixMatch EMA, with finetune vs finetuned FixMatch
    plan = [("t_ema", "t_ema"), ("s_ema", "t_ema"), ("t_ft", "t_ft"), ("s_ft", "t_ft")]
    res = {}
    for col, rcol in plan:
        rows = compare(R, [m for m in meths if m in FT_METHODS], col, args.ref, rcol, args.min_n)
        res[col] = {r[0]: r for r in rows}
        name = dict(COLS)[col]
        print(f"\n{name} − FixMatch {dict(COLS)[rcol]} (paired, Holm over {sum(np.isfinite(r[5]) for r in rows)} methods):\n")
        print("| method | n | mean ± sd | wins | t p | Wilcoxon p | Holm p |")
        print("|---|---|---|---|---|---|---|")
        for m, common, d, p, w, a in rows:
            print(f"| {m} | {len(common)} | {d.mean():+.2f} ± {d.std(ddof=1):.2f} | {int((d > 0).sum())}/{len(d)} | "
                  f"{fmt_p(p)} | {fmt_p(w)} | {fmt_p(a)} |")

    print("\nWithin-run comparisons (paired over the seeds of each method, no multiplicity correction):\n")
    print("| method | student EMA − teacher EMA | student ft − teacher EMA | student ft − teacher ft | teacher ft − teacher EMA |")
    print("|---|---|---|---|---|")
    for m in meths:
        cells = []
        for a, b in (("s_ema", "t_ema"), ("s_ft", "t_ema"), ("s_ft", "t_ft"), ("t_ft", "t_ema")):
            xs = [(x[a], x[b]) for x in R[m].values() if a in x and b in x]
            if len(xs) < 2:
                cells.append("—")
                continue
            x, y = np.array(xs).T
            d = x - y
            cells.append(f"{d.mean():+.2f} ± {d.std(ddof=1):.2f}, {int((d > 0).sum())}/{len(d)}, "
                         f"p = {fmt_p(stats.ttest_rel(x, y).pvalue)}")
        print(f"| {m} | " + " | ".join(cells) + " |")

    if args.cfg == "val":
        print("\nFinetune configurations selected on val (count over runs):\n")
        for m in meths:
            for key in ("t_cfg", "s_cfg"):
                c = [x[key] for x in R[m].values() if key in x]
                if c:
                    u, k = np.unique(c, return_counts=True)
                    print(f"  {m} {key[0]}: " + ", ".join(f"{a} x{b}" for a, b in zip(u, k)))

    if args.tex:
        def tex(m, col):
            s, n = ms(m, col)
            return "$" + s.replace(" ± ", r" \pm ") + "$" if n > 1 else "--"

        def delta(m, col):
            r = res[col].get(m)
            if r is None:
                return "--"
            d = r[2]
            return (f"${d.mean():+.2f}$, {int((d > 0).sum())}/{len(d)}, "
                    f"{fmt_p(r[3])} / {fmt_p(r[4])} / {fmt_p(r[5])}")

        print("\n% variant & teacher EMA & student EMA & teacher ft. & student ft. & "
              "Delta student ft. vs FixMatch ft. (mean, wins, p t / W / Holm)")
        for m in meths:
            if m in FT_METHODS:
                cells = [tex(m, c) for c, _ in COLS] + [delta(m, "s_ft")]
                print(f"{LABEL[m]} & " + " & ".join(cells) + r" \\")


if __name__ == "__main__":
    main()
