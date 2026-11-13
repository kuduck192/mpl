#!/usr/bin/env python3
"""Main table: final test accuracy (EMA teacher = deployed model) per method and label
budget, mean +- std over seeds; paired differences against FixMatch on common seeds
(paired t-test and sign count).  Also writes a markdown table.

  python analysis/main_table.py results/fmnist-ssl-main-a results/fmnist-ssl-main-b [...] --md docs/main_table.md
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy import stats

ORDER = ["sup", "pseudolabel", "meanteacher", "fixmatch", "flexmatch", "mpl", "ebpl", "pbpl",
         "abl_samfb", "abl_self", "abl_fibersam", "abl_labsam", "abl_pbplflex"]
LABEL = {"sup": "Supervised", "pseudolabel": "Pseudo-Label", "meanteacher": "Mean Teacher",
         "fixmatch": "FixMatch", "flexmatch": "FlexMatch", "mpl": "MPL + FixMatch (optimistic)",
         "ebpl": "**EBPL + FixMatch (expected)**", "pbpl": "**PBPL + FixMatch (pessimistic)**",
         "abl_samfb": "PBPL, ε reset each step (= SAM step, blind to X_u)",
         "abl_self": "PBPL, one network (look-ahead student)",
         "abl_fibersam": "FixMatch + fiber-SAM (teacher-anchored, one network)",
         "abl_labsam": "FixMatch + SAM on labeled loss",
         "abl_pbplflex": "PBPL + FlexMatch"}


def load(dirs, key):
    res = defaultdict(dict)  # (method, npc) -> {seed: value}
    extra = defaultdict(dict)
    for d in dirs:
        for f in glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            if "final" not in r or "args" not in r or not r.get("hist"):
                continue
            m = re.match(r"(.+)_(\d+)_s(\d+)$", r["args"]["name"])
            if not m:
                continue
            meth, npc, sd = m.group(1), int(m.group(2)), int(m.group(3))
            res[(meth, npc)][sd] = r["final"].get(key, np.nan)
            extra[(meth, npc)][sd] = dict(r["final"], minutes_total=r.get("minutes", np.nan))
    return res, extra


def fmt(v):
    v = np.asarray(v, float) * 100
    return f"{v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--key", default="test_ema")
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    res, extra = load(args.dirs, args.key)
    npcs = sorted({k[1] for k in res})
    lines = ["| method | " + " | ".join(f"{10 * n} labels" for n in npcs) + " |",
             "|---|" + "---|" * len(npcs)]
    for meth in ORDER:
        cells = []
        for n in npcs:
            v = res.get((meth, n), {})
            cells.append(f"{fmt(list(v.values()))} (n={len(v)})" if v else "—")
        if any(c != "—" for c in cells):
            lines.append(f"| {LABEL.get(meth, meth)} | " + " | ".join(cells) + " |")
    lines += ["", "Paired difference against FixMatch on common seeds (percentage points):", "",
              "| method | labels | seeds | mean diff | wins | paired t p-value |", "|---|---|---|---|---|---|"]
    for meth in ORDER:
        if meth == "fixmatch":
            continue
        for n in npcs:
            a, b = res.get((meth, n), {}), res.get(("fixmatch", n), {})
            common = sorted(set(a) & set(b))
            if len(common) < 2:
                continue
            d = np.array([a[s] - b[s] for s in common]) * 100
            p = stats.ttest_rel([a[s] for s in common], [b[s] for s in common]).pvalue
            lines.append(f"| {LABEL.get(meth, meth)} | {10 * n} | {','.join(map(str, common))} | "
                         f"{d.mean():+.2f} ± {d.std(ddof=1):.2f} | {int((d > 0).sum())}/{len(d)} | {p:.3f} |")
    # wall-clock cost (one T4, 20k iterations) and a few mechanism diagnostics at the end of training
    lines += ["", "Cost and end-of-training diagnostics (mean over runs; 40 + 100 labels pooled):", "",
              "| method | minutes / run (T4) | pseudo-label acc. above threshold | mask rate | student test acc. |",
              "|---|---|---|---|---|"]
    for meth in ORDER:
        rows = [e for (m, n), d in extra.items() if m == meth for e in d.values()]
        if not rows:
            continue

        def mean_of(k, scale=1.0):
            v = [e[k] for e in rows if k in e and e[k] is not None and np.isfinite(e[k])]
            return f"{scale * np.mean(v):.1f}" if v else "—"
        lines.append(f"| {LABEL.get(meth, meth)} | {mean_of('minutes_total')} | {mean_of('pl_acc_masked', 100)} | "
                     f"{mean_of('mask', 100)} | {mean_of('stu_test_ema', 100)} |")
    txt = "\n".join(lines)
    print(txt)
    if args.md:
        with open(args.md, "w") as f:
            f.write(txt + "\n")


if __name__ == "__main__":
    main()
