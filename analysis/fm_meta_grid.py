#!/usr/bin/env python3
"""Ablation grid of the unlabeled losses, 40 labels, seeds 1-10:

    rows    : labeled loss only (sup: labeled batches only; supbn: unlabeled batches in the forward,
              so BatchNorm sees them, no unlabeled loss) and labeled loss + FixMatch;
    columns : no bilevel term, MPL (optimistic), PBPL (pessimistic), EBPL (expected) feedback.

For the test accuracy of the deployed EMA teacher (--metric test_ema, default), its pseudo-label
accuracy on X_u (--metric unl_ema), or the test accuracy of the live, non-averaged teacher
(--metric test; a check on the EMA, whose BatchNorm buffers are copied from the live model): the grid of means, the effect of the meta term within each row
(paired by seed against the same row without it), the effect of FixMatch within each column, their
interaction (meta effect with FixMatch minus meta effect without), and pessimistic / expected
minus optimistic within each row.  Paired t-test, Wilcoxon, Holm over each family.

  python analysis/fm_meta_grid.py results/fmnist-ssl-*
  python analysis/fm_meta_grid.py results/fmnist-ssl-* --metric unl_ema
  python analysis/fm_meta_grid.py results/fmnist-ssl-* --metric test
"""
import argparse
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tenseed_table import fmt_p, holm, load  # noqa: E402

ROWS = [("labeled loss only", {"none": "sup"}),
        ("labeled loss, unlabeled batches in BN", {"none": "supbn", "opt": "supmpl", "pess": "suppbpl", "exp": "supebpl"}),
        ("labeled loss + FixMatch", {"none": "fixmatch", "opt": "mpl", "pess": "pbpl", "exp": "ebpl"})]
COLS = [("none", "no meta"), ("opt", "+ MPL"), ("pess", "+ PBPL"), ("exp", "+ EBPL")]


def paired(R, a, b, key="final"):
    if a not in R or b not in R:
        return None
    s = sorted(set(R[a]) & set(R[b]))
    if len(s) < 2:
        return None
    x = np.array([R[a][i][key] for i in s])
    y = np.array([R[b][i][key] for i in s])
    d = x - y
    w = stats.wilcoxon(x, y).pvalue if len(s) >= 5 and np.any(d != 0) else np.nan
    return {"n": len(s), "d": d, "p": stats.ttest_rel(x, y).pvalue, "w": w, "seeds": s}


def report(title, items):
    """items: list of (label, result) -> printed with Holm over the tested items."""
    rows = [(lab, r) for lab, r in items if r is not None]
    tested = [i for i, (_, r) in enumerate(rows) if r["n"] >= 5]
    adj = np.full(len(rows), np.nan)
    if tested:
        adj[tested] = holm([rows[i][1]["p"] for i in tested])
    print(f"\n{title} (paired by seed; Holm over {len(tested)})\n")
    print("| comparison | n | mean ± sd | wins | t p | Wilcoxon p | Holm p |")
    print("|---|---|---|---|---|---|---|")
    for (lab, r), a in zip(rows, adj):
        d = r["d"]
        print(f"| {lab} | {r['n']} | {d.mean():+.2f} ± {d.std(ddof=1):.2f} | {int((d > 0).sum())}/{len(d)} | "
              f"{fmt_p(r['p'])} | {fmt_p(r['w'])} | {fmt_p(a)} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--metric", default="test_ema", choices=["test_ema", "unl_ema", "test"])
    args = ap.parse_args()
    R = load(args.dirs, metric=args.metric)
    what = {"test_ema": "deployed EMA teacher, test accuracy",
            "unl_ema": "deployed EMA teacher, pseudo-label accuracy on X_u",
            "test": "live (non-averaged) teacher, test accuracy"}[args.metric]
    print(f"{what[0].upper() + what[1:]}, 40 labels, final iterate; mean ± sd (n seeds)\n")
    print("| unlabeled loss | " + " | ".join(c for _, c in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for name, cells in ROWS:
        out = []
        for key, _ in COLS:
            m = cells.get(key)
            if m is None or m not in R:
                out.append("—" if m is None else "not run")
                continue
            v = np.array([x["final"] for x in R[m].values()])
            out.append(f"{v.mean():.2f} ± {v.std(ddof=1):.2f} ({len(v)})" if len(v) > 1 else f"{v.mean():.2f} (1)")
        print(f"| {name} | " + " | ".join(out) + " |")

    meta = []
    for name, cells in ROWS[1:]:
        for key, lab in COLS[1:]:
            meta.append((f"{lab[2:]} − no meta, {name}", paired(R, cells[key], cells["none"])))
    report("Effect of the meta term within each row", meta)
    fm = [(f"FixMatch − none, {lab}", paired(R, ROWS[2][1][key], ROWS[1][1][key])) for key, lab in COLS]
    fm.append(("unlabeled batches in BN − labeled only (no meta)", paired(R, "supbn", "sup")))
    report("Effect of FixMatch within each column (and of the unlabeled forward pass)", fm)
    inter = []
    for key, lab in COLS[1:]:
        a, b = paired(R, ROWS[2][1][key], ROWS[2][1]["none"]), paired(R, ROWS[1][1][key], ROWS[1][1]["none"])
        if a is None or b is None:
            continue
        common = sorted(set(a["seeds"]) & set(b["seeds"]))
        da = np.array([dict(zip(a["seeds"], a["d"]))[s] for s in common])
        db = np.array([dict(zip(b["seeds"], b["d"]))[s] for s in common])
        r = {"n": len(common), "d": da - db, "p": stats.ttest_rel(da, db).pvalue if len(common) > 1 else np.nan,
             "w": stats.wilcoxon(da, db).pvalue if len(common) >= 5 and np.any(da != db) else np.nan, "seeds": common}
        inter.append((f"{lab[2:]}: meta effect with FixMatch − without", r))
    report("Interaction (does FixMatch change what the meta term adds?)", inter)
    pess = []
    for name, cells in ROWS[1:]:
        pess.append((f"PBPL − MPL, {name}", paired(R, cells["pess"], cells["opt"])))
        pess.append((f"EBPL − MPL, {name}", paired(R, cells["exp"], cells["opt"])))
    report("Pessimistic / expected minus optimistic within each row", pess)
    print("\nLabeled loss + meta (no FixMatch) against FixMatch (labeled loss + FixMatch, no meta):")
    report("Sup + meta vs Sup + FM", [(f"{c} − fixmatch", paired(R, c, "fixmatch")) for c in ("supmpl", "suppbpl", "supebpl")])
    print("\nLabeled loss + meta (no FixMatch) against training on the labeled batches alone:")
    report("Sup + meta vs Sup", [(f"{c} − sup", paired(R, c, "sup")) for c in ("supmpl", "suppbpl", "supebpl")])


if __name__ == "__main__":
    main()
