#!/usr/bin/env python3
"""Compact table for screening runs: final accuracies + diagnostics averaged over the
last k evaluation records.   python analysis/screen_table.py results/<slug> [...]"""
import glob
import json
import os
import sys

import numpy as np

DIAG = ["lab_loss_anchor", "lab_loss_adv", "c_norm", "c_opt_norm", "cos_c_copt", "fiber_kl", "eps_norm",
        "pl_acc_masked", "mask", "ent_u", "stu_pl_agree"]


def main(dirs, k=3):
    rows = []
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True)):
            try:
                r = json.load(open(f))
            except Exception:
                continue
            if "hist" not in r or not r["hist"]:
                continue
            h = r["hist"]
            fi = h[-1]
            diag = {key: np.nanmean([x.get(key, np.nan) for x in h[-k:]]) for key in DIAG}
            rows.append((r["args"]["name"], fi.get("val_ema", np.nan), fi.get("test_ema", np.nan),
                         fi.get("stu_val_ema", np.nan), fi.get("stu_test_ema", np.nan), diag, r.get("minutes", 0)))
    rows.sort(key=lambda x: -x[1])
    print(f"{'name':28s} {'val':>6s} {'test':>6s} {'s_val':>6s} {'s_tst':>6s} {'min':>5s} | " +
          " ".join(f"{d[:9]:>9s}" for d in DIAG))
    for name, va, te, sva, ste, diag, mins in rows:
        print(f"{name:28s} {100*va:6.2f} {100*te:6.2f} {100*sva:6.2f} {100*ste:6.2f} {mins:5.0f} | " +
              " ".join(f"{diag[d]:9.4f}" for d in DIAG))


if __name__ == "__main__":
    main(sys.argv[1:])
