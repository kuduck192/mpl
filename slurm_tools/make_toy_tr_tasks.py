#!/usr/bin/env python3
"""Task tables of the Two Moons trust-region runs (checks/toy_two_moons_tr.py) for slurm_tools/toy_tr.sbatch.

One line per Slurm array task: protocol, mode, fibre penalty beta, radius, shard index, number of shards.
  python3 slurm_tools/make_toy_tr_tasks.py > slurm_tools/toy_tr_tasks.txt            # first launch
  python3 slurm_tools/make_toy_tr_tasks.py rerun > slurm_tools/toy_tr_tasks_rerun.txt  # TR-PBPL rerun
The rerun repeats every TR-PBPL run after the zero-start fallback of tr_fiber.pess (the first launch left
the persistent adversary below Phi(0) = L_l at 58-68% of the IFT students), most decisive first, and adds
the Sec. 13 TR-EBPL shards that had not run yet.
"""
import sys

CONFIGS = [("pess", 1e4, 1.0), ("pess", 10.0, 1.0), ("exp", 1e4, 1.0)]
SHARDS = {"sec13": 8, "sec14": 4, "sec15": 2}
if len(sys.argv) > 1 and sys.argv[1] == "rerun":
    ORDER = [("sec14", "pess", 1e4), ("sec15", "pess", 1e4), ("sec14", "pess", 10.0), ("sec15", "pess", 10.0),
             ("sec13", "pess", 1e4), ("sec13", "exp", 1e4), ("sec13", "pess", 10.0)]
else:
    ORDER = [(proto, mode, beta) for proto in ("sec14", "sec15", "sec13") for mode, beta, _ in CONFIGS]
for proto, mode, beta in ORDER:
    n = SHARDS[proto]
    for i in range(n):
        print(f"{proto} {mode} {beta:g} 1 {i} {n}")
