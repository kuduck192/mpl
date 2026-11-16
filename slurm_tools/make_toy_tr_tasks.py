#!/usr/bin/env python3
"""Task table of the Two Moons trust-region runs (checks/toy_two_moons_tr.py) for slurm_tools/toy_tr.sbatch.

One line per Slurm array task: protocol, mode, fibre penalty beta, radius, shard index, number of shards.
  python3 slurm_tools/make_toy_tr_tasks.py > slurm_tools/toy_tr_tasks.txt
"""
CONFIGS = [("pess", 1e4, 1.0), ("pess", 10.0, 1.0), ("exp", 1e4, 1.0)]
SHARDS = {"sec13": 4, "sec14": 4, "sec15": 2}
for proto in ("sec14", "sec15", "sec13"):
    for mode, beta, r in CONFIGS:
        n = SHARDS[proto]
        for i in range(n):
            print(f"{proto} {mode} {beta:g} {r:g} {i} {n}")
