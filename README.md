# TR-BPL: trust-region bilevel pseudo-labeling on FashionMNIST

Pessimistic (PBPL) and expected (EBPL) trust-region bilevel pseudo-labeling, used as a
regularizer on top of FixMatch, plus the baselines and ablations needed to test it.

- `ssl_fmnist.py` — everything (data, GPU augmentation, CNN, FixMatch / FlexMatch /
  Pseudo-Label / Mean Teacher, MPL feedback, PBPL / EBPL feedback, SAM controls).
- `docs/theory.md` — formulation, non-degeneracy propositions and proofs, algorithm.
- `checks/` — numerical checks: exactness of the meta-gradient (`check_metagrad.py`) and
  the fixed-point non-degeneracy proposition on a toy MLP (`check_nondegeneracy.py`).
- `kaggle_tools/kaggle_run.py` — builds a Kaggle script kernel from a sweep file and runs
  one job per GPU (T4 x2); `push`, `status`, `fetch`, `logs`.
- `sweeps/` — the sweep files that were run; `results/` — downloaded outputs;
  `analysis/aggregate.py` — tables.

Quick start (CPU smoke test):

```bash
python ssl_fmnist.py --iters 20 --bl 8 --mu 2 --eval_every 10 --eval_max 500 --device cpu --bilevel two --feedback pess
```

Main configurations:

| method | flags |
|---|---|
| FixMatch | `--base fixmatch` |
| FlexMatch | `--base flexmatch` |
| MPL + FixMatch (optimistic feedback) | `--bilevel two --feedback opt` |
| **PBPL + FixMatch** | `--bilevel two --feedback pess --radius r --beta b` |
| **EBPL + FixMatch** | `--bilevel two --feedback exp --radius r --k_proj K` |
| SAM-feedback control (ε reset, blind to X_u) | `--bilevel two --feedback sam` |
| one-network PBPL (look-ahead student) | `--bilevel self --feedback pess` |
| FixMatch + SAM / fiber-SAM on labeled loss | `--lab_reg sam` / `--lab_reg fibersam` |

Kaggle: put your username in `.kaggle_user` (untracked) and your API token in
`~/.kaggle/access_token`; never commit either.
