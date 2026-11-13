# TR-BPL: trust-region bilevel pseudo-labeling on FashionMNIST

Pessimistic (PBPL) and expected (EBPL) trust-region bilevel pseudo-labeling, used as a
regularizer on top of FixMatch, plus the baselines and ablations needed to test it.

- `ssl_fmnist.py` — everything (data, GPU augmentation, CNN, FixMatch / FlexMatch /
  Pseudo-Label / Mean Teacher, MPL feedback, PBPL / EBPL feedback, SAM controls).
- `docs/theory.md` — formulation, non-degeneracy propositions and proofs, algorithm.
- `checks/` — numerical checks: exactness of the meta-gradient (`check_metagrad.py`) and
  the fixed-point non-degeneracy proposition on a toy MLP (`check_nondegeneracy.py`) and on
  the real CNN (`degeneracy_static.py`); `figure1_fmnist.py` reproduces the review protocol.
- `kaggle_tools/kaggle_run.py` — builds a Kaggle script kernel from a sweep file and runs
  one job per GPU (T4 x2); `push`, `status`, `fetch`, `logs`.
- `sweeps/` — the sweep files that were run; `results/` — downloaded outputs;
  `analysis/` — tables and figures; `docs/report.md` — the experimental report.

Quick start (CPU smoke test):

```bash
python ssl_fmnist.py --iters 20 --bl 8 --mu 2 --eval_every 10 --eval_max 500 --device cpu --bilevel two --feedback pess
```

Main configurations (hyper-parameters selected on the validation split, seed 0, 40 labels):

| method | flags |
|---|---|
| FixMatch | `--base fixmatch` (τ = 0.95, λ_u = 1) |
| FixMatch, tuned (best of a 6-config grid, same protocol) | `--base fixmatch --tau 0.95 --lam_u 2.0` |
| FlexMatch | `--base flexmatch` |
| MPL + FixMatch (optimistic feedback) | `--bilevel two --feedback opt --lam_m 0.1` |
| **PBPL + FixMatch** (pessimistic, hard radius) | `--bilevel two --feedback pess --lam_m 0.1 --radius 0.1` |
| **EBPL + FixMatch** (expected) | `--bilevel two --feedback exp --lam_m 0.3 --radius 3.0` |
| SAM-feedback control (ε reset, blind to X_u) | `--bilevel two --feedback sam --lam_m 0.1 --radius 0.1` |
| one-network PBPL (look-ahead student) | `--bilevel self --feedback pess --lam_m 0.1 --radius 0.1` |
| one-network MPL (optimistic control of the above) | `--bilevel self --feedback opt --lam_m 0.1` |
| FixMatch + SAM / fiber-SAM on labeled loss | `--lab_reg sam --radius 0.1` / `--lab_reg fibersam --radius 0.1` |

Add `--n_per_class {4,10} --seed s --iters 20000` for the main-table runs.  The inner
pessimistic problem uses the ascent-then-retract solver (`--pess_solver retract --k_proj 2
--alpha 0.5`, defaults); the feedback is normalised by its running RMS (`--meta_norm 2`).

Reproducing the experiments (Kaggle, T4 x2, two kernels at a time):

```bash
python3 sweeps/make_sweeps.py main_table
```

```bash
python3 kaggle_tools/kaggle_run.py push sweeps/main-a.json
```

```bash
python3 kaggle_tools/kaggle_run.py fetch fmnist-ssl-main-a
```

```bash
python3 analysis/main_table.py results/fmnist-ssl-main-a results/fmnist-ssl-main-b --md docs/main_table.md
```

The same for `ablations` (`ablate-a.json`, `ablate-b.json`), `replication` (`rep-a.json`,
`rep-b.json`) and `wave3` (`rep-c.json`, `rep-d.json`; it reads `results/fmnist-ssl-rep-b`).
`python3 sweeps/make_sweeps.py remainder <sweep.json> <new-slug> <hours> <results-dirs...>`
writes the jobs of a sweep that have no complete result.  Paired tables on any seed range or
reference, and the ablation diagnostics:

```bash
python3 analysis/main_table.py results/fmnist-ssl-* --seeds 1-10 --ref fixmatch --select final --no_cost
```

```bash
python3 analysis/ablation_table.py results/fmnist-ssl-* --seeds 1-3
```

`checks/degeneracy_static.py`
(Prop. N2 on the real CNN) runs on CPU in about an hour; `analysis/figures.py` renders the
figures in `docs/figures/`.

## Results in one paragraph (details and caveats in `docs/report.md`)

The non-degeneracy result holds where it applies: at supervised-only pairs on the real CNN
the pessimistic feedback on the fiber is a median 134 times the optimistic one and does not
shrink as the network fits the labels (report §5). It does not translate into accuracy. At 40
labels over seeds 1-10 (final iterate, EMA model), FixMatch 72.75, tuned FixMatch 73.71, MPL
73.28 and PBPL 73.32: PBPL − FixMatch = +0.58 ± 3.10 points (7/10 seeds, paired t p = 0.57),
PBPL − MPL = +0.04 ± 3.47. The gain on seeds 1-5 (+2.02, p = 0.019) did not replicate on
seeds 6-10 (−0.87). Non-bilevel SAM-type controls (SAM on the labeled loss, 1.2× the cost of
FixMatch) match or exceed PBPL on three seeds, and the one-network optimistic look-ahead beats
the pessimistic one on two seeds because the running-RMS normalisation of the feedback undoes
the vanishing magnitude that the theory is about (report §6.1).  PBPL costs 3.6× FixMatch.

Kaggle: put your username in `.kaggle_user` (untracked) and your API token in
`~/.kaggle/access_token`; never commit either.
