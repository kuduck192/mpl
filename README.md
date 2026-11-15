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
`rep-b.json`), `wave3` (`rep-c.json`, `rep-d.json`), `wave4` (`rep-e.json`, `rep-f.json`) and
`wave5` (`rep-g.json`, `rep-h.json`: SAM on the labeled loss combined with the two-network
feedback, `--lab_reg sam --lab_radius 0.1`, and the end-of-run labeled finetune of every
deployed model, `--ft_iters 1000`).
`python3 sweeps/make_sweeps.py remainder <sweep.json> <new-slug> <hours> <results-dirs...>`
writes the jobs of a sweep that have no complete result.  Paired tables on any seed range or
reference, and the ablation diagnostics:

```bash
python3 analysis/main_table.py results/fmnist-ssl-* --seeds 1-10 --ref fixmatch --select final --no_cost
```

```bash
python3 analysis/ablation_table.py results/fmnist-ssl-* --seeds 1-3
```

Every 40-label variant on seeds 1-10, paired to FixMatch with t-test, Wilcoxon and Holm:

```bash
python3 analysis/tenseed_table.py results/fmnist-ssl-* --tex
```

The same with the per-seed mean of the original and re-run FixMatch as the reference:

```bash
python3 analysis/tenseed_table.py results/fmnist-ssl-* --ref_avg fixmatch,fixmatchft
```

Teacher EMA, student EMA and both after the end-of-run finetune, compared with the finetuned
FixMatch and FixMatch + SAM (report §12):

```bash
python3 analysis/finetune_table.py results/fmnist-ssl-* --tex
```

Raw feedback sizes of the one-network runs per seed (shrink of the optimistic feedback,
pessimistic over optimistic on the same anchor, report §6.1):

```bash
python3 analysis/feedback_dynamics.py results/fmnist-ssl-* --methods abl_selfopt,abl_self
```

`checks/degeneracy_static.py`
(Prop. N2 on the real CNN) runs on CPU in about an hour; `analysis/figures.py` renders the
figures in `docs/figures/`.

## Results in one paragraph (details and caveats in `docs/report.md`)

The non-degeneracy result holds where it applies: at supervised-only pairs on the real CNN
the pessimistic feedback on the fiber is a median 134 times the optimistic one and does not
shrink as the network fits the labels (report §5); in one-network training the raw optimistic
feedback shrinks 12-39 times while the pessimistic one stays 12-65 times larger (report §6.1).
It does not translate into accuracy. At 40 labels over seeds 1-10 (final iterate, EMA model):
FixMatch 72.75, tuned FixMatch 73.71, MPL 73.28, PBPL 73.32, EBPL 72.74. PBPL − FixMatch =
+0.58 ± 3.10 points (7/10 seeds, paired t p = 0.57), EBPL − FixMatch = −0.01; the gain of PBPL
on seeds 1-5 (+2.02, p = 0.019) did not replicate. On the same seeds the pessimistic rule
does not differ from the optimistic one: +0.04 with two networks, −0.48 with one network, and +0.13
at the exact meta-gradient scale (seeds 1-5, 95% interval −3.9 to +4.2), where the magnitude is
not normalised away and the pessimistic meta steps are 2-6 times larger over training. The two
variants ahead of FixMatch at p < 0.05 (uncorrected) are not pessimistic: SAM on the labeled
loss, no bilevel, 1.25x the cost of FixMatch (75.79, +3.05, p = 0.014; +2.59 on the seeds run
after the decision to test it) and the one-network optimistic look-ahead (74.73, +1.98,
p = 0.041). The best pessimistic variant is the one-network look-ahead (74.25, +1.51,
p = 0.094). After Holm correction over the nine comparisons none is significant at 5%
(report §6.1, §9.2). PBPL costs 3.6x FixMatch.  Combining the two-network feedback with SAM on
the labeled loss does not add up (report §12): MPL / PBPL / EBPL + SAM reach 74.66 / 73.67 /
73.26, i.e. 1.1 / 2.1 / 2.5 points below SAM alone (EBPL: 2/10 seeds, p = 0.011).  The same
labeled finetune applied to every deployed model adds about 0.2 points to a teacher and 0.4 to
a held-out student; the best finetuned student (MPL + SAM, 75.08) stays below the finetuned
FixMatch + SAM (75.92).  Re-running FixMatch on the same seeds moves it by -3.5 to +7.5 points
(GPU non-determinism; the training code is bit-identical on CPU), so a single FixMatch run per
seed carries about 2 points of noise; against the per-seed mean of two FixMatch runs, SAM on the
labeled loss is the only variant significant after Holm over 12 comparisons (+2.65, p = 0.048).

Kaggle: put your username in `.kaggle_user` (untracked) and your API token in
`~/.kaggle/access_token`; never commit either.
