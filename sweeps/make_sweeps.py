#!/usr/bin/env python3
"""Generate sweep files.  Usage: python sweeps/make_sweeps.py <phase>"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def write(slug, jobs, hours, files=()):
    path = os.path.join(HERE, slug.replace("fmnist-ssl-", "") + ".json")
    json.dump({"slug": slug, "budget_hours": hours, "files": list(files), "jobs": jobs}, open(path, "w"), indent=1)
    print(path, len(jobs), "jobs")


def phase1(it=20000):
    """Tuning on the validation split (seed 0, 4 labels/class) + the degeneracy diagnostic."""
    common = f"--n_per_class 4 --seed 0 --iters {it}"
    a, b = [], []
    for lm in [0.1, 0.3, 1.0]:
        for r in [0.1, 0.3, 1.0]:
            a.append({"name": f"tune_pess_lm{lm}_r{r}", "args": f"--bilevel two --feedback pess --lam_m {lm} --radius {r} {common}"})
    for lm in [0.3, 1.0]:
        for r in [1.0, 3.0, 10.0]:
            b.append({"name": f"tune_exp_lm{lm}_r{r}", "args": f"--bilevel two --feedback exp --lam_m {lm} --radius {r} {common}"})
    for lm in [0.1, 0.3, 1.0]:
        b.append({"name": f"tune_opt_lm{lm}", "args": f"--bilevel two --feedback opt --lam_m {lm} {common}"})
    b.append({"name": "tune_fixmatch_s0", "args": f"--base fixmatch {common}"})
    for sd in (1, 2, 3):
        for fbk in ["opt", "pess", "exp"]:
            r = {"opt": 0.3, "pess": 0.3, "exp": 3.0}[fbk]
            job = {"name": f"degen_{fbk}_s{sd}",
                   "args": f"--bilevel two --feedback {fbk} --radius {r} --warm_sup_iters 3000 --lam_u 0 "
                           f"--stu_view weak --meta_norm 0 --lam_m 30 --n_per_class 10 --seed {sd} "
                           f"--iters 4000 --eval_every 250 --log_every 50"}
            (a if fbk != "exp" else b).append(job)
    write("fmnist-ssl-phase1-a", a, 11.0)
    write("fmnist-ssl-phase1-b", b, 11.0)


def screen(it=10000):
    """Phase 1 (v2, after the held-out fix): 10k-iteration screen on the validation split
    (seed 0, 4 labels/class; FixMatch plateaus by ~5k there) + degeneracy diagnostic."""
    common = f"--n_per_class 4 --seed 0 --iters {it}"
    a, b = [], []
    for lm in [0.1, 0.3, 1.0]:
        for r in [0.1, 0.3, 1.0]:
            a.append({"name": f"scr_pess_lm{lm}_r{r}", "args": f"--bilevel two --feedback pess --lam_m {lm} --radius {r} {common}"})
    for lm in [10, 30, 100]:
        a.append({"name": f"scr_pessraw_lm{lm}_r0.3", "args": f"--bilevel two --feedback pess --meta_norm 0 --lam_m {lm} --radius 0.3 {common}"})
    for lm in [0.3, 1.0]:
        for r in [1.0, 3.0, 10.0]:
            b.append({"name": f"scr_exp_lm{lm}_r{r}", "args": f"--bilevel two --feedback exp --lam_m {lm} --radius {r} {common}"})
    for lm in [0.1, 0.3, 1.0]:
        b.append({"name": f"scr_opt_lm{lm}", "args": f"--bilevel two --feedback opt --lam_m {lm} {common}"})
    for lm in [10, 30, 100]:
        b.append({"name": f"scr_optraw_lm{lm}", "args": f"--bilevel two --feedback opt --meta_norm 0 --lam_m {lm} {common}"})
    b.append({"name": "scr_fixmatch", "args": f"--base fixmatch {common}"})
    b.append({"name": "scr_flexmatch", "args": f"--base flexmatch {common}"})
    for sd in (1, 2, 3):
        for fbk in ["opt", "pess", "exp"]:
            r = {"opt": 0.3, "pess": 0.3, "exp": 3.0}[fbk]
            job = {"name": f"degen_{fbk}_s{sd}",
                   "args": f"--bilevel two --feedback {fbk} --radius {r} --warm_sup_iters 3000 --lam_u 0 "
                           f"--stu_view weak --meta_norm 0 --lam_m 30 --n_per_class 10 --seed {sd} "
                           f"--iters 4000 --eval_every 250 --log_every 50"}
            (a if fbk != "exp" else b).append(job)
    write("fmnist-ssl-screen-a", a, 11.0)
    write("fmnist-ssl-screen-b", b, 11.0)


def phase2(pess="--lam_m 0.3 --radius 0.3", exp="--lam_m 0.3 --radius 3.0", opt="--lam_m 0.3",
           it=20000, seeds=(1, 2, 3), npcs=(4, 10)):
    """Main table + ablations on the reporting seeds.  Hyper-parameters of the bilevel
    variants come from the validation screen (seed 0)."""
    jobs_a, jobs_b = [], []
    for npc in npcs:
        for sd in seeds:
            c = f"--n_per_class {npc} --seed {sd} --iters {it}"
            jobs_a += [
                {"name": f"pbpl_{npc}_s{sd}", "args": f"--bilevel two --feedback pess {pess} {c}"},
                {"name": f"fixmatch_{npc}_s{sd}", "args": f"--base fixmatch {c}"},
                {"name": f"flexmatch_{npc}_s{sd}", "args": f"--base flexmatch {c}"},
                {"name": f"sup_{npc}_s{sd}", "args": f"--base sup {c}"},
            ]
            jobs_b += [
                {"name": f"ebpl_{npc}_s{sd}", "args": f"--bilevel two --feedback exp {exp} {c}"},
                {"name": f"mpl_{npc}_s{sd}", "args": f"--bilevel two --feedback opt {opt} {c}"},
                {"name": f"meanteacher_{npc}_s{sd}", "args": f"--base meanteacher --lam_u 50 {c}"},
                {"name": f"pseudolabel_{npc}_s{sd}", "args": f"--base pseudolabel {c}"},
            ]
    for sd in seeds:  # ablations, 4 labels / class
        c = f"--n_per_class 4 --seed {sd} --iters {it}"
        jobs_a += [
            {"name": f"abl_samfb_4_s{sd}", "args": f"--bilevel two --feedback sam {pess} {c}"},
            {"name": f"abl_self_4_s{sd}", "args": f"--bilevel self --feedback pess {pess} {c}"},
            {"name": f"abl_penaltysolver_4_s{sd}", "args": f"--bilevel two --feedback pess --pess_solver penalty {pess} {c}"},
        ]
        jobs_b += [
            {"name": f"abl_fibersam_4_s{sd}", "args": f"--lab_reg fibersam --radius 0.3 {c}"},
            {"name": f"abl_labsam_4_s{sd}", "args": f"--lab_reg sam --radius 0.05 {c}"},
            {"name": f"abl_pbplflex_4_s{sd}", "args": f"--base flexmatch --bilevel two --feedback pess {pess} {c}"},
        ]
    jobs_b.append({"name": "figure1", "script": "figure1_fmnist.py",
                   "args": "--out /kaggle/working/results/figure1.json --device cuda"})
    write("fmnist-ssl-phase2-a", jobs_a, 11.5)
    write("fmnist-ssl-phase2-b", jobs_b, 11.5, files=["checks/figure1_fmnist.py"])


PESS = "--lam_m 0.1 --radius 0.1"   # selected on the validation screen (seed 0)
EXP = "--lam_m 0.3 --radius 3.0"
OPT = "--lam_m 0.1"
FMT = "--base fixmatch --tau 0.95 --lam_u 2.0"  # best FixMatch of the ablate-a grid (screen val)


def main_table(it=20000):
    """Phase 2a: main table.  FixMatch / MPL / PBPL on 5 seeds (paired comparison), the other
    methods on 3 seeds; 40 and 100 labels.  Longest jobs first in each queue."""
    a, b = [], []
    for npc in (4, 10):
        for sd in (1, 2, 3, 4, 5):
            c = f"--n_per_class {npc} --seed {sd} --iters {it}"
            a.append({"name": f"pbpl_{npc}_s{sd}", "args": f"--bilevel two --feedback pess {PESS} {c}"})
            b.append({"name": f"mpl_{npc}_s{sd}", "args": f"--bilevel two --feedback opt {OPT} {c}"})
    for npc in (4, 10):
        for sd in (1, 2, 3):
            c = f"--n_per_class {npc} --seed {sd} --iters {it}"
            b.insert(0, {"name": f"ebpl_{npc}_s{sd}", "args": f"--bilevel two --feedback exp {EXP} {c}"})
    for npc in (4, 10):
        for sd in (1, 2, 3, 4, 5):
            c = f"--n_per_class {npc} --seed {sd} --iters {it}"
            a.append({"name": f"fixmatch_{npc}_s{sd}", "args": f"--base fixmatch {c}"})
    for npc in (4, 10):
        for sd in (1, 2, 3):
            c = f"--n_per_class {npc} --seed {sd} --iters {it}"
            a.append({"name": f"flexmatch_{npc}_s{sd}", "args": f"--base flexmatch {c}"})
            a.append({"name": f"sup_{npc}_s{sd}", "args": f"--base sup {c}"})
            b.append({"name": f"meanteacher_{npc}_s{sd}", "args": f"--base meanteacher --lam_u 50 {c}"})
            b.append({"name": f"pseudolabel_{npc}_s{sd}", "args": f"--base pseudolabel {c}"})
    write("fmnist-ssl-main-a", a, 11.5)
    write("fmnist-ssl-main-b", b, 11.5)


def ablations(it=20000):
    """Phase 2b: ablations (40 labels, seeds 1-3) + diagnostics."""
    a, b = [], []
    for sd in (1, 2, 3):
        c = f"--n_per_class 4 --seed {sd} --iters {it}"
        a.append({"name": f"abl_self_4_s{sd}", "args": f"--bilevel self --feedback pess {PESS} {c}"})
        a.append({"name": f"abl_samfb_4_s{sd}", "args": f"--bilevel two --feedback sam {PESS} {c}"})
        b.append({"name": f"abl_pbplflex_4_s{sd}", "args": f"--base flexmatch --bilevel two --feedback pess {PESS} {c}"})
        b.append({"name": f"abl_fibersam_4_s{sd}", "args": f"--lab_reg fibersam --radius 0.1 {c}"})
        b.append({"name": f"abl_labsam_4_s{sd}", "args": f"--lab_reg sam --radius 0.1 {c}"})
    # FixMatch gets a tuning budget comparable to MPL / EBPL (6 configurations), same protocol
    # as the bilevel screen: validation accuracy, seed 0, 40 labels, 10k iterations.
    for tau in (0.8, 0.9, 0.95):
        for lu in (1.0, 2.0):
            if (tau, lu) != (0.95, 1.0):  # default = scr_fixmatch in the screen
                a.append({"name": f"scr_fm_tau{tau}_lu{lu}",
                          "args": f"--base fixmatch --tau {tau} --lam_u {lu} --n_per_class 4 --seed 0 --iters 10000"})
    # the static degeneracy check runs on CPU (results/local/) unless asked to run on Kaggle
    if os.environ.get("DEGEN_ON_KAGGLE"):
        a.append({"name": "degeneracy_static", "script": "degeneracy_static.py",
                  "args": "--out /kaggle/working/results/degeneracy_static.json --device cuda"})
    b.append({"name": "figure1", "script": "figure1_fmnist.py",
              "args": "--out /kaggle/working/results/figure1.json --device cuda"})
    files = ["checks/figure1_fmnist.py", "checks/degeneracy_static.py"]
    write("fmnist-ssl-ablate-a", a, 11.5, files=files)
    write("fmnist-ssl-ablate-b", b, 11.5, files=files)


def tune(it=20000):
    """Hyper-parameter search on the validation split (seed 0, 4 labels/class)."""
    common = f"--n_per_class 4 --seed 0 --iters {it}"
    a, b = [], []
    for lm in [0.1, 0.3, 1.0]:
        for r in [0.1, 0.3, 1.0]:
            a.append({"name": f"tune_pess_lm{lm}_r{r}", "args": f"--bilevel two --feedback pess --lam_m {lm} --radius {r} {common}"})
    for lm in [0.3, 1.0]:
        for r in [1.0, 3.0, 10.0]:
            b.append({"name": f"tune_exp_lm{lm}_r{r}", "args": f"--bilevel two --feedback exp --lam_m {lm} --radius {r} {common}"})
    for lm in [0.1, 0.3, 1.0]:
        b.append({"name": f"tune_opt_lm{lm}", "args": f"--bilevel two --feedback opt --lam_m {lm} {common}"})
    write("fmnist-ssl-tune-a", a, 11.0)
    write("fmnist-ssl-tune-b", b, 11.0)


def baselines(it=20000, seeds=(1, 2, 3), npcs=(4, 10)):
    """Classical SSL baselines on the reporting seeds (no tuning: standard hyper-parameters)."""
    jobs = []
    for npc in npcs:
        for sd in seeds:
            c = f"--n_per_class {npc} --seed {sd} --iters {it}"
            jobs += [
                {"name": f"fixmatch_{npc}_s{sd}", "args": f"--base fixmatch {c}"},
                {"name": f"flexmatch_{npc}_s{sd}", "args": f"--base flexmatch {c}"},
                {"name": f"meanteacher_{npc}_s{sd}", "args": f"--base meanteacher --lam_u 50 {c}"},
                {"name": f"pseudolabel_{npc}_s{sd}", "args": f"--base pseudolabel {c}"},
                {"name": f"sup_{npc}_s{sd}", "args": f"--base sup {c}"},
            ]
    write("fmnist-ssl-baselines", jobs, 11.0)


def degeneracy(seeds=(1, 2, 3)):
    """Prop. N2 on a real network: start from a supervised-only pair (teacher trained on
    labels only, student = copy), switch off FixMatch, student sees the same weak views as
    the teacher, raw (unnormalised) meta-gradient.  Optimistic feedback should leave the
    teacher (and the pseudo-labels) where they are; trust-region feedback should not."""
    jobs = []
    for sd in seeds:
        for fbk in ["opt", "pess", "exp"]:
            r = {"opt": 0.3, "pess": 0.3, "exp": 3.0}[fbk]
            jobs.append({"name": f"degen_{fbk}_s{sd}",
                         "args": f"--bilevel two --feedback {fbk} --radius {r} --warm_sup_iters 3000 --lam_u 0 "
                                 f"--stu_view weak --meta_norm 0 --lam_m 30 --n_per_class 10 --seed {sd} "
                                 f"--iters 4000 --eval_every 250 --log_every 50"})
    write("fmnist-ssl-degeneracy", jobs, 6.0)


def replication(it=20000):
    """Phase 2c: replication set decided after the main table (seeds 6-10, never looked at
    before): FixMatch / MPL / PBPL at 40 labels and FixMatch / PBPL at 100 labels.  Reported
    separately from seeds 1-5 and pooled.  Two kernels, longest jobs first."""
    a, b = [], []
    for sd in (6, 7, 8, 9, 10):
        a.append({"name": f"pbpl_4_s{sd}", "args": f"--bilevel two --feedback pess {PESS} --n_per_class 4 --seed {sd} --iters {it}"})
    for sd in (6, 7, 8, 9, 10):
        a.append({"name": f"mpl_4_s{sd}", "args": f"--bilevel two --feedback opt {OPT} --n_per_class 4 --seed {sd} --iters {it}"})
    write("fmnist-ssl-rep-a", a, 4.5)
    # rep-b was rewritten after ablate-a.  (1) The FixMatch grid (screen protocol: seed 0,
    # 40 labels, best validation accuracy) selects lam_u = 2, which beat the selected PBPL
    # config on the screen (val 78.04 vs 76.50): this tuned FixMatch is run on seeds 1-10.
    # (2) One-network look-ahead: the anchor fits the labels (L_l ~ 2e-3), i.e. the regime
    # where the optimistic feedback degenerates; the optimistic one-network control
    # (abl_selfopt) is the training-time test of the non-degeneracy argument.  (3) EBPL and
    # the two ablations of ablate-a are completed to seeds 1-5.  Jobs are in priority order;
    # the runner skips a job when the remaining budget is below 1.1 x est_min.  The 100-label
    # replication moves to rep-c.
    b = []
    for sd in (6, 7, 8, 9, 10):
        b.append({"name": f"fixmatch_4_s{sd}", "args": f"--base fixmatch --n_per_class 4 --seed {sd} --iters {it}", "est_min": 12})
    for sd in range(1, 11):
        b.append({"name": f"fmtuned_4_s{sd}", "args": f"{FMT} --n_per_class 4 --seed {sd} --iters {it}", "est_min": 12})
    for sd in (1, 2, 3, 4, 5):
        b.append({"name": f"abl_selfopt_4_s{sd}", "args": f"--bilevel self --feedback opt {OPT} --n_per_class 4 --seed {sd} --iters {it}", "est_min": 30})
    for sd in (4, 5):
        b.append({"name": f"abl_self_4_s{sd}", "args": f"--bilevel self --feedback pess {PESS} --n_per_class 4 --seed {sd} --iters {it}", "est_min": 40})
    for sd in (4, 5):
        b.append({"name": f"ebpl_4_s{sd}", "args": f"--bilevel two --feedback exp {EXP} --n_per_class 4 --seed {sd} --iters {it}", "est_min": 40})
    for sd in (4, 5):
        b.append({"name": f"abl_samfb_4_s{sd}", "args": f"--bilevel two --feedback sam {PESS} --n_per_class 4 --seed {sd} --iters {it}", "est_min": 31})
    write("fmnist-ssl-rep-b", b, float(os.environ.get("REPB_HOURS", 4.3)))
    # rep-c: 100-label replication (seeds 6-10) and tuned FixMatch at 100 labels
    c = []
    for sd in (6, 7, 8, 9, 10):
        c.append({"name": f"pbpl_10_s{sd}", "args": f"--bilevel two --feedback pess {PESS} --n_per_class 10 --seed {sd} --iters {it}", "est_min": 42})
    for sd in (6, 7, 8, 9, 10):
        c.append({"name": f"fixmatch_10_s{sd}", "args": f"--base fixmatch --n_per_class 10 --seed {sd} --iters {it}", "est_min": 12})
    for sd in range(1, 11):
        c.append({"name": f"fmtuned_10_s{sd}", "args": f"{FMT} --n_per_class 10 --seed {sd} --iters {it}", "est_min": 12})
    write("fmnist-ssl-rep-c", c, float(os.environ.get("REPC_HOURS", 4.0)))


def _done(dirs):
    done = set()
    for d in dirs:
        for root, _, fs in os.walk(d):
            for f in fs:
                if not f.endswith(".json"):
                    continue
                try:
                    r = json.load(open(os.path.join(root, f)))
                except Exception:
                    continue
                if "final" in r and "args" in r and not r.get("stopped_early") \
                        and r["final"].get("iter", 0) >= r["args"].get("iters", 1):
                    done.add(r["args"]["name"])
    return done


def wave3(it=20000):
    """Phase 3 (after the weekly quota reset), decided after rep-b.  With 10 seeds, PBPL, MPL,
    FixMatch and tuned FixMatch are within 1 point of each other at 40 labels, so this wave
    takes every remaining candidate of the family to the same seeds 1-10 (paired against the
    FixMatch and tuned-FixMatch runs that already exist) and adds one theory test:
    rep-c: EBPL and the one-network optimistic / pessimistic pair to 5 seeds, SAM-feedback to
           5 seeds, then EBPL on seeds 6-10.
    rep-d: fiber-SAM and labeled-loss SAM (the strongest controls) on seeds 4-10, the
           a screen (seed 0, 10k iterations, validation) of the one-network pair with the
           running-RMS normalisation of the main runs and at the exact meta-gradient scale
           (--meta_norm 0), where the optimistic feedback really vanishes as the anchor fits
           the labels (the normalisation rescales it back up), then the one-network pair on
           seeds 6-10.
           Jobs at the end of a queue are skipped by the runner when they no longer fit.
    python sweeps/make_sweeps.py wave3"""
    def job(name, a, sd, est):
        return {"name": f"{name}_4_s{sd}", "args": f"{a} --n_per_class 4 --seed {sd} --iters {it}", "est_min": est}
    EBPL = f"--bilevel two --feedback exp {EXP}"
    SELF_O = f"--bilevel self --feedback opt {OPT}"
    SELF_P = f"--bilevel self --feedback pess {PESS}"
    SAMFB = f"--bilevel two --feedback sam {PESS}"
    LABSAM, FIBSAM = "--lab_reg sam --radius 0.1", "--lab_reg fibersam --radius 0.1"
    c = [job("ebpl", EBPL, sd, 40) for sd in (4, 5)]
    c += [job("abl_selfopt", SELF_O, sd, 30) for sd in (3, 4, 5)]
    c += [job("abl_self", SELF_P, sd, 41) for sd in (4, 5)]
    c += [job("abl_samfb", SAMFB, sd, 31) for sd in (4, 5)]
    c += [job("ebpl", EBPL, sd, 40) for sd in range(6, 11)]
    c += [job("abl_samfb", SAMFB, sd, 31) for sd in range(6, 11)]
    write("fmnist-ssl-rep-c", c, float(os.environ.get("REPC_HOURS", 4.5)))
    d = [job("abl_fibersam", FIBSAM, sd, 25) for sd in range(4, 11)]
    d += [job("abl_labsam", LABSAM, sd, 15) for sd in range(4, 11)]
    # screen of the one-network pair (seed 0, 10k iterations, validation): running-RMS
    # normalisation (as in the main runs) and exact meta-gradient scale
    scr = "--n_per_class 4 --seed 0 --iters 10000"
    d.append({"name": "scr_selfopt_lm0.1", "args": f"{SELF_O} {scr}", "est_min": 16})
    d.append({"name": "scr_selfpess_lm0.1", "args": f"{SELF_P} {scr}", "est_min": 22})
    for lm in (3, 10):
        d.append({"name": f"scr_selfopt_raw_lm{lm}", "args": f"--bilevel self --feedback opt --lam_m {lm} --meta_norm 0 {scr}", "est_min": 16})
        d.append({"name": f"scr_selfpess_raw_lm{lm}", "args": f"--bilevel self --feedback pess --lam_m {lm} --radius 0.1 --meta_norm 0 {scr}", "est_min": 22})
    d += [job("abl_selfopt", SELF_O, sd, 30) for sd in range(6, 11)]
    d += [job("abl_self", SELF_P, sd, 41) for sd in range(6, 11)]
    write("fmnist-ssl-rep-d", d, float(os.environ.get("REPD_HOURS", 5.0)))


def wave4(it=20000):
    """Phase 4, decided after rep-c/rep-d.  The one-network pair is the only setting where the
    optimistic feedback really vanishes (anchor = teacher that fits the labels), so it is taken
    to the same seeds as everything else:
    rep-e: the normalised one-network PBPL on the seeds rep-d skipped (7-10) and SAM-feedback
           on seeds 6-10 (fills the 10-seed ablation table).
    rep-f: the one-network pair at the exact meta-gradient scale (--meta_norm 0), lam_m = 3,
           the best value of the rep-d screen for both rules (seed 0, validation), seeds 1-5,
           interleaved so that a budget cut drops whole pairs.
    python sweeps/make_sweeps.py wave4"""
    def job(name, a, sd, est):
        return {"name": f"{name}_4_s{sd}", "args": f"{a} --n_per_class 4 --seed {sd} --iters {it}", "est_min": est}
    SELF_P = f"--bilevel self --feedback pess {PESS}"
    SAMFB = f"--bilevel two --feedback sam {PESS}"
    e = [job("abl_self", SELF_P, sd, 41) for sd in range(7, 11)]
    e += [job("abl_samfb", SAMFB, sd, 31) for sd in range(6, 11)]
    write("fmnist-ssl-rep-e", e, float(os.environ.get("REPE_HOURS", 3.3)))
    RAW_O = "--bilevel self --feedback opt --lam_m 3 --meta_norm 0"
    RAW_P = "--bilevel self --feedback pess --lam_m 3 --radius 0.1 --meta_norm 0"
    f = []
    for sd in range(1, 6):
        f += [job("abl_selfraw", RAW_P, sd, 42), job("abl_selfoptraw", RAW_O, sd, 29)]
    write("fmnist-ssl-rep-f", f, float(os.environ.get("REPF_HOURS", 3.5)))


def pool_diag(it=20000):
    """Student-pool diagnostic (seed 0, 40 labels): k = 3 held-out students trained on the
    teacher's pseudo-labels, expected (mean) aggregation driving the teacher, all of min / mean /
    max / cvar feedbacks logged.  Two diversity levels: members differ only by their
    initialisation (same strong view), or also by independent strong views and a fixed 63% subset
    of X_u each.  Students are finetuned on the labels at the end (diagnostic); the teacher EMA is
    the deployed model.
    python sweeps/make_sweeps.py pool_diag"""
    POOL = f"--bilevel two --n_stu 3 --pool_agg mean {OPT} --stu_ft_iters 1000"
    jobs = [{"name": "pooldiag_same_4_s0", "args": f"{POOL} --pool_aug same --n_per_class 4 --seed 0 --iters {it}",
             "est_min": 75},
            {"name": "pooldiag_bag_4_s0",
             "args": f"{POOL} --pool_aug indep --pool_bag 0.63 --n_per_class 4 --seed 0 --iters {it}", "est_min": 75}]
    write("fmnist-ssl-pool-diag", jobs, float(os.environ.get("POOLDIAG_HOURS", 2.2)))


def remainder():
    """python sweeps/make_sweeps.py remainder <sweep.json> <new-slug> <hours> <results-dir> [...]
    Writes a sweep with the jobs of <sweep.json> that have no complete result in the given
    results directories (a run stopped by the budget guard counts as missing)."""
    src, slug, hours, dirs = sys.argv[2], sys.argv[3], float(sys.argv[4]), sys.argv[5:]
    sw = json.load(open(src))
    done = set()
    for d in dirs:
        for root, _, fs in os.walk(d):
            for f in fs:
                if not f.endswith(".json"):
                    continue
                try:
                    r = json.load(open(os.path.join(root, f)))
                except Exception:
                    continue
                if "final" in r and "args" in r and not r.get("stopped_early") \
                        and r["final"].get("iter", 0) >= r["args"].get("iters", 1):
                    done.add(r["args"]["name"])
    jobs = [j for j in sw["jobs"] if j["name"] not in done]
    write(slug, jobs, hours, sw.get("files", ()))


if __name__ == "__main__":
    globals()[sys.argv[1]]()
