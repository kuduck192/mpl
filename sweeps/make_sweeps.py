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


if __name__ == "__main__":
    globals()[sys.argv[1]]()
