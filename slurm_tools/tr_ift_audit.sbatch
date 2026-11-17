#!/bin/bash
# Audit of the TR-PBPL solver along the IFT runs (checks/tr_ift_audit.py): the run's warm solves against long
# multi-start solves, then the whole IFT trajectory with a strong solver, 8 students each.
#   sbatch slurm_tools/tr_ift_audit.sbatch
# Shares the per-user quota (8 CPUs, 2 GPUs) with the toy_tr array: throttle the array to one task meanwhile,
#   scontrol update JobId=<array job> ArrayTaskThrottle=1
#SBATCH -p a5000
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH -t 12:00:00
#SBATCH -J tr-audit
#SBATCH -o /mnt/users/duc.lm3/mpl_toy/logs/%x_%j.out
set -euo pipefail
cd /mnt/users/duc.lm3/mpl_toy/repo
echo "$(date) on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-none}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
~/venvs/rdq/bin/python -u checks/tr_ift_audit.py --procs 4 --tr_device cuda --out checks/tr_ift_audit.json
echo "$(date) audit done"
~/venvs/rdq/bin/python -u checks/tr_ift_audit.py --strong 1 --procs 4 --tr_device cuda --out checks/tr_ift_strong.json
echo "$(date) strong done"
~/venvs/rdq/bin/python -u checks/tr_ift_audit.py --summary checks/tr_ift_audit.json checks/tr_ift_strong.json
