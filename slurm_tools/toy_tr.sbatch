#!/bin/bash
# Two Moons trust-region runs on the CECS A5000 node (Slurm).  One array task = one line of the task table:
#   sbatch --array=0-$(( $(wc -l < slurm_tools/toy_tr_tasks.txt) - 1 ))%2 slurm_tools/toy_tr.sbatch
# The students account allows 8 CPUs, 2 GPUs and 48 GB per user in total (GrpTRES), hence %2 x (4 CPUs, 1 GPU).
# 4 worker processes share the task's GPU (the fp64 linear algebra of the trust-region values); the
# protocol code of the MPL toys runs on their CPUs.
#SBATCH -p a5000
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH -t 23:00:00
#SBATCH -J toy-tr
#SBATCH -o /mnt/users/duc.lm3/mpl_toy/logs/%x_%A_%a.out
set -euo pipefail
cd /mnt/users/duc.lm3/mpl_toy/repo
TASKS=${TASKS:-slurm_tools/toy_tr_tasks.txt}  # another table: sbatch --export=ALL,TASKS=<file> ...
read -r PROTO MODE BETA RADIUS SHARD NSHARD < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${TASKS}")
echo "task ${SLURM_ARRAY_TASK_ID}: ${PROTO} ${MODE} beta=${BETA} r=${RADIUS} shard ${SHARD}/${NSHARD} on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-none}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
~/venvs/rdq/bin/python -u checks/toy_two_moons_tr.py --protocol "${PROTO}" --mode "${MODE}" --beta "${BETA}" \
    --radius "${RADIUS}" --shard "${SHARD}/${NSHARD}" --procs 4 --tr_device cuda
