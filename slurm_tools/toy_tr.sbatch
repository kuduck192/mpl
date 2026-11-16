#!/bin/bash
# Two Moons trust-region runs on the CECS A5000 node (Slurm).  One array task = one line of the task table:
#   sbatch --array=0-$(( $(wc -l < slurm_tools/toy_tr_tasks.txt) - 1 ))%3 slurm_tools/toy_tr.sbatch
# 16 worker processes share the task's GPU (the fp64 linear algebra of the trust-region values); the
# protocol code of the MPL toys runs on their CPUs.
#SBATCH -p a5000
#SBATCH --gres=gpu:1
#SBATCH -c 17
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH -J toy-tr
#SBATCH -o /mnt/users/duc.lm3/mpl_toy/logs/%x_%A_%a.out
set -euo pipefail
cd /mnt/users/duc.lm3/mpl_toy/repo
read -r PROTO MODE BETA RADIUS SHARD NSHARD < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" slurm_tools/toy_tr_tasks.txt)
echo "task ${SLURM_ARRAY_TASK_ID}: ${PROTO} ${MODE} beta=${BETA} r=${RADIUS} shard ${SHARD}/${NSHARD} on $(hostname), GPU ${CUDA_VISIBLE_DEVICES:-none}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
~/venvs/rdq/bin/python -u checks/toy_two_moons_tr.py --protocol "${PROTO}" --mode "${MODE}" --beta "${BETA}" \
    --radius "${RADIUS}" --shard "${SHARD}/${NSHARD}" --procs 16 --tr_device cuda
