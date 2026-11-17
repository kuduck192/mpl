#!/bin/bash
# MPL (mode opt: the original code of the three toys) rerun on the same node, CPU only.
#   sbatch slurm_tools/toy_tr_opt.sbatch
#SBATCH -p a5000
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH -t 04:00:00
#SBATCH -J toy-tr-opt
#SBATCH -o /mnt/users/duc.lm3/mpl_toy/logs/%x_%j.out
set -euo pipefail
cd /mnt/users/duc.lm3/mpl_toy/repo
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
for P in sec13 sec14 sec15; do
    ~/venvs/rdq/bin/python -u checks/toy_two_moons_tr.py --protocol "$P" --mode opt --procs 6
done
