#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem 4GB
#SBATCH -t 12:00:00

echo '    started at:' `date`

module purge

source ~/.bashrc
conda activate torchmd-net-pelaez

python colloids_run_smd.py run_smd.yaml

echo '    finished at:' `date`
