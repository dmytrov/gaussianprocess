#!/bin/bash

# Marc2 launch script.

#$ -S /bin/bash
#$ -cwd
#$ -e /home/velychko/sge-temp
#$ -o /home/velychko/sge-temp

# max run time
#$ -l h_rt=5:00:00

# 2G RAM per CPU
#$ -l h_vmem=2G

# SMP, 8 CPU slots per task
#$ -pe smp* 8

# Task array
#$ -t 1-20

# Email
#$ -m aes
#$ -M velychko@staff.uni-marburg.de


# load the proper module
. /etc/profile.d/modules.sh
module load tools/python-2.7

# echo "Hello from $(hostname)" > hello.sh.log.$SGE_TASK_ID.txt
source ~/venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:${HOME}/projects/vCGPDM/code/"
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
if [[ -z "${DISPLAY}" ]]; then
    echo DISPLAY is not set, using "Agg" rendering backend.
    export MPLBACKEND="Agg"
fi

python learn_cgpdm.py --i $SGE_TASK_ID
