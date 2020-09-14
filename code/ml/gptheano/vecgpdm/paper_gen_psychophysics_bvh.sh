#!/bin/bash

# Marc2 launch script.

#$ -S /bin/bash
#$ -cwd
#$ -e /home/velychko/sge-temp
#$ -o /home/velychko/sge-temp

# max run time
#$ -l h_rt=10:00:00

# 2G RAM per CPU
#$ -l h_vmem=2G

# SMP, 8 CPU slots per task. 16 is the recommended max.
#$ -pe smp* 8

# Task array, 1-ndyn*nlvm
#$ -t 1-81

# Email
#$ -m aes
#$ -M velychko@staff.uni-marburg.de

function run_model()
{
    local dyn=$1
    local lvm=$2
    if [[ -z "${DISPLAY}" ]]; then
        echo DISPLAY is not set, using "Agg" rendering backend.
        export MPLBACKEND="Agg"
    fi
    echo -e "DYN IPs: $dyn \t LVM IPs: $lvm"
    python paper_gen_psychophysics_bvh.py \
            --dir ../../../../log/psychophysics \
            --dataset 0 \
            --dyn $dyn \
            --lvm $lvm
            
}

# Lists of model parameters
dyns=(2 3 4 5 10 15 20 25 30)
lvms=(2 3 4 5 10 15 20 25 30)
ndyn=${#dyns[*]}
nlvm=${#lvms[*]}

if [ -n "${SGE_TASK_ID+set}" ]; then
    # Running on SGE cluster

    # Load the proper module
    . /etc/profile.d/modules.sh
    module unload gcc/6.3.0
    module load gcc/7.2.0
    module load lalibs/openblas/gcc-7.2.0/0.2.20
    module load tools/python-2.7

    source ~/venv/bin/activate
    export PYTHONPATH="${PYTHONPATH}:${HOME}/projects/vCGPDM/code/"
    export OPENBLAS_NUM_THREADS=8
    export OMP_NUM_THREADS=8    

    # Use local node file system for compilation
    export THEANO_FLAGS=base_compiledir=${HPC_LOCAL}/${RANDOM}

    # Zero-based
    TASK_ID=$(($SGE_TASK_ID - 1))
    
    idyn=$(($TASK_ID % $ndyn))
    ilvm=$(($TASK_ID / $ndyn))
    dyn=${dyns[$idyn]}
    lvm=${lvms[$ilvm]}
    
    # Run the script
    run_model ${dyn} ${lvm}
else
    # Running on a local computer
    # Loop through all the model parameters
    END=$(($ndyn * $nlvm))
    for ((TASK_ID=0;TASK_ID<END;TASK_ID++)); do
        idyn=$(($TASK_ID % $ndyn))
        ilvm=$(($TASK_ID / $ndyn))
        dyn=${dyns[$idyn]}
        lvm=${lvms[$ilvm]}
        
        # Run the script
        run_model ${dyn} ${lvm}
    done
fi
