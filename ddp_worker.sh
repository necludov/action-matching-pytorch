#!/bin/bash

echo Hostname: `hostname -s`
echo Node Rank ${SLURM_PROCID}

#env
. /ssd003/home/${USER}/.bashrc
conda activate /ssd003/home/${USER}/condaenvs/pytorch-env

if [[ ${SLURM_PROCID} != '0' ]]
then
    echo waiting for 10 seconds for main worker to start first
    sleep 10
fi

env

NUM_GPUs=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`

cmd="python -m torch.distributed.run \
    --nnodes ${SLURM_NNODES} \
    --node_rank ${SLURM_NODEID} \
    --nproc_per_node ${NUM_GPUs} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
        ../../launch_ddp.py  \
        --checkpoint_dir $PWD/checkpoint_${SLURM_JOB_ID} \
        --dataset cifar
        $* \
    "

echo $cmd
eval $cmd
Footer
