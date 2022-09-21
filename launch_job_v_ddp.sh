#!/bin/sh
# Node resource configurations
#SBATCH --job-name=train_am
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8

# for normal t4v2,t4v1,rtx6000,p100
# for high t4v2
# for deadline t4v2,t4v1,p100
#SBATCH --partition=t4v2

#SBATCH --gres=gpu:4
#SBATC --account=deadline
#SBATCH --qos=high
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.err
# Append is important because otherwise preemption resets the file
#SBATCH --open-mode=append

echo `date`: Job $SLURM_JOB_ID is allocated resource

# these commands don't need to run for all workers, put them here
MAIN_HOST=`hostname -s`
# this is the current host
export MASTER_ADDR=$MAIN_HOST
# pick a random available port
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"

# NCCL options
# This is needed to print debug info from NCCL, can be removed if all goes well
export NCCL_DEBUG=INFO
# This is needed to avoid NCCL to use ifiniband, which the cluster does not have
export NCCL_IB_DISABLE=1
# This is to tell NCCL to use bond interface for network communication
if [[ "${SLURM_JOB_PARTITION}" == "p100" ]] || [[ "${SLURM_JOB_PARTITION}" == "t4v1" ]] || \
   [[ "${SLURM_JOB_PARTITION}" == "t4v2" ]] || [[ "${SLURM_JOB_PARTITION}" == "rtx6000" ]]; then
    echo export NCCL_SOCKET_IFNAME=bond0 on ${SLURM_JOB_PARTITION}
    export NCCL_SOCKET_IFNAME=bond0
fi

# note when number of tasks is greater than 1, srun is needed to launch
# all tasks with the same command
# you should also be careful about parameter expansion, make sure
# they are not expanded here

ulimit -n 5000

mkdir -p workdirs/workdir_${SLURM_JOB_ID}
cp ddp_worker.sh  workdirs/workdir_${SLURM_JOB_ID}/
cd workdirs/workdir_${SLURM_JOB_ID}

# Checkpointing, never forget to do this
# ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} checkpoints
# touch checkpoints/DELAYPURGE

# the recommendation is to keep erything that defines the workload itself in a separate script
# bash run_train.sh
ln -sfn /checkpoint/${USER}/${SLURM_JOB_ID} $PWD/checkpoint_${SLURM_JOB_ID}
touch /checkpoint/${USER}/${SLURM_JOB_ID}/DELAYPURGE

# this will execute "number of tasks" times in parallel, each with
# slightly different env variables for DDP training
/opt/slurm/bin/srun --mem=16G bash -c \
    "bash ddp_worker.sh $* >> log_for_\${SLURM_JOB_ID}_node_\${SLURM_PROCID}.log 2>&1"

echo `date`: "Job $SLURM_JOB_ID finished running, exit code: $?"

date=$(date '+%Y-%m-%d')
archive=$HOME/finished_jobs/$date/$SLURM_JOB_ID
mkdir -p $archive

cp ./logs/slurm-$SLURM_JOB_ID.out $archive/job.out
cp ./logs/slurm-$SLURM_JOB_ID.err $archive/job.err
