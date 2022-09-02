#!/bin/sh
# Node resource configurations
#SBATCH --job-name=train_am
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.err

# Append is important because otherwise preemption resets the file
#SBATCH --open-mode=append

echo `date`: Job $SLURM_JOB_ID is allocated resource

# the recommendation is to keep erything that defines the workload itself in a separate script
# bash run_train.sh
mkdir -p $PWD/checkpoint/${SLURM_JOB_ID}

. /h/${USER}/.bashrc
conda activate /h/${USER}/condaenvs/pytorch-env
python launch.py --checkpoint_dir $PWD/checkpoint/${SLURM_JOB_ID} --dataset mnist

echo `date`: "Job $SLURM_JOB_ID finished running, exit code: $?"

date=$(date '+%Y-%m-%d')
archive=$HOME/finished_jobs/$date/$SLURM_JOB_ID
mkdir -p $archive

cp ./$SLURM_JOB_ID.out $archive/job.out
cp ./$SLURM_JOB_ID.err $archive/job.err
