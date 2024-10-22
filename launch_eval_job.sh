#!/bin/sh
# Node resource configurations
#SBATCH --job-name=train_am
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# for normal t4v2,t4v1,rtx6000,p100
# for high t4v2
# for deadline t4v2,t4v1,p100
#SBATCH --partition=t4v2,t4v1,p100

#SBATCH --gres=gpu:2
#SBATCH --account=deadline
#SBATCH --qos=deadline
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.err

# Append is important because otherwise preemption resets the file
#SBATCH --open-mode=append

echo `date`: Job $SLURM_JOB_ID is allocated resource

. ~/.bashrc
conda activate pytorch-env
python eval.py $*

echo `date`: "Job $SLURM_JOB_ID finished running, exit code: $?"

date=$(date '+%Y-%m-%d')
archive=$HOME/finished_jobs/$date/$SLURM_JOB_ID
mkdir -p $archive

cp ./logs/slurm-$SLURM_JOB_ID.out $archive/job.out
cp ./logs/slurm-$SLURM_JOB_ID.err $archive/job.err
