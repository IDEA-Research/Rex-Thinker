#!/bin/bash
#SBATCH --job-name=ab4
#SBATCH --output=sbatch_logs/ab4.txt
#SBATCH --partition=cvr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=600G
#SBATCH --gres=gpu:hgx:8
#SBATCH -w hgx039

echo "Start time: `date`"   #显示开始时间

echo "SLURM_JOB_ID: $SLURM_JOB_ID"   #显示作业号

echo "SLURM_NNODES: $SLURM_NNODES"   #显示节点数

echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"  #显示每节点任务数

echo "SLURM_NTASKS: $SLURM_NTASKS"   #显示总任务数

echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"   #显示作业分区

nvidia-smi
hostname


export OUTPUT_PATH="work_dirs/rexthinker/qwen25vl_7b_grpo_on_refcocog"
export EXP_NAME="grpo_tune"
export DEBUG_MODE="true"
export LOG_PATH="${OUTPUT_PATH}/log.txt"

# make sure the output directory exists
mkdir -p ${OUTPUT_PATH}

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH="IDEA-Research/Rex-Thinker-GRPO-7B"  # replace it with your local file path

python3 -m rexthinker.verl.trainer.main \
    config=rexthinker/scripts/config.yaml \
    data.train_files=data/Mountchicken/RefCOCOg-RexThinker-20k/refcocog_20000samples_rexthinker_grpo.parquet \
    worker.actor.model.model_path=${MODEL_PATH} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=8 \
    worker.actor.global_batch_size=64 \
    data.rollout_batch_size=64 \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.rollout.n=16 \
    trainer.experiment_name=${EXP_NAME} \
    trainer.save_checkpoint_path=${OUTPUT_PATH} \
    trainer.save_freq=50 \