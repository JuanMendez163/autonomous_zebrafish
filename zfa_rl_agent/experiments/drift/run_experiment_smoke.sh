#!/bin/bash -l
#SBATCH --job-name=drift-smoke
#SBATCH --partition=general
#SBATCH --time=0-06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --gres=gpu:A6000:1
#SBATCH --output=/data/group_data/neuroagents_lab/zfa_training/rl_training_logs/slurm/out/%x-%j.out
#SBATCH --error=/data/group_data/neuroagents_lab/zfa_training/rl_training_logs/slurm/err/%x-%j.err
#SBATCH --mail-type=END
#SBATCH --mail-user=rdkeller@andrew.cmu.edu

conda activate fishies
export PYTHONPATH="/Users/juanmendez/Kozachkov/autonomous_zebrafish:$PYTHONPATH"
export RAY_DEDUP_LOGS=1
export HYDRA_FULL_ERROR=1
export MUJOCO_GL="egl"
export MUJOCO_EGL_DEVICE_ID=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DISABLE_CACHE=true

#### Quick smoke-run parameters ####
CHECKPOINT_POLICY=true
CHECKPOING_WM=false

TOTAL_TIMESTEPS=1000000
CHECKPOINT_FREQ=100000
LEARNING_RATE=0.0003
BATCH_SIZE_MOD=128
N_STEPS=256
N_EPOCHS=3
VALUE_COEFF=0.5
N_TRAIN_ENVS=8

LP_HORIZON=0.99
MMM_HORIZON=0.99
BP_HORIZON=0.9

FORCE_MAG=0.0

IDM_SCALE=1.0
TASK_SCALE=1.0
ACTION_PENALTY_SCALE=0.0
WORLD_MODEL_TYPE="mlp"
IDM_TYPE="progress"
LOAD_DMC=false
USE_FLOW=false

NAME="drift-smoke-${FORCE_MAG}-${IDM_TYPE}-${WORLD_MODEL_TYPE}-nenvs-${N_TRAIN_ENVS}-steps-${TOTAL_TIMESTEPS}"

SCRATCH_DIR="/scratch/rdkeller"
SCRATCH_JOB_DIR="${SCRATCH_DIR}/${NAME}"
PERM_LOG_DIR=/data/group_data/neuroagents_lab/zfa_training/rl_training_logs/${NAME}

echo "====== QUICK SMOKE-RUN ENVIRONMENT REPORT ======"
echo "Node: ${SLURM_NODELIST}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "Scratch directory: ${SCRATCH_JOB_DIR}"
echo "Permanent directory: ${PERM_LOG_DIR}"
echo "=============================================="

echo "Creating scratch directory: ${SCRATCH_JOB_DIR}"
mkdir -p ${SCRATCH_JOB_DIR}
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create scratch directory on node ${SLURM_NODELIST}"
    exit 1
fi

echo "Creating permanent log directory: ${PERM_LOG_DIR}"
mkdir -p ${PERM_LOG_DIR}
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create permanent log directory"
    exit 1
fi

cleanup() {
    echo "Job is ending. Copying current results from scratch to permanent storage..."
    rsync -av --ignore-existing ${SCRATCH_JOB_DIR}/ ${PERM_LOG_DIR}/
    echo "Copy complete."
    echo "Removing scratch directory ${SCRATCH_JOB_DIR}..."
    rm -rf ${SCRATCH_JOB_DIR}
    echo "Cleanup complete."
}

trap cleanup EXIT INT TERM

python /Users/juanmendez/Kozachkov/autonomous_zebrafish/zfa_rl_agent/experiments/drift/run_experiment.py \
    name="$NAME" \
    total_timesteps="$TOTAL_TIMESTEPS" \
    checkpoint_save_freq="$CHECKPOINT_FREQ" \
    batch_mod="$BATCH_SIZE_MOD" \
    learning_rate="$LEARNING_RATE" \
    n_steps="$N_STEPS" \
    n_epochs="$N_EPOCHS" \
    vf_coef="$VALUE_COEFF" \
    ir_scale="$IDM_SCALE" \
    er_scale="$TASK_SCALE" \
    reward_type="$IDM_TYPE" \
    ap_scale="$ACTION_PENALTY_SCALE" \
    use_flow="$USE_FLOW" \
    world_model_class="$WORLD_MODEL_TYPE" \
    n_train_envs="$N_TRAIN_ENVS" \
    parallel=true \
    log_dir="${SCRATCH_JOB_DIR}" \
    job_id="${SLURM_JOB_ID}" \
    mmm_progress_horizon="$MMM_HORIZON" \
    learning_progress_horizon="$LP_HORIZON" \
    load_dmc_agent="$LOAD_DMC" \
    checkpointing="$CHECKPOINT_POLICY" \
    wm_checkpointing="$CHECKPOINT_WM" \
    cycle_horizon="$BP_HORIZON" \
    drift_force="$FORCE_MAG"

echo "Smoke training completed."
