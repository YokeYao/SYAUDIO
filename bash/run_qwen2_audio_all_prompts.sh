#!/bin/bash
# SBATCH script to run Qwen2-Audio-7B-Instruct baseline + 6 sycophancy prompts on four datasets using 4 GPUs.

#SBATCH -J qwen2-audio-split
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o logs/qwen2_audio_%A_%a.out
#SBATCH -e logs/qwen2_audio_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail

source ~/.bashrc
conda activate alm
cd /home/junchi.yao/ICML2026-ALM

mkdir -p logs

MODEL="Qwen/Qwen2-Audio-7B-Instruct"
LIMIT=9999
NUM_GPUS=2
# Two-job split to ease queue wait; SLURM_ARRAY_TASK_ID picks the group.
# 0: mmar + mmau, 1: gsm8k. (mmlu omitted as requested.)
DATASET_GROUPS=(
  # "mmar mmau"
  "mmlu"
)
# Fallback to group 0 if array index missing.
GROUP_IDX="${SLURM_ARRAY_TASK_ID:-0}"
IFS=' ' read -r -a DATASETS <<<"${DATASET_GROUPS[$GROUP_IDX]}"
# prompt entries: "<prompt_key> [variant]" for sycophancy.py
PROMPTS=(
  "bias_feedback strong"
  "bias_feedback medium"
  "bias_feedback low"
  "are_you_sure"
  "answer_sycophancy"
  "mimicry_sycophancy"
)

echo "[$(date)] Starting runs with model=${MODEL}, limit=${LIMIT}, GPUs=${NUM_GPUS}"

for ds in "${DATASETS[@]}"; do
  echo "[$(date)] Dataset=${ds} | baseline"
  python code/audio_eval.py \
    --dataset "${ds}" \
    --limit "${LIMIT}" \
    --model "${MODEL}" \
    --num-gpus "${NUM_GPUS}"

  base_log="result/baseline/$(basename "${MODEL}")/${ds}_baseline_limit${LIMIT}.log"

  for entry in "${PROMPTS[@]}"; do
    read -r prompt variant <<<"${entry}"
    echo "[$(date)] Dataset=${ds} | prompt=${prompt} | variant=${variant:-auto}"
    if [[ -z "${variant:-}" ]]; then
      python code/sycophancy.py \
        --prompt "${prompt}" \
        --baseline-log "${base_log}" \
        --model "${MODEL}" \
        --num-gpus "${NUM_GPUS}"
    else
      python code/sycophancy.py \
        --prompt "${prompt}" \
        --variant "${variant}" \
        --baseline-log "${base_log}" \
        --model "${MODEL}" \
        --num-gpus "${NUM_GPUS}"
    fi
  done
done

echo "[$(date)] All runs completed."
