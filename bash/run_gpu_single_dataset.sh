#!/bin/bash
# SBATCH script to run baseline + sycophancy prompts on a single dataset using 2 GPUs.

#SBATCH -J gpu-single-dataset
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o logs/qwen2_audio_single_%j.out
#SBATCH -e logs/qwen2_audio_single_%j.err

set -euo pipefail

source ~/.bashrc
conda activate alm
cd /home/junchi.yao/ICML2026-ALM

mkdir -p logs

DEFAULT_MODEL="Qwen/Qwen2-Audio-7B-Instruct"
DEFAULT_DATASET="mmlu"
# Optional positional args: 1) model, 2) dataset; env DATASET can also override.
MODEL="${1:-${DEFAULT_MODEL}}"
DATASET="${2:-${DATASET:-${DEFAULT_DATASET}}}"
LIMIT=9999
NUM_GPUS=2

PROMPTS=(
  "bias_feedback strong"
  "bias_feedback medium"
  "bias_feedback low"
  "are_you_sure"
  "answer_sycophancy"
  "mimicry_sycophancy"
)

echo "[$(date)] Starting runs with model=${MODEL}, dataset=${DATASET}, limit=${LIMIT}, GPUs=${NUM_GPUS}"

echo "[$(date)] Dataset=${DATASET} | baseline"
python code/audio_eval.py \
  --dataset "${DATASET}" \
  --limit "${LIMIT}" \
  --model "${MODEL}" \
  --num-gpus "${NUM_GPUS}"

base_log="result/baseline/$(basename "${MODEL}")/${DATASET}_baseline_limit${LIMIT}.log"

for entry in "${PROMPTS[@]}"; do
  read -r prompt variant <<<"${entry}"
  echo "[$(date)] Dataset=${DATASET} | prompt=${prompt} | variant=${variant:-auto}"
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

echo "[$(date)] Dataset=${DATASET} completed."
