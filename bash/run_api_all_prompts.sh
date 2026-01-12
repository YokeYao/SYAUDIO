#!/bin/bash
# Run baseline + sycophancy prompts on all datasets via API model (no GPU required).

#SBATCH -J api-all-prompts
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o logs/gpt_audio_mini_%A_%a.out
#SBATCH -e logs/gpt_audio_mini_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail

source ~/.bashrc
conda activate alm
cd /home/junchi.yao/ICML2026-ALM

DEFAULT_MODEL="gpt-audio-mini"
MODEL="${1:-${DEFAULT_MODEL}}"  # optional positional arg overrides default model
LIMIT=9999
WORKERS=4   # number of API workers (num-gpus flag is repurposed for API parallelism)

DATASETS=(mmar gsm8k mmlu)
# prompt entries: "<prompt_key> [variant]"
PROMPTS=(
  # "bias_feedback strong"
  # "bias_feedback medium"
  # "bias_feedback low"
  "are_you_sure"
  "answer_sycophancy"
  "mimicry_sycophancy"
)

echo "[$(date)] Starting API runs with model=${MODEL}, limit=${LIMIT}, workers=${WORKERS}"

for ds in "${DATASETS[@]}"; do
  echo "[$(date)] Dataset=${ds} | baseline"
  python code/audio_eval.py \
    --dataset "${ds}" \
    --limit "${LIMIT}" \
    --model "${MODEL}" \
    --num-gpus "${WORKERS}"

  base_log="result/baseline/$(basename "${MODEL}")/${ds}_baseline_limit${LIMIT}.log"

  for entry in "${PROMPTS[@]}"; do
    read -r prompt variant <<<"${entry}"
    echo "[$(date)] Dataset=${ds} | prompt=${prompt} | variant=${variant:-auto}"
    if [[ -z "${variant:-}" ]]; then
      python code/sycophancy.py \
        --prompt "${prompt}" \
        --baseline-log "${base_log}" \
        --model "${MODEL}" \
        --num-gpus "${WORKERS}"
    else
      python code/sycophancy.py \
        --prompt "${prompt}" \
        --variant "${variant}" \
        --baseline-log "${base_log}" \
        --model "${MODEL}" \
        --num-gpus "${WORKERS}"
    fi
  done
done

echo "[$(date)] API runs completed."