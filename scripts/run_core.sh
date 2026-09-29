#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
model="${1:-Qwen/Qwen2-Audio-7B-Instruct}"
workers="${NUM_GPUS:-1}"
for dataset in mmau mmar gsm8k mmlu; do
  python code/audio_eval.py --dataset "$dataset" --model "$model" --limit 9999 --num-gpus "$workers"
  model_name="${model##*/}"
  baseline="${SYAUDIO_RESULT_ROOT:-result}/baseline/${model_name}/${dataset}_baseline_limit9999.log"
  for variant in low medium strong; do
    python code/sycophancy.py --model "$model" --baseline-log "$baseline" --prompt bias_feedback --variant "$variant" --num-gpus "$workers"
  done
  for prompt in are_you_sure answer_sycophancy mimicry_sycophancy; do
    python code/sycophancy.py --model "$model" --baseline-log "$baseline" --prompt "$prompt" --num-gpus "$workers"
  done
done
