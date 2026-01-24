#!/bin/bash
# Run audio-only sycophancy prompts on four datasets using API/open-source models.

#SBATCH -J api-audio-sycophancy
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o logs/audio_sycophancy_api_%A_%a.out
#SBATCH -e logs/audio_sycophancy_api_%A_%a.err
#SBATCH --array=0-1

set -eo pipefail

# Bashrc/conda hooks may read unset vars; delay `set -u` until after activation.
set +u
source ~/.bashrc
export CONDA_BACKUP_CXX="${CONDA_BACKUP_CXX-}"
conda activate alm
set -u
export PYTHONNOUSERSITE=1
cd /home/junchi.yao/ICML2026-ALM

mkdir -p logs

DEFAULT_MODEL="gpt-4o-mini-audio-preview"
MODEL="${1:-${DEFAULT_MODEL}}"  # optional positional arg overrides default model
BASELINE_LIMIT=9999
SYCO_LIMIT=100
NUM_WORKERS="${2:-4}"  # num-gpus flag is repurposed for API parallelism
RUN_BASELINE="${RUN_BASELINE:-auto}"  # auto|run|skip
CACHE_LAYOUT="${CACHE_LAYOUT:-split}"  # split|legacy

# Two-job split to ease queue wait; SLURM_ARRAY_TASK_ID picks the group.
# 0: mmar + mmau, 1: gsm8k + mmlu
DATASET_GROUPS=(
  "mmar mmau"
  "gsm8k mmlu"
)
GROUP_IDX="${SLURM_ARRAY_TASK_ID:-0}"
# Guard against out-of-range array index
if (( GROUP_IDX < 0 || GROUP_IDX >= ${#DATASET_GROUPS[@]} )); then
  echo "Invalid group index ${GROUP_IDX}; defaulting to 0" >&2
  GROUP_IDX=0
fi
IFS=' ' read -r -a DATASETS <<<"${DATASET_GROUPS[$GROUP_IDX]}"
# prompt entries: "<prompt_key> [variant]" for audio_sycophancy.py
PROMPTS=(
  "bias_feedback strong"
  "bias_feedback medium"
  "bias_feedback low"
  "are_you_sure"
  "answer_sycophancy"
  "mimicry_sycophancy"
)

usage() {
  cat <<'USAGE'
Usage:
  RUN_BASELINE=auto bash run_api_all_prompts_audio_sycophancy.sh <model> [workers]

Notes:
  - Model names should match directories under result/baseline/.
  - RUN_BASELINE=auto runs baseline only if missing; run forces it; skip requires it.

Examples (two tmux sessions):
  RUN_BASELINE=skip bash run_api_all_prompts_audio_sycophancy.sh gpt-4o-mini-audio-preview 4
  RUN_BASELINE=skip bash run_api_all_prompts_audio_sycophancy.sh vertex-gemini-2.5-flash-lite-preview-09-2025-nothinking 4
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

echo "[$(date)] Starting audio-only sycophancy runs with model=${MODEL}, baseline_limit=${BASELINE_LIMIT}, syco_limit=${SYCO_LIMIT}, workers=${NUM_WORKERS}, run_baseline=${RUN_BASELINE}"

for ds in "${DATASETS[@]}"; do
  base_log="result/baseline/$(basename "${MODEL}")/${ds}_baseline_limit${BASELINE_LIMIT}.log"
  if [[ -f "${base_log}" ]]; then
    if [[ "${RUN_BASELINE}" == "run" ]]; then
      echo "[$(date)] Dataset=${ds} | baseline (forced)"
      python code/audio_eval.py \
        --dataset "${ds}" \
        --limit "${BASELINE_LIMIT}" \
        --model "${MODEL}" \
        --num-gpus "${NUM_WORKERS}"
    elif [[ "${RUN_BASELINE}" == "skip" ]]; then
      echo "[$(date)] Dataset=${ds} | baseline skipped (RUN_BASELINE=skip)"
    else
      echo "[$(date)] Dataset=${ds} | baseline (cached)"
    fi
  else
    if [[ "${RUN_BASELINE}" == "skip" ]]; then
      echo "Missing baseline log: ${base_log}" >&2
      exit 1
    fi
    echo "[$(date)] Dataset=${ds} | baseline"
    python code/audio_eval.py \
      --dataset "${ds}" \
      --limit "${BASELINE_LIMIT}" \
      --model "${MODEL}" \
      --num-gpus "${NUM_WORKERS}"
  fi

  for entry in "${PROMPTS[@]}"; do
    read -r prompt variant <<<"${entry}"
    echo "[$(date)] Dataset=${ds} | prompt=${prompt} | variant=${variant:-auto}"
    if [[ -z "${variant:-}" ]]; then
      python code/ablation_code/audio_sycophancy.py \
        --prompt "${prompt}" \
        --baseline-log "${base_log}" \
        --model "${MODEL}" \
        --num-gpus "${NUM_WORKERS}" \
        --limit "${SYCO_LIMIT}" \
        --cache-layout "${CACHE_LAYOUT}" \
        --dump-prompt-text
    else
      python code/ablation_code/audio_sycophancy.py \
        --prompt "${prompt}" \
        --variant "${variant}" \
        --baseline-log "${base_log}" \
        --model "${MODEL}" \
        --num-gpus "${NUM_WORKERS}" \
        --limit "${SYCO_LIMIT}" \
        --cache-layout "${CACHE_LAYOUT}" \
        --dump-prompt-text
    fi
  done
done

echo "[$(date)] All audio-only sycophancy runs completed."
