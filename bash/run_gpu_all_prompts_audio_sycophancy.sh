#!/bin/bash
# SBATCH script to run audio-only sycophancy prompts on four datasets using existing baseline logs.

#SBATCH -J gpu-audio-sycophancy
#SBATCH -p cscc-gpu-p
#SBATCH -q cscc-gpu-qos
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o logs/audio_sycophancy_%A_%a.out
#SBATCH -e logs/audio_sycophancy_%A_%a.err
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

DEFAULT_MODEL=""Qwen/Qwen2-Audio-7B-Instruct""
MODEL="${1:-${DEFAULT_MODEL}}"  # optional positional arg overrides default model
BASELINE_LIMIT=9999
SYCO_LIMIT=100
NUM_GPUS=2
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

echo "[$(date)] Starting audio-only sycophancy runs with model=${MODEL}, baseline_limit=${BASELINE_LIMIT}, syco_limit=${SYCO_LIMIT}, GPUs=${NUM_GPUS}"

for ds in "${DATASETS[@]}"; do
  base_log="result/baseline/$(basename "${MODEL}")/${ds}_baseline_limit${BASELINE_LIMIT}.log"
  if [[ ! -f "${base_log}" ]]; then
    echo "Missing baseline log: ${base_log}" >&2
    exit 1
  fi

  for entry in "${PROMPTS[@]}"; do
    read -r prompt variant <<<"${entry}"
    echo "[$(date)] Dataset=${ds} | prompt=${prompt} | variant=${variant:-auto}"
    if [[ -z "${variant:-}" ]]; then
      python code/ablation_code/audio_sycophancy.py \
        --prompt "${prompt}" \
        --baseline-log "${base_log}" \
        --model "${MODEL}" \
        --num-gpus "${NUM_GPUS}" \
        --limit "${SYCO_LIMIT}" \
        --cache-layout "${CACHE_LAYOUT}" \
        --dump-prompt-text
    else
      python code/ablation_code/audio_sycophancy.py \
        --prompt "${prompt}" \
        --variant "${variant}" \
        --baseline-log "${base_log}" \
        --model "${MODEL}" \
        --num-gpus "${NUM_GPUS}"\
        --limit "${SYCO_LIMIT}" \
        --cache-layout "${CACHE_LAYOUT}" \
        --dump-prompt-text
    fi
  done
done

echo "[$(date)] All audio-only sycophancy runs completed."
