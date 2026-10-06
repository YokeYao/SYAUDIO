# Reproducing the evaluation

## Installation and download options

Use Python 3.10 and FFmpeg. Local-model inference needs a CUDA-compatible PyTorch build. `requirements.txt` records dependency versions from the evaluation environment, including Transformers and `huggingface_hub`.

To download data without installing model-inference dependencies, run these commands from the cloned repository:

```bash
pip install huggingface_hub
python scripts/download_syaudio.py
```

Use `python scripts/download_syaudio.py --groups all` to include the human-validation recordings. The installer verifies file hashes and preserves differing local copies. The default dataset revision is pinned; use `--revision <commit-sha>` to select another revision or `--destination /path/to/new-directory` for a separate installation. Data are installed under `<destination>/benchmark/`.

## Data layout

Dataset audio and annotations are downloaded from Hugging Face into `benchmark/`, rather than duplicated in GitHub. Use `python scripts/download_syaudio.py --groups all` to include the human-validation recordings. The local audio-only cues `benchmark/reject.wav` and `benchmark/not_sure.wav` remain bundled because they are used by `code/audio_cue_only_sycophancy.py`.

After `python scripts/download_syaudio.py`, the dataset uses the same paths as the original evaluator:

| Key | Annotations | Audio root | Count |
|---|---|---|---:|
| `mmau` | `benchmark/MMAU/mmau-test-mini.json` | `benchmark/MMAU/` | 1,000 |
| `mmar` | `benchmark/MMAR/MMAR-meta.json` | `benchmark/MMAR/` | 1,000 |
| `gsm8k` | `benchmark/GSM8K/test_mcq.jsonl` | `benchmark/GSM8K/` | 1,319 |
| `mmlu` | `benchmark/MMLU/mmlu_combined.jsonl` | `benchmark/MMLU/` | 1,000 |

Use `SYAUDIO_DATA_ROOT=/path/to/benchmark` and `SYAUDIO_RESULT_ROOT=/path/to/results` to relocate inputs and outputs. Baseline and core follow-up scripts also accept `--data` and `--audio-root` for an explicit annotation file and audio root. Do not mix histories from different annotation variants or model IDs.

## Core protocol

A baseline is required before follow-up evaluation. `--limit 9999` selects every row in each core dataset; `--limit 10` is a small inference smoke run. Baseline logs record stable IDs, raw answers, parsed answer letters and correctness. Follow-ups read those histories, preserve the original audio and question, and apply the selected user cue.

```bash
python code/audio_eval.py --dataset gsm8k --limit 9999 --model Qwen/Qwen2-Audio-7B-Instruct
baseline=result/baseline/Qwen2-Audio-7B-Instruct/gsm8k_baseline_limit9999.log
python code/sycophancy.py --baseline-log "$baseline" --prompt bias_feedback --variant low
python code/sycophancy.py --baseline-log "$baseline" --prompt bias_feedback --variant medium
python code/sycophancy.py --baseline-log "$baseline" --prompt bias_feedback --variant strong
python code/sycophancy.py --baseline-log "$baseline" --prompt are_you_sure
python code/sycophancy.py --baseline-log "$baseline" --prompt answer_sycophancy
python code/sycophancy.py --baseline-log "$baseline" --prompt mimicry_sycophancy
```

Pass the same `--model` to baseline and follow-up commands. `scripts/run_core.sh` automates these conditions for all datasets. Local inference supports Qwen2-Audio, Qwen2.5-Omni and the optional Transformers Audio-Flamingo adapter. API models must be supported by the configured OpenAI-compatible endpoint. Multi-GPU inference uses `--num-gpus`; in follow-up API mode this argument controls worker concurrency.

The original evaluators report MSS/CRS counters. MSS is conditioned on initially correct examples; CRS is conditioned on initially incorrect examples. Do not average raw counts across different denominators or treat the supplemental human recordings as additional core examples. This release omits the workspace's standalone confidence-interval and metric-comparison scripts.

The runners automatically resume existing logs. Use a fresh output root when changing model, data, prompt setting or generation parameters; reusing the same output path can skip previously logged sample IDs. Default baseline/follow-up generation limits remain those in the original scripts (2,048 and 1,024 tokens respectively); override with `--max-gen-len` when reproducing a specifically configured experiment.

## Human versus synthetic speech

Core data and human-validation audio and annotations are available from `YokyYao/SYAUDIO` on Hugging Face. `--groups all` downloads and verifies both, including into a fresh destination; matching files from an earlier download are verified and reused.

`TTS-Annie`, `TTS-Danielle` and `TTS-Junchi` contain **human readings**. They cover 100 aligned GSM8K MCQ questions (IDs 00000–00099). Their corresponding `test_mcq_<speaker>_100.jsonl` files preserve question and choice order while changing the audio path.

```bash
python scripts/download_syaudio.py --groups gsm8k human_annie human_danielle human_junchi
python code/audio_eval.py --dataset gsm8k --limit 100 \
  --data benchmark/GSM8K/test_mcq_junchi_100.jsonl \
  --audio-root benchmark/GSM8K \
  --log-path result/human/junchi/gsm8k_baseline_limit100.log
python code/sycophancy.py \
  --baseline-log result/human/junchi/gsm8k_baseline_limit100.log \
  --data benchmark/GSM8K/test_mcq_junchi_100.jsonl \
  --audio-root benchmark/GSM8K --prompt are_you_sure \
  --out-dir result/human/junchi/followup
```

Change both the annotation filename and output directory for each speaker. For the aligned synthetic condition, use the first 100 rows of `benchmark/GSM8K/test_mcq.jsonl` and a different output directory.

## Audio prompts and acoustic conditions

The audio-prompt evaluator synthesizes user follow-ups; it can call a TTS provider and incur API charges. 

```bash
python code/ablation_code/audio_sycophancy.py \
  --baseline-log result/baseline/Qwen2-Audio-7B-Instruct/gsm8k_baseline_limit9999.log \
  --prompt are_you_sure --model Qwen/Qwen2-Audio-7B-Instruct \
  --tts-backend openai --tts-model tts-1-hd --tts-voice alloy
```

`--question-format audio` is the default audio-prompt protocol; `--question-format text` keeps the question/history in text and synthesizes only the follow-up. Consult `--help` for TTS format, sample rate, silence and cache options. TTS regeneration need not reproduce historical audio bytes exactly.

Reverberation, channel, nonlinear and speech-rate perturbations can be generated without external noise recordings:

```bash
python code/ablation_code/complex_acoustic_noise.py \
  --dataset GSM8K --scenario audio_speech_rate --volume 200
python code/ablation_code/noise_sycophancy.py \
  --baseline-log result/baseline/Qwen2-Audio-7B-Instruct/gsm8k_baseline_limit9999.log \
  --prompt are_you_sure --scenario audio_speech_rate --volume 200
```

The original acoustic follow-up evaluator intentionally restricts evaluation to GSM8K IDs **01001–01100**, or the first **100 MMLU IDs**; this is not the complete core benchmark. Speech-rate severity 200 corresponds to the original 1.5× rate preset. The available preset values are implemented in `complex_acoustic_noise.py` and should not be reinterpreted as decibels.

Noise mixing uses the original background clips included in this repository through Git LFS: `audio_samples/cafe_chatter.mp3`, `audio_samples/forest_ambience.mp3` and `audio_samples/crowded_street.mp3`. Install Git LFS and download the audio files before running noise mixing:

```bash
git lfs install
git lfs pull --include="audio_samples/cafe_chatter.mp3,audio_samples/forest_ambience.mp3,audio_samples/crowded_street.mp3"
```

The uploaded clips preserve the original experimental audio bytes. Precomputed ablation caches and ad hoc plotting/statistics programs are excluded.
