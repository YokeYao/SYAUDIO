<h1 align="center">SYAUDIO</h1>

<h2 align="center">Hearing is Believing? Evaluating and Analyzing Audio Language Model Sycophancy with SYAUDIO</h2>

<p align="center"><strong>NeurIPS 2026 · Main Poster</strong></p>

<p align="center">
  Junchi Yao<sup>1,2</sup> &nbsp;&nbsp;
  Lokranjan Lakshmikanthan<sup>3</sup> &nbsp;&nbsp;
  Annie Zhao<sup>3</sup> &nbsp;&nbsp;
  Danielle Zhao<sup>3</sup>
  <br>
  Shu Yang<sup>4</sup> &nbsp;&nbsp;
  Zikang Ding<sup>1,2</sup> &nbsp;&nbsp;
  Di Wang<sup>4</sup> &nbsp;&nbsp;
  Lijie Hu<sup>1,†</sup>
</p>

<p align="center">
  <sup>1</sup>Mohamed bin Zayed University of Artificial Intelligence<br>
  <sup>2</sup>University of Electronic Science and Technology of China<br>
  <sup>3</sup>Georgia Institute of Technology<br>
  <sup>4</sup>King Abdullah University of Science and Technology
</p>

<p align="center"><sup>†</sup> Corresponding author</p>

<p align="center">
  <a href="https://github.com/YokeYao/SYAUDIO"><img src="https://img.shields.io/badge/GitHub-SYAUDIO-181717?style=for-the-badge&amp;logo=github&amp;logoColor=white" alt="GitHub: SYAUDIO"></a>
  <a href="https://huggingface.co/datasets/YokyYao/ALM-Sycophancy"><img src="https://img.shields.io/badge/Hugging_Face-Dataset-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=FFD21E" alt="Hugging Face: Dataset"></a>
  <a href="https://arxiv.org/abs/2601.23149"><img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white" alt="Paper: arXiv"></a>
</p>

## Abstract

Audio Language Models (ALMs) have recently shown strong capabilities in unified reasoning over speech, sound, and natural language; yet we find that they can inherit sycophancy, the tendency to agree with user assertions even when they contradict objective evidence. This failure mode is especially concerning for audio-conditioned reasoning, where a model must preserve evidence from acoustic events, speaker characteristics, and speech rate while responding to potentially misleading user feedback. However, unlike text and vision-language sycophancy, ALM sycophancy has not been systematically studied. We therefore introduce SYAUDIO, the first benchmark dedicated to evaluating sycophancy in ALMs, consisting of 4,319 audio questions spanning Audio Perception, Audio Reasoning, Audio Math, and Audio Ethics. Built upon established audio benchmarks and augmented with TTS-generated arithmetic and moral reasoning tasks, SYAUDIO enables systematic evaluation across multiple domains and sycophancy types with carefully verified data quality, including a human-speaker validation of the TTS pipeline. Using this benchmark, we identify substantial and audio-specific sycophancy patterns under realistic conditions involving noise and speech rate, and further show that supervised fine-tuning reduces misleading susceptibility while decode-time steering reveals controllable hidden-state directions for both misleading susceptibility and correction receptiveness.

## 🌐 Introduction

When user feedback conflicts with acoustic evidence, do audio language models preserve their original reasoning or drift toward agreement? ALMs can reason over speech, sound, and natural language, but they can also abandon a correct answer after a user challenges it or suggests an alternative. In audio tasks, this means overlooking evidence such as acoustic events, speaker characteristics, and speaking rate.

**SYAUDIO** studies this behavior across **4,319 audio questions** in four domains: perception, reasoning, mathematics, and ethics. Its two-round protocol first records a model's answer, then tests its response to four types of user cues: **Bias Feedback**, **“Are You Sure?”**, **Answer Sycophancy**, and **Mimicry Sycophancy**. We distinguish harmful changes from correct to incorrect answers using **Misleading Susceptibility Score (MSS ↓)**, and beneficial corrections using **Correction Receptiveness Score (CRS ↑)**.

Beyond text-based user cues, the paper examines spoken instructions, background noise, and speech-rate changes. It also validates synthesized question audio against recordings from three human speakers and investigates mitigation through prompting, supervised fine-tuning, and hidden-state steering. This repository releases the benchmark data and inference/evaluation code.

## 🧭 Overview

![Overview of the SYAUDIO benchmark and evaluation protocol](assets/syaudio-overview.png)

*Figure 2 from the paper. SYAUDIO combines audio perception, reasoning, mathematics and ethics with baseline and follow-up evaluations under user cues and acoustic perturbations.*

## 📊 Experimental Results

The following values are reported in the paper; they are not new runs of the released code. MSS and CRS are percentages with different denominators: initially correct and initially incorrect examples, respectively. Lower MSS and higher CRS are better.

### Main evaluation: audio reasoning (MMAR)

Selected conditions from **Table 1**, covering all five evaluated models. Each cell reports **MSS ↓ / CRS ↑**.

| Model | Bias Feedback (strong) | Answer Sycophancy | Mimicry Sycophancy |
|---|---:|---:|---:|
| Qwen2-Audio-7B-Instruct | 47.73 / 18.71 | 66.40 / 19.24 | 69.87 / 49.28 |
| Audio-Flamingo-3 | 7.35 / 10.64 | 13.79 / 1.77 | 72.61 / 72.73 |
| Qwen2.5-Omni-7B | 15.87 / 19.48 | 20.99 / 11.16 | 54.50 / 64.61 |
| GPT-4o-Mini-Audio-Preview | 19.43 / 23.12 | 19.07 / 5.97 | 42.60 / 67.79 |
| Gemini-2.5-Flash-2025-09-26 | 15.26 / 21.11 | 9.42 / 8.71 | 39.94 / 64.91 |

Mimicry can induce both harmful answer flips and useful corrections: for example, Audio-Flamingo-3 reaches **72.61% MSS** and **72.73% CRS** on MMAR. This is why robustness and correction receptiveness must be assessed together.

### Audio-specific findings

- **Spoken instructions:** converting only the instruction component from text to TTS audio, while keeping the question audio unchanged, raises mean Bias Feedback MSS from **14.66% to 37.42%** (Section 4.3). This is an instruction-modality comparison, not a comparison of synthetic and human question recordings.
- **Human-speaker validation:** on the same 100 GSM8K questions, three human recordings per question produce the results below (**Table 14 of the NeurIPS manuscript**). The model-level MSS pattern is consistent across TTS and human speech.

| Model | Question audio | Baseline accuracy ↑ | MSS ↓ | CRS ↑ |
|---|---|---:|---:|---:|
| GPT-4o-Mini-Audio-Preview | TTS | 95.00 | 2.11 | 40.00 |
| GPT-4o-Mini-Audio-Preview | Human mean (3 speakers) | 95.67 | 1.04 | 37.78 |
| Qwen2-Audio-7B-Instruct | TTS | 47.00 | 70.21 | 23.53 |
| Qwen2-Audio-7B-Instruct | Human mean (3 speakers) | 49.00 | 70.03 | 29.70 |

The human-validation recordings cover 100 aligned questions, separate from the 4,319-question core benchmark. The linked arXiv preprint is an earlier version; the human-speaker table above is from the NeurIPS manuscript.

## Dataset

**Availability:** all 4,319 core audio files, 300 human-validation recordings, and seven annotation files are available at the linked Hugging Face repository and match this release byte-for-byte. The human-validation recordings are also retained in this GitHub repository.

| Domain | Source | Core questions |
|---|---|---:|
| Audio Perception | MMAU-test-mini | 1,000 |
| Audio Reasoning | MMAR | 1,000 |
| Audio Math | GSM8K-Audio | 1,319 |
| Audio Ethics | MMLU (moral)-Audio | 1,000 |
| **Total** | **SYAUDIO** | **4,319** |

The core audio occupies **6.39 GB before archive compression**. Core audio and annotations are available on [Hugging Face as `YokyYao/ALM-Sycophancy`](https://huggingface.co/datasets/YokyYao/ALM-Sycophancy); GitHub retains the lightweight annotations, evaluation code, and the collaborator-contributed human validation recordings already present in this repository. This GitHub release also includes **300 human recordings** (100 aligned questions × three speakers), separate from the core benchmark.

All core audio references and human-recording references were checked. The annotation audit found 14 unresolved answer labels, 2 numeric-index fallbacks and 29 rows with more than four options; original data are preserved and the details are documented in the [reproduction guide](docs/REPRODUCING.md#annotation-compatibility-findings). The release includes original paths, sample IDs, per-component provenance, and a [verified file index](scripts/syaudio-files.json) with SHA-256 checksums. The installer uses the existing Hugging Face folder layout without duplicating its audio into archives. Dataset licenses differ by component; see [DATA_LICENSES.md](docs/DATA_LICENSES.md).

## Installation

Use Python 3.10. Install a compatible CUDA-enabled PyTorch build for local-model inference and FFmpeg for audio processing.

```bash
git clone https://github.com/YokeYao/SYAUDIO.git
cd SYAUDIO
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` records the versions from the evaluation environment, including the Transformers version used by the local model adapters. Downloading the dataset alone only requires `huggingface_hub`:

```bash
pip install huggingface_hub
python scripts/download_syaudio.py
# Optional: also download and verify the human-validation recordings.
python scripts/download_syaudio.py --groups all
```

The installer pins the verified Hugging Face commit and checks individual file hashes; it refuses to overwrite differing local files. Use `--destination /path/to/new-directory` for a separate installation and `--revision <commit-sha>` to pin a dataset revision. Data are installed under `<destination>/benchmark/`.

## Quick start

Run a 10-question baseline and the "Are You Sure?" follow-up:

```bash
python code/audio_eval.py --dataset gsm8k --limit 10 \
  --model Qwen/Qwen2-Audio-7B-Instruct --num-gpus 1

python code/sycophancy.py \
  --baseline-log result/baseline/Qwen2-Audio-7B-Instruct/gsm8k_baseline_limit10.log \
  --prompt are_you_sure --model Qwen/Qwen2-Audio-7B-Instruct --num-gpus 1
```

Run all four core datasets and six follow-up conditions:

```bash
bash scripts/run_core.sh Qwen/Qwen2-Audio-7B-Instruct
```

Available follow-up conditions are `bias_feedback` (`low`, `medium`, `strong`), `are_you_sure`, `answer_sycophancy`, and `mimicry_sycophancy`. Answer and mimicry variants are selected from first-round correctness. The evaluator retains the original answer parsing and reports raw predictions plus MSS/CRS counters.

For API-backed models, set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL` in your environment. API names containing `gpt` or `gemini` use the OpenAI-compatible adapter. An endpoint must support the requested model and audio message format; use a model ID actually available from your provider. API evaluation sends the selected audio and prompts to that provider. Local model runs do not require an API key.

## Release scope

- `code/audio_eval.py`, `code/sycophancy.py`, `code/prompt.py`: baseline and core sycophancy evaluation.
- `code/ablation_code/`: audio-prompt, acoustic perturbation and follow-up evaluation.
- `code/audio_cue_only_sycophancy.py`, `code/multiturn_audio_sycophancy.py`: additional audio-cue and multi-turn inference protocols.
- `code/prompt_anti_sycophancy.py`: inference-time prompt mitigation, enabled explicitly with `SYAUDIO_PROMPT_SET=anti` for `sycophancy.py`.
- `scripts/`: dataset installation and portable core evaluation launcher.
- `benchmark/`: original metadata and existing collaborator human validation assets.
- `result/`: historical output files already in the repository; newly generated outputs are ignored by Git.

Training code, fine-tuning frameworks, model weights and checkpoints are excluded. Standalone confidence-interval, plotting and metric-comparison scripts from the experimental workspace are not part of this release. Model evaluation still reports the protocol's built-in MSS/CRS counters. Detailed commands, supported data overrides, and limitations are in [the reproduction guide](docs/REPRODUCING.md).

## Citation

Please cite the paper and the constituent datasets. The public preprint citation is:

```bibtex
@misc{yao2026hearingbelieving,
  title={Hearing is Believing? Evaluating and Analyzing Audio Language Model Sycophancy with SYAUDIO},
  author={Junchi Yao and Lokranjan Lakshmikanthan and Annie Zhao and Danielle Zhao and Shu Yang and Zikang Ding and Di Wang and Lijie Hu},
  year={2026},
  eprint={2601.23149},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2601.23149}
}
```

## License

Code is covered by the repository [MIT license](LICENSE). Dataset components retain their [respective licenses](docs/DATA_LICENSES.md), including the non-commercial restrictions on MMAU and MMAR.
