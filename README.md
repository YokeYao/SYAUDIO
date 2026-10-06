
<h2 align="center">Hearing is Believing? Evaluating and Analyzing Audio Language Model Sycophancy with SYAUDIO</h2>

<p align="center"><strong>NeurIPS 2026 · Main Poster</strong></p>

<p align="center">
  Junchi Yao<sup>1</sup> &nbsp;&nbsp;
  Lokranjan Lakshmikanthan<sup>2</sup> &nbsp;&nbsp;
  Annie Zhao<sup>2</sup> &nbsp;&nbsp;
  Danielle Zhao<sup>2</sup>
  <br>
  Shu Yang<sup>3</sup> &nbsp;&nbsp;
  Zikang Ding<sup>1,4</sup> &nbsp;&nbsp;
  Di Wang<sup>3,†</sup> &nbsp;&nbsp;
  Lijie Hu<sup>1,†</sup>
</p>

<p align="center">
  <sup>1</sup>Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)<br>
  <sup>2</sup>Georgia Institute of Technology<br>
  <sup>3</sup>King Abdullah University of Science and Technology (KAUST)<br>
  <sup>4</sup>University of Electronic Science and Technology of China (UESTC)
</p>

<p align="center"><sup>†</sup> Corresponding author</p>

<p align="center">
  <a href="https://github.com/YokeYao/SYAUDIO"><img src="https://img.shields.io/badge/GitHub-SYAUDIO-181717?style=for-the-badge&amp;logo=github&amp;logoColor=white" alt="GitHub: SYAUDIO"></a>
  <a href="https://huggingface.co/datasets/YokyYao/SYAUDIO"><img src="https://img.shields.io/badge/Hugging_Face-Dataset-FFD21E?style=for-the-badge&amp;logo=huggingface&amp;logoColor=FFD21E" alt="Hugging Face: Dataset"></a>
  <a href="https://arxiv.org/abs/2601.23149"><img src="https://img.shields.io/badge/Paper-arXiv-B31B1B?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white" alt="Paper: arXiv"></a>
</p>

## Abstract

Audio Language Models (ALMs) have recently shown strong capabilities in unified reasoning over speech, sound, and natural language; yet we find that they can inherit sycophancy, the tendency to agree with user assertions even when they contradict objective evidence. This failure mode is especially concerning for audio-conditioned reasoning, where a model must preserve evidence from acoustic events, speaker characteristics, and speech rate while responding to potentially misleading user feedback. However, unlike text and vision-language sycophancy, ALM sycophancy has not been systematically studied. We therefore introduce SYAUDIO, the first benchmark dedicated to evaluating sycophancy in ALMs, consisting of 4,319 audio questions spanning Audio Perception, Audio Reasoning, Audio Math, and Audio Ethics. Built upon established audio benchmarks and augmented with TTS-generated arithmetic and moral reasoning tasks, SYAUDIO enables systematic evaluation across multiple domains and sycophancy types with carefully verified data quality, including a human-speaker validation of the TTS pipeline. Using this benchmark, we identify substantial and audio-specific sycophancy patterns under realistic conditions involving noise and speech rate, and further show that supervised fine-tuning reduces misleading susceptibility while decode-time steering reveals controllable hidden-state directions for both misleading susceptibility and correction receptiveness.

## 🌐 Introduction

When user feedback conflicts with acoustic evidence, do audio language models preserve their original reasoning or drift toward agreement? ALMs can reason over speech, sound, and natural language, but they can also abandon a correct answer after a user challenges it or suggests an alternative. In audio tasks, this means overlooking evidence such as acoustic events, speaker characteristics, and speaking rate.

**SYAUDIO** studies this behavior across **4,319 audio questions** in four domains: perception, reasoning, mathematics, and ethics. Its two-round protocol first records a model's answer, then tests its response to four types of user cues: **Bias Feedback**, **“Are You Sure?”**, **Answer Sycophancy**, and **Mimicry Sycophancy**. We distinguish harmful changes from correct to incorrect answers using **Misleading Susceptibility Score (MSS ↓)**, and beneficial corrections using **Correction Receptiveness Score (CRS ↑)**.

Beyond text-based user cues, the paper examines spoken instructions, background noise, and speech-rate changes. It also validates synthesized question audio against recordings from three human speakers and investigates mitigation through prompting, supervised fine-tuning, and hidden-state steering. This repository provides inference/evaluation code and a downloader for the benchmark hosted on Hugging Face.

## 🧭 Overview

![Overview of the SYAUDIO benchmark and evaluation protocol](assets/syaudio-overview.png)

*Figure 2 from the paper. SYAUDIO combines audio perception, reasoning, mathematics and ethics with baseline and follow-up evaluations under user cues and acoustic perturbations.*

## 📊 Experimental Results

MSS and CRS are percentages with different denominators: initially correct and initially incorrect examples, respectively. Lower MSS and higher CRS are better.

The complete Table 1 covers all five models, four datasets and six follow-up conditions:

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
- **Human-speaker validation:** on the same 100 GSM8K questions, three human recordings per question produce the results below (**Table 17 of arXiv v2**). The model-level MSS pattern is consistent across TTS and human speech.

| Model | Question audio | Baseline accuracy ↑ | MSS ↓ | CRS ↑ |
|---|---|---:|---:|---:|
| GPT-4o-Mini-Audio-Preview | TTS | 95.00 | 2.11 | 40.00 |
| GPT-4o-Mini-Audio-Preview | Human mean (3 speakers) | 95.67 | 1.04 | 37.78 |
| Qwen2-Audio-7B-Instruct | TTS | 47.00 | 70.21 | 23.53 |
| Qwen2-Audio-7B-Instruct | Human mean (3 speakers) | 49.00 | 70.03 | 29.70 |

The human-validation recordings cover 100 aligned questions, separate from the 4,319-question core benchmark. The human-speaker values above are reported in Table 17 of arXiv v2.

## Dataset

**Availability:** all 4,319 core audio files, 300 human-validation recordings, and seven annotation files are hosted on Hugging Face and can be downloaded with the verified installer.

| Domain | Source | Core questions |
|---|---|---:|
| Audio Perception | MMAU-test-mini | 1,000 |
| Audio Reasoning | MMAR | 1,000 |
| Audio Math | GSM8K-Audio | 1,319 |
| Audio Ethics | MMLU (moral)-Audio | 1,000 |
| **Total** | **SYAUDIO** | **4,319** |

Core audio and annotations are available on [Hugging Face as `YokyYao/SYAUDIO`](https://huggingface.co/datasets/YokyYao/SYAUDIO). The `benchmark/` directory is populated by the dataset downloader; core data and human-validation recordings are hosted on Hugging Face. The two audio-only cue clips (`reject.wav` and `not_sure.wav`) remain here for the corresponding evaluation.

## Installation

Requires Python 3.10 and FFmpeg; local-model inference also requires a CUDA-compatible PyTorch build.

```bash
git clone https://github.com/YokeYao/SYAUDIO.git
cd SYAUDIO
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_syaudio.py
```

Data are downloaded to `benchmark/`. See the [reproduction guide](docs/REPRODUCING.md) for optional downloads and detailed setup.

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
