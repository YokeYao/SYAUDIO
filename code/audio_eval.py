"""Batch inference on MMAU / MMAR / GSM8K / MMLU multiple-choice audio QA with Qwen2-Audio."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import re
import time
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from openai import OpenAI
import librosa
from transformers import (
    AutoProcessor,
    AudioFlamingo3ForConditionalGeneration,
    Qwen2AudioForConditionalGeneration,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)

from prompt import PROMPTS

CHOICE_LETTERS = "ABCD"
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = REPO_ROOT / "result"
BENCHMARK_DIR = REPO_ROOT / "benchmark"
OPENAI_BASE_URL = "https://api.ohmygpt.com/v1"
OPENAI_API_KEY = "sk-2Nqq2VWF6dcE36A03473T3BlbKFJ3c87A119658845D29Bcc"


def _patch_torch_autocast():
    # Some torch builds expose is_autocast_enabled() without a device_type arg; shim to ignore extras.
    try:
        torch.is_autocast_enabled("cuda")
    except TypeError:
        _orig = torch.is_autocast_enabled

        def _shim(*args, **kwargs):
            return _orig()

        torch.is_autocast_enabled = _shim


_patch_torch_autocast()


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    default_data: Path
    default_audio_root: Path
    audio_field: str


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "mmau": DatasetConfig(
        name="mmau",
        default_data=BENCHMARK_DIR / "MMAU" / "mmau-test-mini.json",
        default_audio_root=BENCHMARK_DIR / "MMAU",
        audio_field="audio_id",
    ),
    "mmar": DatasetConfig(
        name="mmar",
        default_data=BENCHMARK_DIR / "MMAR" / "MMAR-meta.json",
        default_audio_root=BENCHMARK_DIR / "MMAR",
        audio_field="audio_path",
    ),
    "gsm8k": DatasetConfig(
        name="gsm8k",
        default_data=BENCHMARK_DIR / "GSM8K" / "test_mcq.jsonl",
        default_audio_root=BENCHMARK_DIR / "GSM8K",
        audio_field="audio_path",
    ),
    "mmlu": DatasetConfig(
        name="mmlu",
        default_data=BENCHMARK_DIR / "MMLU" / "mmlu_combined.jsonl",
        default_audio_root=BENCHMARK_DIR / "MMLU",
        audio_field="audio_path",
    ),
}


def model_dir_name(model_id: str) -> str:
    """Return a filesystem-friendly folder name for a model id."""
    return model_id.rstrip("/").split("/")[-1]


def build_log_path(
    dataset: str,
    limit: int,
    model_id: str | None = None,
) -> Path:
    """Construct a descriptive log path under the result directory."""
    prompt_dir = "baseline"
    parts = [dataset, prompt_dir, f"limit{limit}"]
    filename = f"{'_'.join(parts)}.log"

    if model_id:
        return RESULT_DIR / prompt_dir / model_dir_name(model_id) / filename
    return RESULT_DIR / filename


def setup_logger(log_path: Path, resume: bool = False) -> logging.Logger:
    logger = logging.getLogger("mmau_inference")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    mode = "a" if resume else "w"
    file_handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def load_audio(wav_path: Path, sampling_rate: int):
    waveform, _ = librosa.load(wav_path, sr=sampling_rate)
    return waveform


def load_dataset(path: Path, limit: int | None = None) -> List[Dict]:
    if path.suffix == ".jsonl":
        data: list[dict] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    return data if limit is None else data[:limit]


def normalize_choice_value(value) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    filtered = "".join(ch for ch in value if ch.isalnum() or ch in {".", "-", "+"})
    return filtered.lower()


def get_answer_letter(sample: Dict) -> str:
    choices = sample.get("choices", [])
    answer = sample.get("answer")

    if not choices or answer is None:
        return ""

    # Numeric index answers (e.g., MMLU).
    if isinstance(answer, int):
        return CHOICE_LETTERS[answer] if 0 <= answer < len(choices) else ""

    # String answers that look numeric might be actual choice text (GSM8K) or an index.
    if isinstance(answer, str) and answer.strip().isdigit():
        normalized_answer = normalize_choice_value(answer)
        for idx, choice in enumerate(choices):
            if normalize_choice_value(choice) == normalized_answer:
                return CHOICE_LETTERS[idx]

        # Fall back to treating it as an index if no textual match is found.
        idx = int(answer.strip())
        return CHOICE_LETTERS[idx] if 0 <= idx < len(choices) else ""

    # Direct string match.
    try:
        answer_idx = choices.index(answer)
        return CHOICE_LETTERS[answer_idx]
    except ValueError:
        pass

    normalized_answer = normalize_choice_value(answer)
    for idx, choice in enumerate(choices):
        if normalize_choice_value(choice) == normalized_answer:
            return CHOICE_LETTERS[idx]
    return ""


def build_prompt(sample: Dict) -> str:
    """Build the fixed baseline prompt."""
    template = PROMPTS["baseline"]
    choices = list(sample.get("choices", []))
    if len(choices) < 4:
        choices.extend([""] * (4 - len(choices)))

    return template.format(
        question=sample["question"],
        choices=choices,
    ).strip()


def build_conversation(audio_path: Path, prompt_text: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are an audio question answering assistant.",
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": str(audio_path)},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


def extract_boxed_content(prediction: str) -> str:
    """Return the last \\boxed{...} content if present."""
    matches = re.findall(r"\\boxed\{([^}]*)\}", prediction)
    if matches:
        return matches[-1].strip()
    return ""


def extract_boxed_letter(prediction: str) -> str:
    content = extract_boxed_content(prediction)
    if not content:
        return ""

    for ch in content:
        if ch in CHOICE_LETTERS:
            return ch
    return ""


def normalize_prediction(prediction: str) -> str:
    prediction = prediction.strip()

    boxed_letter = extract_boxed_letter(prediction)
    if boxed_letter:
        return boxed_letter

    # Fallback: scan from the end, return the first A/B/C/D encountered.
    for ch in reversed(prediction):
        if ch in CHOICE_LETTERS:
            return ch
    return ""


def is_omni_model(model_id: str) -> bool:
    normalized = model_id.lower()
    return "qwen2.5-omni" in normalized or "qwen2_5-omni" in normalized


def is_openai_api_model(model_id: str) -> bool:
    lower = model_id.lower()
    # Treat both OpenAI GPT endpoints and Gemini/Vertex endpoints as API-backed models.
    return ("gpt" in lower) or ("gemini" in lower)


def is_flamingo_model(model_id: str) -> bool:
    return "audio-flamingo-3" in model_id.lower()


def load_previous_results(log_path: Path) -> tuple[set[str], int, int]:
    """Parse an existing log to recover processed ids and counters."""
    if not log_path.exists():
        return set(), 0, 0

    pattern = re.compile(r"id=([^|]+).*?correct=(True|False)", re.IGNORECASE)
    seen_ids: set[str] = set()
    total = 0
    correct = 0

    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if not match:
            continue
        sample_id, correct_flag = match.groups()
        sample_id = sample_id.strip()
        seen_ids.add(sample_id)
        total += 1
        correct += 1 if correct_flag.lower() == "true" else 0

    return seen_ids, total, correct


def load_model_and_processor(model_id: str):
    """Select correct processor/model pair for the given model id."""
    if is_omni_model(model_id):
        processor = Qwen2_5OmniProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True
        )
    elif is_flamingo_model(model_id):
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", torch_dtype="auto", trust_remote_code=True
        )
    else:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id, device_map="auto", trust_remote_code=True
        )
    return processor, model


def run_inference(
    samples: List[Dict],
    audio_root: Path,
    dataset: DatasetConfig,
    logger: logging.Logger,
    model_id: str,
    max_gen_len: int = 256,
    processed_ids: set[str] | None = None,
    initial_total: int = 0,
    initial_correct: int = 0,
    resume: bool = False,
) -> None:
    if is_openai_api_model(model_id):
        run_inference_openai_api(
            samples=samples,
            audio_root=audio_root,
            dataset=dataset,
            logger=logger,
            model_id=model_id,
            max_gen_len=max_gen_len,
            processed_ids=processed_ids,
            initial_total=initial_total,
            initial_correct=initial_correct,
            resume=resume,
        )
        return

    processor, model = load_model_and_processor(model_id)

    processed_ids = set(processed_ids or ())
    total = initial_total
    correct = initial_correct

    if resume and processed_ids:
        log_file = getattr(logger.handlers[0], "baseFilename", "") if logger.handlers else ""
        logger.info(
            "Resuming: found %d completed samples (correct=%d) in %s",
            total,
            correct,
            log_file,
        )

    for idx, sample in enumerate(samples, start=1):
        sample_id = sample.get("id") or sample.get("question_id") or f"idx{idx}"
        if sample_id in processed_ids:
            logger.info("Skipping already processed sample id=%s", sample_id)
            continue

        log_idx = total + 1
        prompt_text = build_prompt(sample)
        audio_field = dataset.audio_field
        if audio_field not in sample:
            raise KeyError(f"Sample missing '{audio_field}' for dataset {dataset.name}")
        audio_path = (audio_root / sample[audio_field]).resolve()
        audio_waveform = load_audio(
            audio_path, sampling_rate=processor.feature_extractor.sampling_rate
        )

        conversation = build_conversation(audio_path, prompt_text)
        text = processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )
        # AudioFlamingo3 expects audio features padded to its max length (1500 tokens post-conv),
        # so force max-length padding for that family; other models can keep dynamic padding.
        processor_kwargs = {
            "text": text,
            "audio": [audio_waveform],
            "sampling_rate": processor.feature_extractor.sampling_rate,
            "return_tensors": "pt",
        }
        if is_flamingo_model(model_id):
            processor_kwargs.update({"padding": "max_length", "truncation": True})
        else:
            processor_kwargs.update({"padding": True})

        inputs = processor(**processor_kwargs).to(model.device)

        # Use max_new_tokens to avoid HF warning when generation_config sets max_length.
        generated = model.generate(**inputs, max_new_tokens=max_gen_len)

        # Omni models may return (sequences, audio_outputs). Standard models return a tensor or ModelOutput.
        if hasattr(generated, "sequences"):
            sequences = generated.sequences
        elif isinstance(generated, tuple):
            sequences = generated[0]
        else:
            sequences = generated

        sequences = sequences[:, inputs.input_ids.size(1) :]
        response = processor.batch_decode(
            sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        boxed_content = extract_boxed_content(response)
        predicted_letter = normalize_prediction(response)
        response_text = response.strip()
        response_one_line = response_text.replace("\n", "\\n")
        gold_letter = get_answer_letter(sample)
        is_correct = predicted_letter == gold_letter and gold_letter != ""

        total += 1
        correct += int(is_correct)
        running_acc = correct / total if total else 0.0

        logger.info(
            "Q%02d id=%s | pred=%s | gold=%s | correct=%s | running_acc=%.2f%% | raw=%s",
            log_idx,
            sample_id,
            predicted_letter or response_text,
            gold_letter,
            is_correct,
            running_acc * 100,
            response_one_line,
        )

    logger.info(
        "Finished %d questions | accuracy=%.2f%% (%d/%d)",
        total,
        (correct / total * 100) if total else 0.0,
        correct,
        total,
    )


def encode_audio_for_openai(audio_path: Path) -> tuple[str, str]:
    """Return base64-encoded audio bytes and detected format."""
    audio_bytes = audio_path.read_bytes()
    audio_format = audio_path.suffix.lstrip(".") or "wav"
    return base64.b64encode(audio_bytes).decode("utf-8"), audio_format


def extract_text_from_message_content(content) -> str:
    """Extract plain text from OpenAI message content (list or string)."""
    if isinstance(content, str):
        return content.strip()
    parts: list[str] = []
    for item in content or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(text)
        elif isinstance(item, dict):
            value = item.get("text")
            if value:
                parts.append(str(value))
    return "\n".join(parts).strip()


def run_inference_openai_api(
    samples: List[Dict],
    audio_root: Path,
    dataset: DatasetConfig,
    logger: logging.Logger,
    model_id: str,
    max_gen_len: int = 512,
    processed_ids: set[str] | None = None,
    initial_total: int = 0,
    initial_correct: int = 0,
    resume: bool = False,
) -> None:
    client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    processed_ids = set(processed_ids or ())
    total = initial_total
    correct = initial_correct

    if resume and processed_ids:
        log_file = getattr(logger.handlers[0], "baseFilename", "") if logger.handlers else ""
        logger.info(
            "Resuming: found %d completed samples (correct=%d) in %s",
            total,
            correct,
            log_file,
        )

    for idx, sample in enumerate(samples, start=1):
        sample_id = sample.get("id") or sample.get("question_id") or f"idx{idx}"
        if sample_id in processed_ids:
            logger.info("Skipping already processed sample id=%s", sample_id)
            continue

        log_idx = total + 1
        prompt_text = build_prompt(sample)
        audio_field = dataset.audio_field
        if audio_field not in sample:
            raise KeyError(f"Sample missing '{audio_field}' for dataset {dataset.name}")
        audio_path = (audio_root / sample[audio_field]).resolve()
        audio_b64, audio_format = encode_audio_for_openai(audio_path)

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are an audio question answering assistant."}
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": audio_b64, "format": audio_format},
                    },
                    {"type": "text", "text": prompt_text},
                ],
        },
    ]

        response_text = ""
        max_retries = 2
        for attempt in range(max_retries + 1):
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.0,
                max_tokens=max_gen_len,
            )
            message = resp.choices[0].message
            response_text = extract_text_from_message_content(message.content)

            if response_text:
                break
            if attempt < max_retries:
                logger.warning(
                    "Empty response for id=%s attempt=%d/%d; retrying after 1s",
                    sample["id"],
                    attempt + 1,
                    max_retries,
                )
                time.sleep(1)

        boxed_content = extract_boxed_content(response_text)
        predicted_letter = normalize_prediction(response_text)
        response_one_line = response_text.replace("\n", "\\n")
        gold_letter = get_answer_letter(sample)
        is_correct = predicted_letter == gold_letter and gold_letter != ""

        total += 1
        correct += int(is_correct)
        running_acc = correct / total if total else 0.0

        logger.info(
            "Q%02d id=%s | pred=%s | gold=%s | correct=%s | running_acc=%.2f%% | raw=%s",
            log_idx,
            sample_id,
            predicted_letter or response_text,
            gold_letter,
            is_correct,
            running_acc * 100,
            response_one_line,
        )

    logger.info(
        "Finished %d questions | accuracy=%.2f%% (%d/%d)",
        total,
        (correct / total * 100) if total else 0.0,
        correct,
        total,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Audio models inference on baseline audio QA."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=sorted(DATASET_CONFIGS.keys()),
        default="mmau",
        help="Dataset to evaluate: mmau, mmar, gsm8k, or mmlu.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of questions to run from the dataset.",
    )
    parser.add_argument(
        "--max-gen-len",
        type=int,
        default=2048,
        help="Maximum generation length.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-Audio-7B-Instruct",
        help="Qwen/Qwen2-Audio-7B-Instruct, nvidia/audio-flamingo-3-hf, gpt-audio-mini, vertex-gemini-2.5-flash-lite-preview-09-2025-nothinking",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing log: skip processed ids and keep counts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_cfg = DATASET_CONFIGS[args.dataset]
    data_path = dataset_cfg.default_data
    audio_root = dataset_cfg.default_audio_root
    log_path = build_log_path(
        dataset=dataset_cfg.name,
        limit=args.limit,
        model_id=args.model,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    processed_ids: set[str] = set()
    initial_total = 0
    initial_correct = 0
    if args.resume:
        processed_ids, initial_total, initial_correct = load_previous_results(log_path)

    logger = setup_logger(log_path, resume=args.resume)

    samples = load_dataset(data_path, limit=args.limit)
    logger.info(
        "Starting inference: dataset=%s | %d samples | prompt=baseline | limit=%d | model=%s | log=%s",
        dataset_cfg.name,
        len(samples),
        args.limit,
        args.model,
        log_path,
    )

    run_inference(
        samples,
        audio_root=audio_root,
        dataset=dataset_cfg,
        logger=logger,
        model_id=args.model,
        max_gen_len=args.max_gen_len,
        processed_ids=processed_ids,
        initial_total=initial_total,
        initial_correct=initial_correct,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
