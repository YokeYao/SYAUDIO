"""Batch inference on MMAU / MMAR / GSM8K / MMLU multiple-choice audio QA with Qwen2-Audio."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import time
import torch
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List
MODEL_CACHE_DIR = os.environ.get("HF_HUB_CACHE")

from openai import OpenAI
import librosa
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    Qwen2_5OmniForConditionalGeneration,
    Qwen2_5OmniProcessor,
)
try:
    from transformers import AudioFlamingo3ForConditionalGeneration
except ImportError:  # pragma: no cover - optional dependency
    AudioFlamingo3ForConditionalGeneration = None
try:
    from qwen_omni_utils import process_mm_info
except ImportError:  # pragma: no cover - optional dependency
    process_mm_info = None

from prompt import PROMPTS

CHOICE_LETTERS = "ABCD"
REPO_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = Path(os.environ.get("SYAUDIO_RESULT_ROOT", str(REPO_ROOT / "result")))
BENCHMARK_DIR = Path(os.environ.get("SYAUDIO_DATA_ROOT", str(REPO_ROOT / "benchmark")))
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

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
    ablation_audio_root: Path = None


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
        ablation_audio_root=BENCHMARK_DIR / "ablation_data" / "background_noise" / "GSM8K",
        audio_field="audio_path",
    ),
    "mmlu": DatasetConfig(
        name="mmlu",
        default_data=BENCHMARK_DIR / "MMLU" / "mmlu_combined.jsonl",
        default_audio_root=BENCHMARK_DIR / "MMLU",
        ablation_audio_root=BENCHMARK_DIR / "ablation_data" / "background_noise" / "MMLU",
        audio_field="audio_path",
    ),
}


def model_dir_name(model_id: str) -> str:
    """Return a filesystem-friendly folder name for a model id."""
    # Local Hugging Face snapshot paths end in an opaque commit hash. Keep the
    # human-readable model name stable across snapshots and evaluation modes.
    if "Qwen2-Audio-7B-Instruct" in model_id:
        return "Qwen2-Audio-7B-Instruct"
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


def build_omni_conversation(audio_path: Path, prompt_text: str) -> list[dict]:
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
                {"type": "audio", "audio": str(audio_path)},
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
    """Parse existing log(s) to recover processed ids and counters (base + per-GPU)."""
    pattern = re.compile(r"id=([^|]+).*?correct=(True|False)", re.IGNORECASE)
    seen_ids: set[str] = set()
    total = 0
    correct = 0

    def _accumulate(path: Path) -> None:
        nonlocal total, correct
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            match = pattern.search(line)
            if not match:
                continue
            sample_id, correct_flag = match.groups()
            sample_id = sample_id.strip()
            seen_ids.add(sample_id)
            total += 1
            correct += 1 if correct_flag.lower() == "true" else 0

    _accumulate(log_path)
    gpu_logs = log_path.parent.glob(f"{log_path.stem}.gpu*.log")
    for gpu_log in gpu_logs:
        _accumulate(gpu_log)
    return seen_ids, total, correct


def _log_path_for_device(base_log: Path, device_id: int) -> Path:
    return base_log.with_name(f"{base_log.stem}.gpu{device_id}{base_log.suffix}")


def load_model_and_processor(model_id: str, device_id: int | None = None):
    """Select correct processor/model pair for the given model id."""
    device_map = f"cuda:{device_id}" if device_id is not None else "auto"
    if is_omni_model(model_id):
        processor = Qwen2_5OmniProcessor.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=MODEL_CACHE_DIR
        )
        model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype="auto",
            trust_remote_code=True,
            cache_dir=MODEL_CACHE_DIR,
        )
    elif is_flamingo_model(model_id):
        if AudioFlamingo3ForConditionalGeneration is None:
            raise ImportError(
                "AudioFlamingo3ForConditionalGeneration is not available in this environment."
            )
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=MODEL_CACHE_DIR
        )
        model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype="auto",
            trust_remote_code=True,
            cache_dir=MODEL_CACHE_DIR,
        )
    else:
        processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=MODEL_CACHE_DIR
        )
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device_map,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE_DIR,
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
    device_id: int | None = None,
) -> Dict[str, int]:
    if is_openai_api_model(model_id):
        return run_inference_openai_api(
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

    processor, model = load_model_and_processor(model_id, device_id=device_id)

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
        if is_omni_model(model_id):
            if process_mm_info is None:
                raise RuntimeError(
                    "qwen_omni_utils is not available; cannot run omni models."
                )
            conversation = build_omni_conversation(audio_path, prompt_text)
            text = processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )
            audios, images, videos = process_mm_info(
                conversation, use_audio_in_video=False
            )
            inputs = processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False,
            ).to(model.device)
            generated = model.generate(
                **inputs,
                max_new_tokens=max_gen_len,
                return_audio=False,
            )
        else:
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
                processor_kwargs.update({"padding": "longest"})

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
    return {"total": total, "correct": correct}


def _run_worker(
    device_id: int,
    samples: List[Dict],
    audio_root: Path,
    dataset: DatasetConfig,
    model_id: str,
    max_gen_len: int,
    log_path: Path,
    resume: bool,
    processed_ids: set[str],
    initial_total: int,
    initial_correct: int,
    result_queue: mp.Queue,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    logger = setup_logger(log_path, resume=resume)
    stats = run_inference(
        samples,
        audio_root=audio_root,
        dataset=dataset,
        logger=logger,
        model_id=model_id,
        max_gen_len=max_gen_len,
        processed_ids=processed_ids,
        initial_total=initial_total,
        initial_correct=initial_correct,
        resume=resume,
        device_id=device_id,
    )
    result_queue.put(stats)


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
        
        logger.info("Starting API call for sample id=%s", sample_id)

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
    return {"total": total, "correct": correct}


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
        help="Qwen/Qwen2-Audio-7B-Instruct, Qwen/Qwen2.5-Omni-7B, nvidia/audio-flamingo-3-hf, gpt-4o-mini-audio-preview, vertex-gemini-2.5-flash-lite-preview-09-2025-nothinking",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for local models (1, 2, or 4). Ignored for API models.",
    )
    parser.add_argument("--data", type=Path, default=None, help="Override the dataset annotation file.")
    parser.add_argument("--audio-root", type=Path, default=None, help="Override the audio root directory.")
    parser.add_argument("--log-path", type=Path, default=None, help="Override baseline log path; keep dataset prefix, e.g. gsm8k_human.log.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_cfg = DATASET_CONFIGS[args.dataset]
    data_path = args.data or dataset_cfg.default_data
    audio_root = args.audio_root or dataset_cfg.default_audio_root
    log_path = build_log_path(
        dataset=dataset_cfg.name,
        limit=args.limit,
        model_id=args.model,
    )
    if args.log_path is not None:
        log_path = args.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)
    processed_ids, initial_total, initial_correct = load_previous_results(log_path)
    existing_logs = [log_path] + list(log_path.parent.glob(f"{log_path.stem}.gpu*.log"))
    resume_logging = any(p.exists() for p in existing_logs)
    resume_flag = bool(processed_ids)

    samples = load_dataset(data_path, limit=args.limit)

    skipped = 0
    if processed_ids:
        before = len(samples)
        samples = [s for s in samples if (s.get("id") or s.get("question_id")) not in processed_ids]
        skipped = before - len(samples)

    if not samples:
        print(
            f"No usable samples left after skipping seen ids ({len(processed_ids)});"
            " assuming baseline already complete."
        )
        return

    if is_openai_api_model(args.model) or args.num_gpus == 1:
        logger = setup_logger(log_path, resume=resume_logging)
        logger.info(
            "Starting inference: dataset=%s | %d samples | prompt=baseline | limit=%d | model=%s | log=%s | gpus=%s",
            dataset_cfg.name,
            len(samples),
            args.limit,
            args.model,
            log_path,
            "API" if is_openai_api_model(args.model) else [0],
        )
        if skipped:
            logger.info(
                "Resuming: skipped %d samples already logged (seen=%d, prior_total=%d, prior_correct=%d)",
                skipped,
                len(processed_ids),
                initial_total,
                initial_correct,
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
            resume=resume_flag,
        )
        return

    available_gpus = torch.cuda.device_count()
    device_ids = list(range(min(max(1, args.num_gpus), available_gpus)))
    if not device_ids:
        raise RuntimeError("No CUDA devices available for local models.")

    shards: List[List[Dict]] = [samples[i:: len(device_ids)] for i in range(len(device_ids))]
    shards = [s for s in shards if s]
    mp.set_start_method("spawn", force=True)
    result_queue: mp.Queue = mp.Queue()
    processes: List[mp.Process] = []

    def _log_path_for_device(base_log: Path, device_id: int) -> Path:
        return base_log.with_name(f"{base_log.stem}.gpu{device_id}{base_log.suffix}")

    logger = setup_logger(log_path, resume=resume_logging)
    logger.info(
        "Launched %d GPU workers on devices %s; per-GPU logs at %s.gpu<id>.log",
        len(device_ids),
        device_ids[: len(shards)],
        log_path,
    )
    if skipped:
        logger.info(
            "Resuming: skipped %d samples already logged (seen=%d, prior_total=%d, prior_correct=%d)",
            skipped,
            len(processed_ids),
            initial_total,
            initial_correct,
        )

    for device_id, shard in zip(device_ids, shards):
        worker_log = _log_path_for_device(log_path, device_id)
        p = mp.Process(
            target=_run_worker,
            args=(
                device_id,
                shard,
                audio_root,
                dataset_cfg,
                args.model,
                args.max_gen_len,
                worker_log,
                resume_logging,
                processed_ids,
                0,
                0,
                result_queue,
            ),
            name=f"audio-eval-gpu{device_id}",
        )
        p.start()
        processes.append(p)

    aggregate = {"total": initial_total, "correct": initial_correct}
    for _ in processes:
        stats = result_queue.get()
        for key in aggregate:
            aggregate[key] += stats.get(key, 0)

    for p in processes:
        p.join()

    gpu_logs = sorted(log_path.parent.glob(f"{log_path.stem}.gpu*.log"))
    for handler in logger.handlers:
        flush_fn = getattr(handler, "flush", None)
        if flush_fn:
            flush_fn()
    if gpu_logs:
        with log_path.open("a", encoding="utf-8") as main_log_file:
            for gpu_log in gpu_logs:
                if not gpu_log.exists():
                    continue
                content = gpu_log.read_text(encoding="utf-8")
                if content:
                    if not content.endswith("\n"):
                        content += "\n"
                    main_log_file.write(content)
                gpu_log.unlink(missing_ok=True)

    acc = (aggregate["correct"] / aggregate["total"] * 100) if aggregate["total"] else 0.0
    logger.info(
        "Aggregate finished %d questions | accuracy=%.2f%% (%d/%d)",
        aggregate["total"],
        acc,
        aggregate["correct"],
        aggregate["total"],
    )


if __name__ == "__main__":
    main()
