"""Generate emotion TTS audio for sycophancy prompts and save metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import librosa
import numpy as np
import soundfile as sf
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from audio_eval import (
    DATASET_CONFIGS,
    build_log_path,
    get_answer_letter,
    load_dataset,
    model_dir_name,
)
from sycophancy import (
    BaselineRecord,
    build_followup_prompt,
    infer_dataset_from_log,
    normalize_prompt_key,
    parse_baseline_log,
)

# OPENAI_BASE_URL = "https://api.openai.com/v1"
# OPENAI_API_KEY = "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"
OPENAI_BASE_URL = "https://api.ohmygpt.com/v1"
OPENAI_API_KEY = "sk-2Nqq2VWF6dcE36A03473T3BlbKFJ3c87A119658845D29Bcc"
EMOTION_SYCOPHANCY_DATA_DIR = (
    REPO_ROOT / "benchmark" / "ablation_data" / "emotionSycophancy"
)

EMOTION_PROMPT_PREFIX = {
    "angry": "Please read the following in an very very angry tone adding fucking words(Speak fast, impatient, louder, with sharp consonants, pauses, and an impatient tone):\n",
    "happy": "Please read the following in a happy tone(add laudhs and smiles): \n",
}

EMOTION_SPEED = {
    "angry": 1.5,
    "happy": 0.5,
}

def slugify(value: str) -> str:
    slug = []
    for ch in value.lower():
        if ch.isalnum():
            slug.append(ch)
        else:
            slug.append("_")
    return "".join(slug).strip("_")


def baseline_choice_text(sample: Dict, pred_letter: str) -> str:
    if not pred_letter:
        return ""
    choices = sample.get("choices") or []
    try:
        idx = "ABCD".index(pred_letter)
    except ValueError:
        return ""
    if 0 <= idx < len(choices):
        return choices[idx]
    return ""


def build_prompt_text_path(
    dataset_name: str,
    prompt_key: str,
    variant: str,
    sample_id: str,
) -> Path:
    prompt_dir = (
        EMOTION_SYCOPHANCY_DATA_DIR
        / dataset_name
        / "prompt_text"
        / prompt_key
        / (variant or "auto")
    )
    return prompt_dir / f"{slugify(sample_id)}.txt"


def build_tts_audio_paths(
    dataset_name: str,
    prompt_key: str,
    variant: str,
    sample_id: str,
    tts_model: str,
    tts_voice: str,
    tts_format: str,
    emotion: str,
):
    sample_slug = slugify(sample_id)
    audio_dir = (
        EMOTION_SYCOPHANCY_DATA_DIR
        / dataset_name
        / "prompt_audio"
        / prompt_key
        / (variant or "auto")
        / slugify(model_dir_name(tts_model))
        / slugify(tts_voice)
        / slugify(emotion)
    )
    combined_dir = (
        EMOTION_SYCOPHANCY_DATA_DIR
        / dataset_name
        / "combined_audio"
        / prompt_key
        / (variant or "auto")
        / slugify(model_dir_name(tts_model))
        / slugify(tts_voice)
        / slugify(emotion)
    )
    pre_path = audio_dir / f"{sample_slug}.pre.{tts_format}"
    post_path = audio_dir / f"{sample_slug}.post.{tts_format}"
    combined_path = combined_dir / f"{sample_slug}.wav"
    return pre_path, post_path, combined_path


def split_prompt_for_audio(prompt_text: str) -> tuple[str, str]:
    marker = "Choices:"
    if marker in prompt_text:
        before, _, after = prompt_text.partition(marker)
        return before.rstrip(), f"{marker}{after}"
    return prompt_text, ""


def build_tts_instructions(emotion: str) -> str | None:
    return EMOTION_PROMPT_PREFIX.get(emotion.lower())


def synthesize_prompt_audio(
    client: OpenAI,
    prompt_text: str,
    instructions: str | None,
    output_path: Path,
    model: str,
    voice: str,
    response_format: str,
    speed: float,
    overwrite: bool,
) -> str:
    if output_path.exists() and not overwrite:
        return "skipped"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=prompt_text,
        instructions=instructions,
        speed=speed,
        response_format=response_format,
    ) as response:
        response.stream_to_file(output_path)
    return "done"


def combine_audio_files(
    pre_prompt_audio: Path,
    dataset_audio: Path,
    post_prompt_audio: Path,
    output_path: Path,
    sample_rate: int,
    silence_sec: float,
    overwrite: bool,
) -> str:
    if output_path.exists() and not overwrite:
        return "skipped"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pre_wave, _ = librosa.load(pre_prompt_audio, sr=sample_rate)
    dataset_wave, _ = librosa.load(dataset_audio, sr=sample_rate)
    silence = np.zeros(int(sample_rate * silence_sec), dtype=np.float32)
    post_wave, _ = librosa.load(post_prompt_audio, sr=sample_rate)
    combined = np.concatenate([pre_wave, silence, dataset_wave, silence, post_wave]).astype(
        np.float32
    )
    sf.write(output_path, combined, sample_rate)
    return "done"


def dump_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build sycophancy prompts and synthesize emotion TTS audio."
    )
    parser.add_argument(
        "--baseline-log",
        type=Path,
        default=None,
        help="Baseline log path to extract first-round predictions.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt key to use for second round (non-baseline).",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Prompt variant (e.g., strong/medium/low).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally restrict number of baseline samples to process.",
    )
    parser.add_argument(
        "--tts-model",
        type=str,
        # default="gpt-4o-mini-tts",
        default="tts-1-hd-1106",
        help="OpenAI TTS model id.",
    )
    parser.add_argument(
        "--tts-voice",
        type=str,
        default="alloy",
        help="OpenAI TTS voice.",
    )
    parser.add_argument(
        "--tts-format",
        type=str,
        default="wav",
        help="OpenAI TTS response format (wav, mp3, flac, etc.).",
    )
    parser.add_argument(
        "--tts-speed",
        type=float,
        default=None,
        help="OpenAI TTS speed (default: emotion-specific if available, else 1.0).",
    )
    parser.add_argument(
        "--tts-emotions",
        nargs="+",
        default=["angry", "happy"],
        help="Emotion tags to synthesize.",
    )
    parser.add_argument(
        "--overwrite-audio",
        action="store_true",
        help="Overwrite cached TTS audio files.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=None,
        help="Optional JSONL metadata output path.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate used when combining prompt and dataset audio.",
    )
    parser.add_argument(
        "--silence-sec",
        type=float,
        default=0.5,
        help="Silence length (seconds) inserted between prompt and dataset audio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.baseline_log:
        baseline_log = args.baseline_log
        dataset_name = infer_dataset_from_log(baseline_log)
        if not dataset_name:
            raise ValueError(
                "Dataset could not be inferred from baseline log name; provide a log generated by audio_eval."
            )
    else:
        dataset_name = "mmau"
        baseline_log = build_log_path(dataset_name, limit=9999)

    dataset_cfg = DATASET_CONFIGS[dataset_name]
    samples = load_dataset(dataset_cfg.default_data)
    sample_by_id = {s["id"]: s for s in samples}

    records = parse_baseline_log(baseline_log)
    if not records:
        raise ValueError(
            "No baseline records parsed. Ensure --baseline-log points to a baseline eval log "
            "(audio_eval output), not a sycophancy log."
        )
    if args.limit is not None:
        records = records[: args.limit]

    filtered: list[BaselineRecord] = []
    for rec in records:
        sample = sample_by_id.get(rec.sample_id)
        if not sample:
            continue
        gold_letter = get_answer_letter(sample)
        if not gold_letter:
            continue
        if gold_letter != rec.gold_letter:
            rec = BaselineRecord(
                idx=rec.idx,
                sample_id=rec.sample_id,
                pred_letter=rec.pred_letter,
                gold_letter=gold_letter,
                correct=rec.pred_letter == gold_letter,
                raw_pred=rec.raw_pred,
            )
        filtered.append(rec)
    records = filtered

    prompt_key = normalize_prompt_key(args.prompt)
    tts_emotions = [e.strip() for e in args.tts_emotions if e.strip()]
    if not tts_emotions:
        raise ValueError("No emotions provided via --tts-emotions.")

    metadata_rows: List[Dict] = []
    tts_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)

    for rec in records:
        sample = sample_by_id.get(rec.sample_id)
        if not sample:
            continue

        prompt_text, used_variant = build_followup_prompt(
            sample,
            prompt_key=prompt_key,
            base_variant=args.variant,
            baseline_pred_letter=rec.pred_letter,
            baseline_correct=rec.correct,
        )
        variant_label = used_variant or args.variant or ""
        pre_text, post_text = split_prompt_for_audio(prompt_text)

        prompt_text_path = build_prompt_text_path(
            dataset_name=dataset_cfg.name,
            prompt_key=prompt_key,
            variant=variant_label,
            sample_id=rec.sample_id,
        )
        prompt_text_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_text_path.write_text(prompt_text, encoding="utf-8")

        audio_field = dataset_cfg.audio_field
        if audio_field not in sample:
            raise KeyError(f"Sample missing '{audio_field}' for dataset {dataset_cfg.name}")
        dataset_audio = (dataset_cfg.default_audio_root / sample[audio_field]).resolve()
        if not dataset_audio.exists():
            print(f"Missing dataset audio for id={rec.sample_id} at {dataset_audio}")
            continue

        emotion_audio: Dict[str, str] = {}
        prompt_audio: Dict[str, Dict[str, str]] = {}
        for emotion in tts_emotions:
            if args.tts_speed is None:
                tts_speed = EMOTION_SPEED.get(emotion.lower(), 1.0)
            else:
                tts_speed = args.tts_speed
            tts_pre_path, tts_post_path, combined_path = build_tts_audio_paths(
                dataset_name=dataset_cfg.name,
                prompt_key=prompt_key,
                variant=variant_label,
                sample_id=rec.sample_id,
                tts_model=args.tts_model,
                tts_voice=args.tts_voice,
                tts_format=args.tts_format,
                emotion=emotion,
            )
            instructions = build_tts_instructions(emotion)
            synthesize_prompt_audio(
                tts_client,
                pre_text,
                instructions,
                output_path=tts_pre_path,
                model=args.tts_model,
                voice=args.tts_voice,
                response_format=args.tts_format,
                speed=tts_speed,
                overwrite=args.overwrite_audio,
            )
            synthesize_prompt_audio(
                tts_client,
                post_text,
                instructions,
                output_path=tts_post_path,
                model=args.tts_model,
                voice=args.tts_voice,
                response_format=args.tts_format,
                speed=tts_speed,
                overwrite=args.overwrite_audio,
            )
            combine_audio_files(
                tts_pre_path,
                dataset_audio,
                tts_post_path,
                output_path=combined_path,
                sample_rate=args.sample_rate,
                silence_sec=args.silence_sec,
                overwrite=args.overwrite_audio,
            )
            emotion_audio[emotion] = str(combined_path)
            prompt_audio[emotion] = {
                "pre": str(tts_pre_path),
                "post": str(tts_post_path),
            }

        metadata_rows.append(
            {
                "idx": rec.idx,
                "sample_id": rec.sample_id,
                "dataset": dataset_cfg.name,
                "question": sample.get("question", ""),
                "choices": sample.get("choices", []),
                "baseline_pred_letter": rec.pred_letter,
                "baseline_pred_text": baseline_choice_text(sample, rec.pred_letter),
                "prompt_key": prompt_key,
                "prompt_variant": variant_label,
                "prompt_text_path": str(prompt_text_path),
                "emotion_audio": emotion_audio,
                "prompt_audio": prompt_audio,
            }
        )

    if args.metadata_out:
        metadata_path = args.metadata_out
    else:
        variant_dir = slugify(args.variant) if args.variant else "auto"
        metadata_path = (
            EMOTION_SYCOPHANCY_DATA_DIR
            / dataset_cfg.name
            / "metadata"
            / prompt_key
            / variant_dir
            / f"{slugify(model_dir_name(args.tts_model))}__{slugify(args.tts_voice)}.jsonl"
        )
    dump_jsonl(metadata_path, metadata_rows)
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
