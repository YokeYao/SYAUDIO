#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TTS 生成语音质量检测（conversation/conversion QC）
输出：CSV 报告（每条音频一行）
依赖：
  pip install numpy pandas soundfile jiwer
  # ASR:
  pip install -U transformers torch

用法：
  python tts_qc.py --meta metadata.jsonl --out report.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import soundfile as sf

# WER/CER
try:
    from jiwer import wer, cer
    HAS_JIWER = True
except Exception:
    HAS_JIWER = False

from pathlib import Path
def _patch_hf_env():
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("DISABLE_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    home_cache = Path.home() / ".cache" / "huggingface"
    env_paths = {
        "HF_HOME": home_cache,
        "HUGGINGFACE_HUB_CACHE": home_cache / "hub",
        "TRANSFORMERS_CACHE": home_cache / "transformers",
        "HF_TOKEN_PATH": home_cache / "token",
    }
    for key, value in env_paths.items():
        current = os.environ.get(key)
        if (not current) or current.startswith("/l/users/"):
            os.environ[key] = str(value)
        try:
            value.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass


_patch_hf_env()
# Whisper ASR (HuggingFace Transformers)
HAS_TRANSFORMERS = False
_whisper_model = None
_whisper_processor = None
try:
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    try:
        from transformers.utils import import_utils as _import_utils
        _import_utils._tf_available = False
    except Exception:
        pass
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

# Progress bar (optional)
try:
    from tqdm import tqdm
    HAS_TQDM = True
except Exception:
    HAS_TQDM = False


@dataclass
class QCResult:
    id: str
    audio_path: str
    asr_text: Optional[str] = None
    expected_text: Optional[str] = None
    wer: Optional[float] = None
    cer: Optional[float] = None
    issues: Optional[str] = None


def load_audio_mono(path: str) -> Tuple[np.ndarray, int]:
    """Return mono float32 audio in [-1, 1], sr."""
    x, sr = sf.read(path, always_2d=True)
    x = x.astype(np.float32)
    x_mono = np.mean(x, axis=1)
    x_mono = np.clip(x_mono, -1.0, 1.0)
    return x_mono, sr


def resample_audio(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return x
    if x.size == 0:
        return x
    try:
        import librosa
        return librosa.resample(x, orig_sr=orig_sr, target_sr=target_sr)
    except Exception:
        # Simple linear resample fallback
        ratio = float(target_sr) / float(orig_sr)
        n = int(round(x.size * ratio))
        if n <= 1:
            return x[:1]
        xp = np.linspace(0.0, 1.0, num=x.size, endpoint=True)
        fp = x
        x_new = np.interp(np.linspace(0.0, 1.0, num=n, endpoint=True), xp, fp)
        return x_new.astype(np.float32)


def lazy_load_whisper_hf(model_name: str = "openai/whisper-small"):
    global _whisper_model, _whisper_processor
    if _whisper_model is None or _whisper_processor is None:
        _whisper_processor = WhisperProcessor.from_pretrained(model_name)
        _whisper_model = WhisperForConditionalGeneration.from_pretrained(model_name)
        # 默认不开 forced decoder（自动语言检测），需要时再设置
        _whisper_model.config.forced_decoder_ids = None
    return _whisper_processor, _whisper_model


def run_asr_whisper_hf(
    x: np.ndarray,
    sr: int,
    model_name: str = "openai/whisper-small",
    language: Optional[str] = None,
    device: Optional[str] = None
) -> str:
    processor, model = lazy_load_whisper_hf(model_name)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # 设置语言提示（可选）
    if language:
        forced_ids = processor.get_decoder_prompt_ids(language=language, task="transcribe")
        model.config.forced_decoder_ids = forced_ids
    else:
        model.config.forced_decoder_ids = None

    target_sr = int(getattr(processor.feature_extractor, "sampling_rate", 16000))
    if sr != target_sr:
        x = resample_audio(x, sr, target_sr)
        sr = target_sr

    inputs = processor(x, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(input_features)
    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return (text or "").strip()


def normalize_text_zh(s: str) -> str:
    # 非严格：去掉空白；你也可以扩展成全角半角、标点统一等
    return "".join(s.split())


def qc_one(
    sample: Dict[str, Any],
    whisper_model: str = "openai/whisper-small",
    whisper_language: Optional[str] = None,
    do_asr: bool = True
) -> QCResult:
    sid = str(sample.get("id", ""))
    path = str(sample.get("audio_path", ""))
    expected = sample.get("expected_text", None)
    if expected is None:
        expected = sample.get("question", None)
    expected = str(expected) if expected is not None else None

    if not os.path.exists(path):
        raise FileNotFoundError(f"audio not found: {path}")

    x, sr = load_audio_mono(path)

    asr_text = None
    WER = None
    CER = None

    if do_asr and HAS_TRANSFORMERS:
        try:
            asr_text = run_asr_whisper_hf(
                x,
                sr,
                model_name=whisper_model,
                language=whisper_language
            )
        except Exception:
            asr_text = ""

        if expected and HAS_JIWER:
            # 中文更推荐 CER；WER 也给出（按空格分词会偏怪，所以下面用“字符级空格”技巧）
            exp_n = normalize_text_zh(expected)
            hyp_n = normalize_text_zh(asr_text or "")

            CER = float(cer(exp_n, hyp_n))

            # WER：把每个汉字当作“词”
            exp_w = " ".join(list(exp_n))
            hyp_w = " ".join(list(hyp_n))
            WER = float(wer(exp_w, hyp_w))

    # issue tags
    issues = []
    if expected and CER is not None and CER > 0.25:
        issues.append("high_CER")
    if expected and WER is not None and WER > 0.35:
        issues.append("high_WER")

    return QCResult(
        id=sid,
        audio_path=path,
        asr_text=asr_text,
        expected_text=expected,
        wer=WER,
        cer=CER,
        issues="|".join(issues) if issues else ""
    )


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="metadata.jsonl")
    ap.add_argument("--out", required=True, help="output report csv path")
    ap.add_argument("--whisper_model", default="openai/whisper-small", help="HF model name, e.g. openai/whisper-small")
    ap.add_argument("--whisper_lang", default=None, help="force language (e.g. zh, en). default: auto")
    ap.add_argument("--no_asr", action="store_true", help="disable ASR checks")
    ap.add_argument("--resume", action="store_true", help="resume if out csv exists (skip processed rows)")
    args = ap.parse_args()

    samples = read_jsonl(args.meta)

    # Determine fieldnames from QCResult dataclass for consistent CSV schema
    fieldnames = list(QCResult.__dataclass_fields__.keys())

    processed = set()
    out_exists = os.path.exists(args.out)
    if args.resume and out_exists:
        # Load already processed ids to skip
        try:
            with open(args.out, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rid = (row.get("id") or "").strip()
                    if rid:
                        processed.add(rid)
        except Exception:
            processed = set()

    mode = "a" if (args.resume and out_exists) else "w"
    with open(args.out, mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()

        iterable = samples
        if HAS_TQDM:
            iterable = tqdm(samples, total=len(samples), initial=len(processed))
        for s in iterable:
            sid = str(s.get("id", ""))
            if args.resume and sid in processed:
                continue
            try:
                r = qc_one(
                    s,
                    whisper_model=args.whisper_model,
                    whisper_language=args.whisper_lang,
                    do_asr=(not args.no_asr)
                )
                row = asdict(r)
            except Exception as e:
                # 某条失败也要落报告
                row = {
                    "id": sid,
                    "audio_path": str(s.get("audio_path", "")),
                    "issues": f"exception:{type(e).__name__}:{e}"
                }

            # Ensure all columns exist
            for k in fieldnames:
                row.setdefault(k, None)
            writer.writerow(row)
            f.flush()

    print(f"Saved report to: {args.out}")

    # 也可输出一个 “疑似问题样本” 列表
    try:
        df = pd.read_csv(args.out, encoding="utf-8-sig")
        bad = df[df.get("issues", "").astype(str).str.len() > 0]
        print(f"Flagged samples: {len(bad)}/{len(df)}")
        if len(bad) > 0:
            print(bad[["id", "audio_path", "issues"]].head(20).to_string(index=False))
    except Exception:
        pass


if __name__ == "__main__":
    main()
