from __future__ import annotations

import argparse
import math
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy import signal

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIO_SAMPLES_DIR = REPO_ROOT / "audio_samples"

NOISE_LIBRARY = {
    "cafe": AUDIO_SAMPLES_DIR / "cafe_chatter.mp3",
    "forest": AUDIO_SAMPLES_DIR / "forest_ambience.mp3",
    "street": AUDIO_SAMPLES_DIR / "crowded_street.mp3",
}

SEVERITY_PRESETS = {
    50: {
        "noise_gain_db": -12.0,
        "reverb_decay": 0.18,
        "lowpass_hz": 4600.0,
        "highpass_hz": 110.0,
        "downsample_sr": 12000,
        "drive": 1.15,
        "speech_rate": 1.10,
    },
    100: {
        "noise_gain_db": -7.5,
        "reverb_decay": 0.28,
        "lowpass_hz": 3600.0,
        "highpass_hz": 180.0,
        "downsample_sr": 9000,
        "drive": 1.35,
        "speech_rate": 1.25,
    },
    200: {
        "noise_gain_db": -4.0,
        "reverb_decay": 0.42,
        "lowpass_hz": 2800.0,
        "highpass_hz": 260.0,
        "downsample_sr": 7000,
        "drive": 1.65,
        "speech_rate": 1.50,
    },
    300: {
        "noise_gain_db": -2.5,
        "reverb_decay": 0.55,
        "lowpass_hz": 2400.0,
        "highpass_hz": 320.0,
        "downsample_sr": 6000,
        "drive": 1.80,
        "speech_rate": 1.75,
    },
    400: {
        "noise_gain_db": -1.5,
        "reverb_decay": 0.70,
        "lowpass_hz": 2000.0,
        "highpass_hz": 380.0,
        "downsample_sr": 5500,
        "drive": 1.95,
        "speech_rate": 2.00,
    },
    500: {
        "noise_gain_db": -0.5,
        "reverb_decay": 0.85,
        "lowpass_hz": 1700.0,
        "highpass_hz": 420.0,
        "downsample_sr": 5000,
        "drive": 2.10,
        "speech_rate": 2.25,
    },
    900: {
        "noise_gain_db": 0.0,
        "reverb_decay": 1.00,
        "lowpass_hz": 1500.0,
        "highpass_hz": 500.0,
        "downsample_sr": 4000,
        "drive": 2.40,
        "speech_rate": 5.00,
    },
}


def rms(signal_values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal_values), dtype=np.float64) + 1e-12))


def normalize_peak(waveform: np.ndarray, peak: float = 0.95) -> np.ndarray:
    current_peak = float(np.max(np.abs(waveform)))
    if current_peak <= 1e-8:
        return waveform
    return waveform * (peak / current_peak)


def build_rir(sr: int, decay: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rir_len = max(int(sr * (0.22 + decay)), 256)
    rir = np.zeros(rir_len, dtype=np.float32)
    rir[0] = 1.0

    early_reflections = [
        (0.011, 0.55),
        (0.019, 0.41),
        (0.031, 0.29),
        (0.047, 0.22),
    ]
    for delay_s, gain in early_reflections:
        idx = min(rir_len - 1, int(delay_s * sr))
        rir[idx] += gain

    tail_start = int(0.055 * sr)
    tail_time = np.arange(max(rir_len - tail_start, 1), dtype=np.float32) / sr
    tail = rng.standard_normal(tail_time.shape[0]).astype(np.float32)
    tail *= np.exp(-tail_time / max(decay, 1e-3))
    rir[tail_start:] += 0.12 * tail

    rir /= np.linalg.norm(rir) + 1e-8
    return rir


def apply_reverb(waveform: np.ndarray, sr: int, decay: float, seed: int) -> np.ndarray:
    rir = build_rir(sr, decay=decay, seed=seed)
    reverbed = signal.fftconvolve(waveform, rir, mode="full")[: waveform.shape[0]]
    return reverbed.astype(np.float32)


def apply_channel_distortion(
    waveform: np.ndarray,
    sr: int,
    highpass_hz: float,
    lowpass_hz: float,
    downsample_sr: int,
) -> np.ndarray:
    downsample_sr = max(4000, min(int(downsample_sr), sr))
    degraded = librosa.resample(waveform, orig_sr=sr, target_sr=downsample_sr, res_type="soxr_hq")
    degraded = librosa.resample(degraded, orig_sr=downsample_sr, target_sr=sr, res_type="soxr_hq")

    sos_high = signal.butter(4, highpass_hz, btype="highpass", fs=sr, output="sos")
    sos_low = signal.butter(4, lowpass_hz, btype="lowpass", fs=sr, output="sos")
    degraded = signal.sosfiltfilt(sos_high, degraded).astype(np.float32)
    degraded = signal.sosfiltfilt(sos_low, degraded).astype(np.float32)
    return degraded


def apply_nonlinearity(waveform: np.ndarray, drive: float) -> np.ndarray:
    compressed = np.tanh(drive * waveform)
    return (compressed / max(np.max(np.abs(compressed)), 1e-8)).astype(np.float32)


def apply_speech_rate(waveform: np.ndarray, rate: float) -> np.ndarray:
    if rate <= 0:
        raise ValueError(f"Speech rate must be positive, got {rate}")
    stretched = librosa.effects.time_stretch(waveform, rate=rate)
    return stretched.astype(np.float32)


def add_background_noise(
    waveform: np.ndarray,
    sr: int,
    noise_name: str,
    noise_gain_db: float,
    seed: int,
) -> np.ndarray:
    noise_path = NOISE_LIBRARY[noise_name]
    noise, _ = librosa.load(noise_path, sr=sr, mono=True)
    if noise.shape[0] < waveform.shape[0]:
        repeat = math.ceil(waveform.shape[0] / max(noise.shape[0], 1))
        noise = np.tile(noise, repeat)

    rng = np.random.default_rng(seed)
    max_start = max(noise.shape[0] - waveform.shape[0], 0)
    start = int(rng.integers(0, max_start + 1)) if max_start else 0
    noise = noise[start : start + waveform.shape[0]]

    target_noise_rms = rms(waveform) * (10.0 ** (noise_gain_db / 20.0))
    noise = noise * (target_noise_rms / max(rms(noise), 1e-8))
    return (waveform + noise).astype(np.float32)


def process_audio_file(
    src_path: Path,
    dst_path: Path,
    scenario: str,
    severity: int,
    sr: int = 16000,
) -> None:
    preset = SEVERITY_PRESETS[severity]
    seed = sum(ord(ch) for ch in src_path.stem) % (2**32)
    waveform, _ = librosa.load(src_path, sr=sr, mono=True)
    waveform = normalize_peak(waveform.astype(np.float32))

    processed = waveform
    if scenario == "audio_reverb":
        processed = apply_reverb(processed, sr=sr, decay=preset["reverb_decay"], seed=seed)
    elif scenario == "audio_channel":
        processed = apply_channel_distortion(
            processed,
            sr=sr,
            highpass_hz=preset["highpass_hz"],
            lowpass_hz=preset["lowpass_hz"],
            downsample_sr=preset["downsample_sr"],
        )
    elif scenario == "audio_nonlinear":
        processed = apply_nonlinearity(processed, drive=preset["drive"])
    elif scenario == "audio_speech_rate":
        processed = apply_speech_rate(processed, rate=preset["speech_rate"])
    elif scenario.startswith("audio_noise_"):
        noise_name = scenario.removeprefix("audio_noise_")
        processed = add_background_noise(
            processed,
            sr=sr,
            noise_name=noise_name,
            noise_gain_db=preset["noise_gain_db"],
            seed=seed + 17,
        )
    elif scenario.startswith("audio_compound_"):
        noise_name = scenario.removeprefix("audio_compound_")
        processed = apply_reverb(processed, sr=sr, decay=preset["reverb_decay"], seed=seed)
        processed = apply_channel_distortion(
            processed,
            sr=sr,
            highpass_hz=preset["highpass_hz"],
            lowpass_hz=preset["lowpass_hz"],
            downsample_sr=preset["downsample_sr"],
        )
        processed = apply_nonlinearity(processed, drive=preset["drive"])
        processed = add_background_noise(
            processed,
            sr=sr,
            noise_name=noise_name,
            noise_gain_db=preset["noise_gain_db"],
            seed=seed + 17,
        )
    else:
        raise ValueError(f"Unsupported scenario: {scenario}")
    processed = normalize_peak(processed)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sf.write(tmp_path, processed, sr)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(tmp_path),
                "-codec:a",
                "libmp3lame",
                "-q:a",
                "3",
                str(dst_path),
            ],
            check=True,
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def build_output_dir(dataset: str, scenario: str, severity: int) -> Path:
    return REPO_ROOT / "benchmark" / "ablation_data" / "background_noise" / dataset / scenario / str(severity)


def ensure_audio_alias(output_dir: Path, filename: str) -> None:
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    alias_path = audio_dir / filename
    if alias_path.exists() or alias_path.is_symlink():
        return
    alias_path.symlink_to(Path("..") / filename)


def iter_source_files(dataset: str) -> list[Path]:
    input_dir = REPO_ROOT / "benchmark" / dataset / "audio"
    files = sorted(input_dir.glob("*.mp3"))
    return [path for path in files if 1000 < int(path.stem[-5:]) <= 1100]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create compound acoustic noise ablations for audio QA.")
    parser.add_argument("--dataset", type=str, required=True, help="GSM8K or MMLU")
    parser.add_argument(
        "--scenario",
        type=str,
        default="audio_reverb",
        choices=[
            "audio_reverb",
            "audio_channel",
            "audio_nonlinear",
            "audio_speech_rate",
            "audio_noise_cafe",
            "audio_noise_forest",
            "audio_noise_street",
            "audio_compound_cafe",
            "audio_compound_forest",
            "audio_compound_street",
        ],
    )
    parser.add_argument("--volume", type=int, default=200, choices=sorted(SEVERITY_PRESETS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.upper()
    output_dir = build_output_dir(dataset, args.scenario, args.volume)
    source_files = iter_source_files(dataset)

    def _run_one(src_path: Path) -> str:
        dst_path = output_dir / src_path.name
        if dst_path.exists() and not args.force:
            ensure_audio_alias(output_dir, src_path.name)
            return "skipped"
        process_audio_file(src_path, dst_path, scenario=args.scenario, severity=args.volume)
        ensure_audio_alias(output_dir, src_path.name)
        return "done"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_run_one, path): path for path in source_files}
        iterator = as_completed(futures)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(futures), desc=f"{args.scenario}-{args.volume}", unit="file")
        for future in iterator:
            future.result()


if __name__ == "__main__":
    main()
