import torch
import soundfile as sf
from pathlib import Path
import os
import subprocess
# Due to the collapse of MBZUAI server's Lustre, I change the default model storage
def _patch_hf_env():
    home_cache = Path.home() / ".cache" / "huggingface"
    env_paths = {
        "HF_HOME": str(home_cache),
        "HUGGINGFACE_HUB_CACHE": str(home_cache / "hub"),
        "TRANSFORMERS_CACHE": str(home_cache / "transformers"),
        "HF_TOKEN_PATH": str(home_cache / "token"),
    }
    for key, value in env_paths.items():
        current = os.environ.get(key)
        if not current or current.startswith("/l/users/"):
            os.environ[key] = value


_patch_hf_env()

from qwen_tts import Qwen3TTSModel

MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
MODEL_DIR = Path.home() / ".cache" / "huggingface" / "models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice"

# Ensure model files are present (downloads if missing).
MODEL_DIR.mkdir(parents=True, exist_ok=True)
subprocess.run(
    [
        "huggingface-cli",
        "download",
        MODEL_ID,
        "--local-dir",
        str(MODEL_DIR),
    ],
    check=True,
)

model = Qwen3TTSModel.from_pretrained(
    str(MODEL_DIR),
    device_map="auto",
    dtype=torch.bfloat16,
)

# single inference
wavs, sr = model.generate_custom_voice(
    text="其实我真的有发现，我是一个特别善于观察别人情绪的人。",
    language="Chinese", # Pass `Auto` (or omit) for auto language adaptive; if the target language is known, set it explicitly.
    speaker="Vivian",
    instruct="用特别愤怒的语气说", # Omit if not needed.
)
sf.write("output_custom_voice.wav", wavs[0], sr)

# batch inference
wavs, sr = model.generate_custom_voice(
    text=[
        "其实我真的有发现，我是一个特别善于观察别人情绪的人。", 
        "She said she would be here by noon."
    ],
    language=["Chinese", "English"],
    speaker=["Vivian", "Ryan"],
    instruct=["", "Very happy."]
)
sf.write("output_custom_voice_1.wav", wavs[0], sr)
sf.write("output_custom_voice_2.wav", wavs[1], sr)
