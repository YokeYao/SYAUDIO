"""Download Qwen2-Audio-7B-Instruct into the local Hugging Face cache."""

from __future__ import annotations

import os
from pathlib import Path


def _patch_hf_env(cache_root: Path) -> None:
    env_paths = {
        "HF_HOME": str(cache_root),
        "HUGGINGFACE_HUB_CACHE": str(cache_root / "hub"),
        "TRANSFORMERS_CACHE": str(cache_root / "transformers"),
        "HF_TOKEN_PATH": str(cache_root / "token"),
    }
    for key, value in env_paths.items():
        current = os.environ.get(key)
        if current and current.startswith("/l/users/"):
            os.environ[key] = value


def main() -> None:
    model_id = "Qwen/Qwen2-Audio-7B-Instruct"
    cache_root = Path("/nfs-stor/junchi.yao")
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / ".created_by_download_script").touch(exist_ok=True)
    _patch_hf_env(cache_root)

    # Trigger HF download by loading processor/model with cache_dir.
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    AutoProcessor.from_pretrained(model_id, cache_dir=str(cache_root))
    Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id, cache_dir=str(cache_root)
    )


if __name__ == "__main__":
    main()
