import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydub import AudioSegment, effects
from pydub.generators import WhiteNoise, Sine

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


crowded_street_noise = AudioSegment.from_mp3("./ablation_audios/crowded_street.mp3")
forest_ambience_noise = AudioSegment.from_mp3("./ablation_audios/forest_ambience.mp3")


def add_background_noise(
    input_dir: Path,
    output_dir: Path,
    scenario: str,
    workers: int = 8,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_files = list(input_dir.glob("*.mp3"))

    def process_one(path: Path) -> str:
        out_path = output_dir / path.name
        if out_path.exists():
            return "skipped"

        speech = AudioSegment.from_mp3(path)

        if scenario == "street":
            noise = crowded_street_noise
            print(len(noise))
            start = random.randint(0, len(noise) - len(speech))
            noise = noise[start : start + len(speech)]
        elif scenario == "forest":
            noise = forest_ambience_noise
            print(len(noise))
            start = random.randint(0, len(noise) - len(speech))
            noise = noise[start : start + len(speech)]
        else:
            raise ValueError("scenario must be 'street' or 'forest'")

        # adjust noise to half the speech loudness
        noise = noise.apply_gain((speech.dBFS - 20) - noise.dBFS)
        mixed = speech.overlay(noise)
        mixed.export(out_path, format="mp3")

        return "done"

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(process_one, path): path
            for path in audio_files
        }

        iterator = as_completed(futures)
        if tqdm:
            iterator = tqdm(iterator, total=len(futures), desc="Adding noise", unit="file")

        for future in iterator:
            _ = future.result()


def main() -> None:
    base_path = Path(__file__).resolve().parent.parent
    input_dir = base_path / "benchmark" / "GSM8K" / "audio"
    # input_dir = base_path / "benchmark" / "MMLU" / "audio"

    # crowded street
    add_background_noise(
        input_dir=input_dir,
        output_dir=base_path / "benchmark" / "GSM8K" / "audio_street",
        # output_dir=base_path / "benchmark" / "MMLU" / "audio_street",
        scenario="street",
        workers=8,
    )

    # forest ambience
    add_background_noise(
        input_dir=input_dir,
        output_dir=base_path / "benchmark" / "GSM8K" / "audio_forest",
        # output_dir=base_path / "benchmark" / "MMLU" / "audio_forest",
        scenario="forest",
        workers=8,
    )


if __name__ == "__main__":
    main()
