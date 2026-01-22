import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydub import AudioSegment, effects
from pydub.generators import WhiteNoise, Sine

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


forest_ambience_noise = AudioSegment.from_mp3("./audio_samples/forest_ambience.mp3")
cafe_chatter_noise = AudioSegment.from_mp3("./audio_samples/cafe_chatter.mp3")


def add_background_noise(
    input_dir: Path,
    output_dir: Path,
    scenario: str,
    volume: int,
    workers: int = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_files = sorted(list(input_dir.glob("*.mp3")))[:100]

    def process_one(path: Path) -> str:
        out_path = output_dir / path.name
        if out_path.exists():
            return "skipped"

        speech = AudioSegment.from_mp3(path)
        noise = None

        if scenario == "forest":
            noise = forest_ambience_noise
        elif scenario == "cafe":
            noise = cafe_chatter_noise
        else:
            print(scenario)
            raise ValueError("scenario must be 'forest' or 'cafe'")

        # clip background noise to match speech length
        start = random.randint(0, len(noise) - len(speech))
        noise = noise[start : start + len(speech)]

        # adjust noise to half the speech loudness
        if volume == 50:
            adjustment = -10
        elif volume == 100:
            adjustment = 0
        elif volume == 200:
            adjustment = 10
        else:
            raise ValueError("volume must be 50, 100, or 200")
        noise = noise.apply_gain((speech.dBFS + adjustment) - noise.dBFS)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add background noise to original audio.")
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default=None,
        help="gsm8k, mmlu",
    )
    parser.add_argument(
        "-s",
        "--scenario",
        type=str,
        default=None,
        help="forest, cafe",
    )
    parser.add_argument(
        "-v",
        "--volume",
        type=int,
        default=None,
        help="50, 100, 200",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    base_path = Path(__file__).resolve().parents[2]
    dataset = args.dataset.upper()
    scenario = args.scenario
    volume = args.volume
    input_dir = base_path / "benchmark" / dataset / "audio"

    # forest ambience
    add_background_noise(
        input_dir=input_dir,
        output_dir=base_path / "benchmark" / "ablation_data" / "background_noise" / dataset / f"audio_{scenario}" / f"{volume}",
        scenario=scenario,
        volume=volume
    )

    # cafe chatter
    add_background_noise(
        input_dir=input_dir,
        output_dir=base_path / "benchmark" / "ablation_data" / "background_noise" / dataset / f"audio_{scenario}" / f"{volume}",
        scenario=scenario,
        volume=volume
    )


if __name__ == "__main__":
    main()
