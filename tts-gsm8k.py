import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm


def load_questions(path: Path) -> list[dict]:
    questions = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def synthesize_questions(
    client: OpenAI, questions: list[dict], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize_one(question_text: str, output_path: Path) -> str:
        if output_path.exists():
            return "skipped"

        with client.audio.speech.with_streaming_response.create(
            model="tts-1-hd-1106",
            voice="alloy",
            input=question_text,
        ) as response:
            response.stream_to_file(output_path)

        return "done"

    tasks = []
    for idx, item in enumerate(questions):
        question_text = item.get("question")
        if not question_text:
            continue

        audio_name = f"{item.get('id', f'question_{idx:05d}')}.mp3"
        output_path = output_dir / audio_name
        tasks.append((question_text, output_path))

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(synthesize_one, question_text, output_path): output_path
            for question_text, output_path in tasks
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Synthesizing",
            unit="q",
        ):
            output_path = futures[future]
            result = future.result()
            if result == "done":
                print(f"Saved audio: {output_path}")
            elif result == "skipped":
                print(f"Skipped existing: {output_path}")


def main() -> None:
    client = OpenAI(
        base_url="https://api.ohmygpt.com/v1",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    base_path = Path(__file__).resolve().parent
    data_path = base_path / "benchmark" / "GSM8K" / "test_mcq.jsonl"
    output_dir = base_path / "benchmark" / "GSM8K" / "audio"

    questions = load_questions(data_path)
    synthesize_questions(client, questions, output_dir)


if __name__ == "__main__":
    main()
