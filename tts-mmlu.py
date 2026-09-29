import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict

from openai import OpenAI

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def load_questions(path: Path) -> List[Dict]:
    questions: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def ensure_ids(questions: List[Dict], prefix: str = "mmlu") -> List[Dict]:
    updated = []
    for idx, q in enumerate(questions):
        item = dict(q)
        if "id" not in item or not item["id"]:
            item["id"] = f"{prefix}_{idx:05d}"
        updated.append(item)
    return updated


def save_questions(path: Path, questions: List[Dict]) -> None:
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
        print(f"[backup] Saved original to {backup}")

    with path.open("w", encoding="utf-8") as f:
        for item in questions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_question_text(item: Dict) -> str:
    for key in ("question", "input", "prompt"):
        text = item.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def synthesize_questions(
    client: OpenAI,
    questions: List[Dict],
    output_dir: Path,
    workers: int = 8,
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
    for item in questions:
        q_text = extract_question_text(item)
        if not q_text:
            continue
        audio_name = f"{item['id']}.mp3"
        tasks.append((q_text, output_dir / audio_name))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(synthesize_one, q_text, audio_path): audio_path
            for q_text, audio_path in tasks
        }
        iterator = as_completed(futures)
        if tqdm:
            iterator = tqdm(iterator, total=len(futures), desc="Synthesizing", unit="q")

        for future in iterator:
            audio_path = futures[future]
            result = future.result()
            if not tqdm:
                if result == "done":
                    print(f"Saved audio: {audio_path}")
                elif result == "skipped":
                    print(f"Skipped existing: {audio_path}")


def main() -> None:
    base_path = Path(__file__).resolve().parent
    data_path = base_path / "benchmark" / "MMLU" / "mmlu_combined.jsonl"
    output_dir = base_path / "benchmark" / "MMLU" / "audio"

    client = OpenAI(
        base_url="https://api.ohmygpt.com/v1",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
    )

    questions = load_questions(data_path)
    questions_with_ids = ensure_ids(questions, prefix="mmlu")
    save_questions(data_path, questions_with_ids)
    synthesize_questions(client, questions_with_ids, output_dir, workers=8)


if __name__ == "__main__":
    main()
