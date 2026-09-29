import os
import argparse
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Sequence

import openai
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Target model config
MODEL_CONFIG = {
    "model_id": "gemini-2.5-flash-thinking",
    "api_key": os.environ.get("OPENAI_API_KEY", ""),
    "base_url": "http://35.220.164.252:3888/v1/",
}


def extract_final_answer(solution_text: str) -> str:
    """Pull the canonical short answer from the GSM8K solution field."""
    if not solution_text:
        return ""
    match = re.findall(r"####\s*(.+)", solution_text)
    if match:
        return match[-1].strip()
    # Fallback: take the last non-empty line
    lines = [line.strip() for line in solution_text.splitlines() if line.strip()]
    if lines:
        return lines[-1].strip()
    return ""


def build_openai_client() -> openai.OpenAI:
    return openai.OpenAI(
        api_key=MODEL_CONFIG["api_key"],
        base_url=MODEL_CONFIG["base_url"],
    )


def _normalize_choices(raw_choices: Sequence[str]) -> List[str]:
    seen = set()
    cleaned: List[str] = []
    for choice in raw_choices:
        text = str(choice).strip()
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
    return cleaned


def _fallback_distractors(correct_answer: str) -> List[str]:
    """Generate simple numeric distractors when the model output is unusable."""
    distractors: List[str] = []
    numeric_match = re.match(r"^-?\$?\s*([0-9]+(?:\.[0-9]+)?)", correct_answer)
    if numeric_match:
        try:
            base_val = float(numeric_match.group(1))
            tweaks = [base_val + 1, base_val - 1, base_val * 2]
            distractors = [str(int(t)) if t.is_integer() else f"{t:.2f}" for t in tweaks]
        except Exception:
            pass
    if not distractors:
        distractors = [
            f"Not {correct_answer}",
            f"About half of {correct_answer}",
            f"{correct_answer} minus one",
        ]
    return distractors


def ensure_four_choices(choices: List[str], correct_answer: str) -> List[str]:
    """Guarantee we return four options containing the correct answer exactly once."""
    choices = _normalize_choices(choices)
    correct_normalized = correct_answer.strip()
    if correct_normalized and correct_normalized not in choices:
        choices.append(correct_normalized)
    # Trim duplicates of the correct answer
    deduped: List[str] = []
    seen = set()
    for choice in choices:
        key = choice.strip()
        if key in seen:
            continue
        deduped.append(choice)
        seen.add(key)
    choices = deduped

    if len(choices) < 4:
        needed = 4 - len(choices)
        choices.extend(_fallback_distractors(correct_normalized)[:needed])
    if len(choices) > 4:
        choices = choices[:4]
    random.shuffle(choices)
    return choices


def call_model_for_choices(client: openai.OpenAI, question: str, correct_answer: str) -> List[str]:
    """Use the target model to propose three distractors plus the correct answer."""
    system_msg = {
        "role": "system",
        "content": (
            "You convert math word problems into multiple choice questions. "
            "Return strictly JSON with key 'choices' as an array of four short answer strings. "
            "Include the provided correct answer exactly once and add three plausible but wrong distractors. "
            "No reasoning, no extra keys."
        ),
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Question: {question}\n"
            f"Correct answer: {correct_answer}\n"
            "Respond with JSON: {\"choices\": [\"...\", \"...\", \"...\", \"...\"]}"
        ),
    }

    resp = client.chat.completions.create(
        model=MODEL_CONFIG["model_id"],
        messages=[system_msg, user_msg],
        temperature=0.4,
        response_format={"type": "json_object"},
    )
    payload = json.loads(resp.choices[0].message.content)
    return payload.get("choices", [])


def convert_file(input_path: Path, output_path: Path, resume: bool = False, max_retries: int = 3, workers: int = 8) -> None:
    client = build_openai_client()
    random.seed(42)

    if resume and output_path.exists():
        print(f"[resume] Loading existing output: {output_path}")
        processed: Dict[str, Dict] = {}
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                processed[obj["id"]] = obj
    else:
        processed = {}

    raw_lines = input_path.read_text(encoding="utf-8").splitlines()
    total = len(raw_lines)
    print(f"[info] Loaded {total} lines from {input_path}")

    def process_entry(idx: int, original: dict, entry_id: str):
        solution_text = original.get("answer", "")
        correct_answer = extract_final_answer(solution_text)
        retries = 0
        choices: List[str] = []
        while retries < max_retries:
            try:
                choices = call_model_for_choices(client, original["question"], correct_answer)
                break
            except Exception as e:
                retries += 1
                wait = 2 * retries
                if not tqdm:
                    print(f"[warn] API call failed ({e}), retry {retries}/{max_retries} after {wait}s")
                time.sleep(wait)
        choices = ensure_four_choices(choices, correct_answer)

        obj = dict(original)
        obj["solution"] = solution_text
        obj["answer"] = correct_answer
        obj["choices"] = choices
        obj["id"] = entry_id
        return idx, obj

    with output_path.open("w", encoding="utf-8") as fout:
        progress_bar = tqdm(total=total, desc="Converting", unit="item") if tqdm else None

        pending: Dict[int, Dict] = {}
        tasks = []
        next_to_write = 0

        # Queue tasks and enqueue already processed entries
        for idx, line in enumerate(raw_lines):
            if not line.strip():
                if progress_bar:
                    progress_bar.update(1)
                continue
            original = json.loads(line)
            entry_id = original.get("id") or f"gsm8k_mcq_{idx:05d}"
            if entry_id in processed:
                pending[idx] = processed[entry_id]
            else:
                tasks.append((idx, original, entry_id))

            while next_to_write in pending:
                fout.write(json.dumps(pending.pop(next_to_write), ensure_ascii=False) + "\n")
                fout.flush()
                if progress_bar:
                    progress_bar.update(1)
                elif ((next_to_write + 1) % 50 == 0) or ((next_to_write + 1) == total):
                    print(f"[progress] {next_to_write + 1}/{total}")
                next_to_write += 1

        # Process remaining entries in parallel
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {
                executor.submit(process_entry, idx, original, entry_id): idx
                for idx, original, entry_id in tasks
            }
            for future in as_completed(future_to_idx):
                idx, obj = future.result()
                pending[idx] = obj

                while next_to_write in pending:
                    fout.write(json.dumps(pending.pop(next_to_write), ensure_ascii=False) + "\n")
                    fout.flush()
                    if progress_bar:
                        progress_bar.update(1)
                    elif ((next_to_write + 1) % 50 == 0) or ((next_to_write + 1) == total):
                        print(f"[progress] {next_to_write + 1}/{total}")
                    next_to_write += 1

        if progress_bar:
            progress_bar.close()

    print(f"[done] Wrote MCQ file to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert GSM8K JSONL to MCQ format using gemini-2.5-flash-thinking.")
    parser.add_argument("--input", type=Path, default=Path(__file__).parent / "test.jsonl", help="Path to source GSM8K JSONL.")
    parser.add_argument("--output", type=Path, help="Where to write MCQ JSONL. Default overwrites input.")
    parser.add_argument("--resume", action="store_true", help="Reuse already written lines in output if present.")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers for API calls.")
    args = parser.parse_args()

    input_path: Path = args.input
    output_path: Path = args.output or input_path

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if output_path.resolve() == input_path.resolve():
        backup = input_path.with_suffix(input_path.suffix + ".bak")
        if not backup.exists():
            backup.write_text(input_path.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[backup] Original file saved to {backup}")

    convert_file(input_path, output_path, resume=args.resume, max_retries=3, workers=args.workers)


if __name__ == "__main__":
    main()
