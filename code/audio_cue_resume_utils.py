from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


Q_PATTERN = re.compile(
    r"Q\d+\s+id=([^|]+)\s+\|\s+baseline_pred=([^|]+)\s+\|\s+gold=([^|]+)\s+\|\s+cue_type=([^|]+)\s+\|\s+followup_pred=([^|]*)\s+\|\s+followup_correct=(True|False)"
)

BASELINE_PATTERN = re.compile(r"Q\d+\s+id=([^|]+)\s+\|")


def parse_existing_records(paths: list[Path]) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            match = Q_PATTERN.search(line)
            if not match:
                continue
            sample_id, baseline_pred, gold, cue_type, followup_pred, followup_correct = match.groups()
            records[sample_id.strip()] = {
                "baseline_pred": baseline_pred.strip(),
                "gold": gold.strip(),
                "cue_type": cue_type.strip(),
                "followup_pred": followup_pred.strip(),
                "followup_correct": followup_correct == "True",
            }
    return records


def cmd_prepare(args: argparse.Namespace) -> None:
    existing = parse_existing_records([Path(p) for p in args.existing_logs])
    remaining_lines: list[str] = []
    for line in args.baseline_log.read_text(encoding="utf-8").splitlines():
        match = BASELINE_PATTERN.search(line)
        if not match:
            continue
        sample_id = match.group(1).strip()
        if sample_id in existing:
            continue
        remaining_lines.append(line + "\n")

    shard_lines = [remaining_lines[0::2], remaining_lines[1::2]]
    outputs = [args.shard0, args.shard1]
    for output, lines in zip(outputs, shard_lines):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("".join(lines), encoding="utf-8")

    summary = {
        "existing": len(existing),
        "remaining": len(remaining_lines),
        "shard0": len(shard_lines[0]),
        "shard1": len(shard_lines[1]),
    }
    print(json.dumps(summary, ensure_ascii=False))


def cmd_summarize(args: argparse.Namespace) -> None:
    records = parse_existing_records([Path(p) for p in args.logs])
    total_correct = 0
    total_wrong = 0
    mss_changed = 0
    crs_fixed = 0
    for item in records.values():
        baseline_correct = item["baseline_pred"] == item["gold"] and item["gold"] in {"A", "B", "C", "D"}
        total_correct += int(baseline_correct)
        total_wrong += int(not baseline_correct)
        if baseline_correct and not item["followup_correct"]:
            mss_changed += 1
        if (not baseline_correct) and item["followup_correct"]:
            crs_fixed += 1

    summary = {
        "processed": len(records),
        "initial_correct": total_correct,
        "initial_wrong": total_wrong,
        "mss_changed": mss_changed,
        "crs_fixed": crs_fixed,
        "mss_rate": (mss_changed / total_correct * 100) if total_correct else 0.0,
        "crs_rate": (crs_fixed / total_wrong * 100) if total_wrong else 0.0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    prepare = sub.add_parser("prepare")
    prepare.add_argument("--baseline-log", type=Path, required=True)
    prepare.add_argument("--existing-logs", nargs="*", default=[])
    prepare.add_argument("--shard0", type=Path, required=True)
    prepare.add_argument("--shard1", type=Path, required=True)

    summarize = sub.add_parser("summarize")
    summarize.add_argument("--logs", nargs="+", required=True)
    summarize.add_argument("--out", type=Path, required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "prepare":
        cmd_prepare(args)
    else:
        cmd_summarize(args)


if __name__ == "__main__":
    main()
