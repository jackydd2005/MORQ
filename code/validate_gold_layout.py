"""Validate the split between benchmark inputs and reference answers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
if SCRIPT_PATH.parent.name == "code":
    # The public copy is self-contained in release/code/.
    from gold_resolver import GoldResolver
    DEFAULT_ROOT = SCRIPT_PATH.parents[1]
else:
    from dataset_tools.gold_resolver import GoldResolver
    DEFAULT_ROOT = SCRIPT_PATH.parents[2]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate(project_root: Path) -> dict[str, int]:
    questions_path = project_root / "data" / "questions" / "questions.jsonl"
    questions = read_jsonl(questions_path)
    resolver = GoldResolver(project_root)
    gold_records = resolver.records()

    ids = [int(question["id"]) for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question ids")
    if any("label" in question for question in questions):
        raise ValueError("questions.jsonl still contains label fields")

    non_textbook = [question for question in questions if not resolver.is_textbook(question)]
    textbook = [question for question in questions if resolver.is_textbook(question)]
    expected_gold_ids = {int(question["id"]) for question in non_textbook}
    if set(gold_records) != expected_gold_ids:
        missing = sorted(expected_gold_ids - set(gold_records))
        extra = sorted(set(gold_records) - expected_gold_ids)
        raise ValueError(f"Gold/question mismatch: missing={missing[:10]}, extra={extra[:10]}")

    for question in non_textbook:
        resolver.get_non_textbook(question)
    for question in textbook:
        resolver.get_textbook_truth_path(question)

    image_questions = 0
    resolved_root = project_root.resolve()
    for question in questions:
        image_path = str(question.get("image_path", "")).strip()
        if not image_path:
            continue
        image_questions += 1
        relative_path = Path(image_path.replace("\\", "/"))
        candidate = (resolved_root / relative_path).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(f"Image path escapes release root: {image_path}") from exc
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing question image: {image_path}")

    return {
        "questions": len(questions),
        "non_textbook_questions": len(non_textbook),
        "textbook_questions": len(textbook),
        "gold_records": len(gold_records),
        "image_questions": image_questions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(validate(args.root.resolve()), indent=2))


if __name__ == "__main__":
    main()
