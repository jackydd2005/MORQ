"""Single entry point for MORQ reference answers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class GoldResolver:
    """Resolve non-textbook answers without duplicating textbook reaction truth."""

    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.gold_path = self.project_root / "data" / "gold" / "non_textbook.jsonl"
        self._by_id: dict[int, dict[str, Any]] | None = None

    def _load(self) -> dict[int, dict[str, Any]]:
        if self._by_id is not None:
            return self._by_id
        if not self.gold_path.exists():
            raise FileNotFoundError(f"Missing non-textbook gold file: {self.gold_path}")

        by_id: dict[int, dict[str, Any]] = {}
        with self.gold_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                try:
                    question_id = int(record["id"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"Invalid gold record at {self.gold_path}:{line_no}") from exc
                if question_id in by_id:
                    raise ValueError(f"Duplicate gold record for question id {question_id}")
                if not isinstance(record.get("gold_value"), str) or not record["gold_value"].strip():
                    raise ValueError(f"Missing gold_value for question id {question_id}")
                by_id[question_id] = record
        self._by_id = by_id
        return by_id

    @staticmethod
    def is_textbook(question: dict[str, Any]) -> bool:
        return "textbook" in str(question.get("type", "")).lower()

    def get_non_textbook(self, question: dict[str, Any]) -> str:
        """Return the reference answer and verify its task identity."""
        if self.is_textbook(question):
            raise ValueError("Textbook truth must be read from its reaction.json record")
        question_id = int(question["id"])
        record = self._load().get(question_id)
        if record is None:
            raise KeyError(f"No gold record for non-textbook question id {question_id}")
        for field in ("org", "type"):
            if str(record.get(field, "")) != str(question.get(field, "")):
                raise ValueError(f"Gold {field} mismatch for question id {question_id}")
        return str(record["gold_value"]).strip()

    def get_textbook_truth_path(self, question: dict[str, Any]) -> Path:
        """Return the truth record bound to a public textbook question row."""
        if not self.is_textbook(question):
            raise ValueError("Non-textbook truth is stored in data/gold/non_textbook.jsonl")
        declared = str(question.get("reference_truth_source") or "").strip()
        if declared:
            path = (self.project_root / Path(declared.replace("\\", "/"))).resolve()
            path.relative_to(self.project_root.resolve())
        else:
            path = self.project_root / "data" / "reactions" / str(question["org"]) / "reaction.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing textbook truth record: {path}")
        return path

    def records(self) -> dict[int, dict[str, Any]]:
        return dict(self._load())
