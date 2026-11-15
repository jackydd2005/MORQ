"""Validate the public historical human-spot-check aggregate and crosswalk."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


EXPECTED_PRIMARY = {
    "items": 50,
    "raw_interactions": 1633,
    "user_identifiers": 26,
    "user_item_trajectories": 1209,
    "first_label_agreements": 802,
    "first_label_agreement_pct": 66.34,
    "final_label_agreements": 1119,
    "final_label_agreement_pct": 92.56,
    "repeated_trajectories": 375,
    "incorrect_to_correct": 327,
    "correct_to_incorrect": 10,
    "maximum_attempts": 6,
    "explicit_error_option_E": 0,
    "nonempty_feedback": 0,
}

EXPECTED_ALL = {
    "items": 52,
    "raw_interactions": 1636,
    "user_identifiers": 27,
    "user_item_trajectories": 1212,
    "first_label_agreements": 805,
    "first_label_agreement_pct": 66.42,
    "final_label_agreements": 1122,
    "final_label_agreement_pct": 92.57,
    "repeated_trajectories": 375,
    "incorrect_to_correct": 327,
    "correct_to_incorrect": 10,
    "maximum_attempts": 6,
    "explicit_error_option_E": 0,
    "nonempty_feedback": 0,
}

EXPECTED_STATUSES = {
    "pilot_only_unreconstructed": 2,
    "reusable": 13,
    "answer_set_carryover": 5,
    "changed": 29,
    "retired": 3,
}

EXPECTED_VERSION_ROWS = {
    "exact": (0, 0),
    "semantically_reusable": (13, 13),
    "answer_set_carryover": (5, 5),
    "mapped_but_changed": (29, 13),
    "retired": (3, 0),
    "unmatched": (0, 0),
}

SUM_FIELDS = (
    "raw_interactions",
    "user_item_trajectories",
    "first_label_agreements",
    "final_label_agreements",
    "repeated_trajectories",
    "incorrect_to_correct",
    "correct_to_incorrect",
    "explicit_error_option_E",
    "nonempty_feedback",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def as_int(row: dict[str, str], field: str) -> int:
    return int(row[field])


def as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Expected true/false, found {value!r}")
    return normalized == "true"


def default_root() -> Path:
    script = Path(__file__).resolve()
    candidates = (
        Path.cwd(),
        script.parents[1],
        script.parents[2] / "release" / "MORQ_release",
    )
    for candidate in candidates:
        if (candidate / "data" / "questions" / "questions.jsonl").is_file():
            return candidate
    return Path.cwd()


def validate(root: Path) -> dict[str, Any]:
    errors: list[str] = []
    root = root.resolve()
    spot = root / "metadata" / "human_spotcheck"
    summary_path = spot / "summary.json"
    item_path = spot / "human_spotcheck_item_summary.csv"
    crosswalk_path = spot / "question_version_crosswalk.csv"
    version_path = spot / "version_comparison_summary.csv"
    questions_path = root / "data" / "questions" / "questions.jsonl"
    gold_path = root / "data" / "gold" / "non_textbook.jsonl"

    required = (summary_path, item_path, crosswalk_path, version_path, questions_path, gold_path)
    missing = [path.relative_to(root).as_posix() for path in required if not path.is_file()]
    if missing:
        return {"status": "FAIL", "errors": [f"Missing required file: {path}" for path in missing]}

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    item_rows, item_fields = read_csv(item_path)
    crosswalk, crosswalk_fields = read_csv(crosswalk_path)
    version_rows, version_fields = read_csv(version_path)

    forbidden_exact_fields = {"user_id", "username", "timestamp", "selected_option", "feedback"}
    for name, fields in (
        (item_path.name, item_fields),
        (crosswalk_path.name, crosswalk_fields),
        (version_path.name, version_fields),
    ):
        leaked = forbidden_exact_fields.intersection(fields)
        if leaked:
            errors.append(f"{name} contains disallowed user-level fields: {sorted(leaked)}")

    if len(item_rows) != 52:
        errors.append(f"Expected 52 item-summary rows, found {len(item_rows)}")
    if len(crosswalk) != 52:
        errors.append(f"Expected 52 crosswalk rows, found {len(crosswalk)}")

    item_keys = {(row["legacy_rxn"], row["legacy_question_id"]) for row in item_rows}
    crosswalk_keys = {(row["legacy_rxn"], row["legacy_question_id"]) for row in crosswalk}
    if len(item_keys) != len(item_rows):
        errors.append("Item-summary legacy keys are not unique")
    if len(crosswalk_keys) != len(crosswalk):
        errors.append("Crosswalk legacy keys are not unique")
    if item_keys != crosswalk_keys:
        errors.append("Item-summary and crosswalk legacy keys differ")

    statuses = Counter(row["status"] for row in crosswalk)
    if dict(statuses) != EXPECTED_STATUSES:
        errors.append(f"Crosswalk status counts differ: {dict(statuses)}")

    primary_items = [row for row in item_rows if row["item_role"] == "primary_spotcheck"]
    pilot_items = [row for row in item_rows if row["item_role"] == "interface_pilot"]
    if len(primary_items) != 50 or len(pilot_items) != 2:
        errors.append(f"Expected 50 primary and 2 pilot rows, found {len(primary_items)} and {len(pilot_items)}")

    for role, rows, expected in (
        ("primary", primary_items, EXPECTED_PRIMARY),
        ("all", item_rows, EXPECTED_ALL),
    ):
        observed = {field: sum(as_int(row, field) for row in rows) for field in SUM_FIELDS}
        observed["items"] = len(rows)
        observed["maximum_attempts"] = max(as_int(row, "maximum_attempts") for row in rows)
        # User identifiers cannot be summed across items; the public summary reports the global distinct count.
        observed["user_identifiers"] = int(summary[f"{role}_50_item_spotcheck" if role == "primary" else "all_recorded_versions"]["user_identifiers"]) if role == "primary" else int(summary["all_recorded_versions"]["user_identifiers"])
        observed["first_label_agreement_pct"] = round(100 * observed["first_label_agreements"] / observed["user_item_trajectories"], 2)
        observed["final_label_agreement_pct"] = round(100 * observed["final_label_agreements"] / observed["user_item_trajectories"], 2)
        for field, value in expected.items():
            if observed[field] != value:
                errors.append(f"{role} aggregate {field}: expected {value}, found {observed[field]}")

    for section, expected in (
        ("primary_50_item_spotcheck", EXPECTED_PRIMARY),
        ("all_recorded_versions", EXPECTED_ALL),
    ):
        payload = summary.get(section, {})
        for field, value in expected.items():
            if payload.get(field) != value:
                errors.append(f"summary.json {section}.{field}: expected {value}, found {payload.get(field)!r}")

    if sum(as_int(row, "raw_interactions") for row in pilot_items) != 3:
        errors.append("The two pilot rows must comprise three interactions")

    version_index = {row["category"]: row for row in version_rows}
    if set(version_index) != set(EXPECTED_VERSION_ROWS):
        errors.append(f"Unexpected version-summary categories: {sorted(version_index)}")
    else:
        for category, (legacy_count, current_count) in EXPECTED_VERSION_ROWS.items():
            row = version_index[category]
            observed = (as_int(row, "legacy_items"), as_int(row, "current_questions"))
            if observed != (legacy_count, current_count):
                errors.append(f"Version summary {category}: expected {(legacy_count, current_count)}, found {observed}")

    questions = read_jsonl(questions_path)
    gold = read_jsonl(gold_path)
    question_index = {(str(row["org"]), int(row["id"]), str(row["type"])): row for row in questions}
    gold_index = {(str(row["org"]), int(row["id"]), str(row["type"])): row for row in gold}

    primary_crosswalk = [row for row in crosswalk if row["status"] != "pilot_only_unreconstructed"]
    current_org_rows = [row for row in primary_crosswalk if row["current_org"]]
    current_question_rows = [row for row in primary_crosswalk if row["current_question_id"]]
    if len(current_org_rows) != 49 or len({row["current_org"] for row in current_org_rows}) != 32:
        errors.append("Expected 49 current-reaction traces across 32 reaction records")
    mapped_keys = {(row["current_org"], int(row["current_question_id"])) for row in current_question_rows}
    if len(current_question_rows) != 47 or len(mapped_keys) != 30:
        errors.append("Expected 47 mapped legacy items across 30 current questions")

    task_type = "missing_all_reactants_mcq_image"
    for row in current_question_rows:
        key = (row["current_org"], int(row["current_question_id"]), task_type)
        if key not in question_index:
            errors.append(f"Mapped current question is absent: {key}")
            continue
        if key not in gold_index:
            errors.append(f"Mapped current gold record is absent: {key}")
            continue
        current_label = str(gold_index[key]["gold_value"])
        if current_label != row["current_gold_label"]:
            errors.append(f"Current label mismatch for {key}: crosswalk={row['current_gold_label']} gold={current_label}")
        try:
            if as_bool(row["label_same"]) != (row["legacy_gold_label"] == row["current_gold_label"]):
                errors.append(f"label_same is inconsistent for {row['legacy_rxn']} question {row['legacy_question_id']}")
        except ValueError as exc:
            errors.append(str(exc))

    for row in primary_crosswalk:
        status = row["status"]
        try:
            reaction_same = as_bool(row["reaction_json_exact"])
            target_same = as_bool(row["reactant_gold_same"])
            exact_item = as_bool(row["exact_item"])
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if exact_item:
            errors.append(f"No historical item should be marked exact: {row['legacy_rxn']}")
        if status == "reusable" and not (reaction_same and target_same):
            errors.append(f"Reusable classification is inconsistent for {row['legacy_rxn']}")
        if status == "answer_set_carryover" and not (not reaction_same and target_same):
            errors.append(f"Answer-set carryover classification is inconsistent for {row['legacy_rxn']}")
        if status == "changed" and target_same:
            errors.append(f"Changed classification is inconsistent for {row['legacy_rxn']}")

    integrity = summary.get("integrity", {})
    hash_checks = {
        "current_questions_jsonl_sha256": sha256(questions_path),
        "current_non_textbook_gold_sha256": sha256(gold_path),
    }
    for field, value in hash_checks.items():
        if integrity.get(field) != value:
            errors.append(f"Integrity mismatch for {field}: expected {integrity.get(field)!r}, found {value}")

    return {
        "status": "PASS" if not errors else "FAIL",
        "primary_items": len(primary_items),
        "all_recorded_versions": len(item_rows),
        "mapped_legacy_items": len(current_question_rows),
        "distinct_current_questions": len(mapped_keys),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root(), help="MORQ release-package root")
    args = parser.parse_args()
    result = validate(args.root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
