#!/usr/bin/env python3
"""Run one reproducibility entry point for all released MORQ task types.

The evaluator works from the released per-prediction scoring snapshot.  It
validates coverage and writes one task-level summary, the textbook conditioned
field summaries, and (when the archived raw responses are supplied) a compact
answer-wrapper compliance sidecar.  Raw provider responses are never copied
into the release; only derived format flags are retained.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


MODELS = (
    "claude-opus-4-8",
    "claude-sonnet-5",
    "gpt-4o",
    "gpt-5.2",
    "gemini-4-argon",
    "gemini-3.8-flash",
)
TEXTBOOK_CATEGORIES = ("Missing Reactants", "Missing Products", "Missing Conditions")
ROLE_FIELDS = ("reactants", "reagents", "products")
CONDITION_FIELDS = ("temperature", "time", "operation")
TARGETS = {
    "Missing Reactants": {("role", "reactants")},
    "Missing Products": {("role", "products")},
    "Missing Conditions": {("condition", field) for field in CONDITION_FIELDS},
}
REPLACEMENT_MODELS = ("claude-sonnet-5", "gemini-4-argon", "gemini-3.8-flash")
METRIC_FIELDS = (
    "match", "gold_total", "pred_total", "extra_count", "missing_count", "pred_invalid_count"
)


def package_root() -> Path:
    script = Path(__file__).resolve()
    return script.parents[1] if script.parent.name == "code" else script.parents[2]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_questions(root: Path) -> dict[tuple[int, str], dict[str, Any]]:
    path = root / "data" / "questions" / "questions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Question table not found: {path}")
    rows = read_jsonl(path)
    index = {(int(row["id"]), str(row["type"])): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Question table contains duplicate (id, type) keys")
    return index


def add_counts(target: Counter[str], values: dict[str, Any]) -> None:
    for field in METRIC_FIELDS:
        target[field] += int(values.get(field, 0) or 0)


def ratios(counts: Counter[str]) -> dict[str, float]:
    match = counts["match"]
    predicted = counts["pred_total"]
    gold = counts["gold_total"]
    precision = match / predicted if predicted else (1.0 if gold == 0 else 0.0)
    recall = match / gold if gold else (1.0 if predicted == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision_pct": precision * 100, "recall_pct": recall * 100, "f1_pct": f1 * 100}


def task_profile(row: dict[str, Any]) -> str:
    if row.get("family") == "textbook_analysis":
        return "textbook_scoped_fields"
    if row.get("answer_format") == "mcq":
        return "mcq_primary_accuracy"
    if row.get("answer_format") == "smiles_fitb":
        return "smiles_strict_and_non_isomeric"
    return "task_specific_primary_accuracy"


def build_task_summary(rows: list[dict[str, Any]], format_rows: dict[tuple[str, int, str], dict[str, Any]] | None) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["model"]), str(row["task_type"]))
        if key not in groups:
            groups[key] = {
                "model": key[0], "task_type": key[1], "family": row.get("family", ""),
                "modality": row.get("modality", ""), "answer_format": row.get("answer_format", ""),
                "total": 0, "scorable": 0, "primary_correct": 0, "strict_correct": 0,
                "nostereo_correct": 0, "invalid_pred_parts": 0, "format_observed": 0,
                "format_compliant": 0,
            }
        group = groups[key]
        group["total"] += 1
        group["scorable"] += int(bool(row.get("scorable")))
        group["primary_correct"] += int(bool(row.get("correct_primary")))
        group["strict_correct"] += int(bool(row.get("strict_correct")))
        group["nostereo_correct"] += int(bool(row.get("nostereo_correct")))
        group["invalid_pred_parts"] += int(row.get("pred_invalid_count", 0) or 0)
        if format_rows is not None:
            flags = format_rows.get((str(row["model"]), int(row["id"]), str(row["task_type"])))
            if flags and flags.get("raw_available"):
                group["format_observed"] += 1
                group["format_compliant"] += int(bool(flags.get("format_compliant")))

    output = []
    for key in sorted(groups):
        row = groups[key]
        total = row["total"]
        observed = row["format_observed"]
        output.append({
            **row,
            "primary_accuracy_pct": 100 * row["primary_correct"] / total if total else 0.0,
            "strict_accuracy_pct": 100 * row["strict_correct"] / total if total else 0.0,
            "nostereo_accuracy_pct": 100 * row["nostereo_correct"] / total if total else 0.0,
            "invalid_pred_parts_per_100_questions": 100 * row["invalid_pred_parts"] / total if total else 0.0,
            "format_compliance_pct": 100 * row["format_compliant"] / observed if observed else "",
            "format_observed": observed,
            "evaluation_profile": task_profile(row),
        })
    return output


def format_flags(raw: str) -> dict[str, Any]:
    text = raw or ""
    opens = len(re.findall(r"<ANS\b[^>]*>", text, flags=re.IGNORECASE))
    closes = len(re.findall(r"</ANS\s*>", text, flags=re.IGNORECASE))
    blocks = list(re.finditer(r"<ANS\b[^>]*>(.*?)</ANS\s*>", text, flags=re.IGNORECASE | re.DOTALL))
    trailing = text[blocks[-1].end():].strip() if blocks else ""
    return {
        "raw_available": True,
        "raw_nonempty": bool(text.strip()),
        "ans_open_count": opens,
        "ans_close_count": closes,
        "ans_block_count": len(blocks),
        "trailing_text": bool(trailing),
        "wrapper_present": bool(blocks),
        # The system contract requires one complete block and no text after it.
        "format_compliant": len(blocks) == 1 and opens == 1 and closes == 1 and not trailing,
    }


def iter_text_raw(raw_root: Path) -> Iterable[dict[str, Any]]:
    base = raw_root / "data" / "reactions"
    if not base.exists():
        return
    for org_dir in base.glob("org_*"):
        respond = org_dir / "respond"
        if not respond.exists():
            continue
        for task_dir in respond.iterdir():
            if not task_dir.is_dir():
                continue
            for id_dir in task_dir.iterdir():
                if not id_dir.is_dir() or not id_dir.name.isdigit():
                    continue
                for path in id_dir.glob("*.txt"):
                    yield {
                        "id": int(id_dir.name), "org": org_dir.name, "task_type": task_dir.name,
                        "model": path.stem, "raw": path.read_text(encoding="utf-8", errors="replace"),
                        "source": path.relative_to(raw_root).as_posix(),
                    }


def iter_replacement_raw(raw_root: Path) -> Iterable[dict[str, Any]]:
    base = raw_root / "results" / "model_outputs" / "replacement_v1_raw"
    for model in REPLACEMENT_MODELS:
        for path in (base / model).glob("*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            yield {
                "id": int(row["question_id"]), "org": str(row["org"]),
                "task_type": str(row["task_type"]), "model": model,
                "raw": str(row.get("content") or ""),
                "source": path.relative_to(raw_root).as_posix(),
            }


def build_format_sidecar(
    rows: list[dict[str, Any]], question_index: dict[tuple[int, str], dict[str, Any]], raw_root: Path | None
) -> dict[tuple[str, int, str], dict[str, Any]] | None:
    if raw_root is None:
        return None
    raw_rows = list(iter_text_raw(raw_root)) + list(iter_replacement_raw(raw_root))
    raw_index: dict[tuple[str, int, str], dict[str, Any]] = {}
    for raw in raw_rows:
        key = (raw["model"], raw["id"], raw["task_type"])
        if key in raw_index:
            raise ValueError(f"Duplicate raw response key: {key}")
        raw_index[key] = {**raw, **format_flags(raw["raw"])}
    expected = {(str(row["model"]), int(row["id"]), str(row["task_type"])) for row in rows}
    missing = sorted(expected - set(raw_index))
    if missing:
        raise ValueError(f"Raw response archive is missing {len(missing)} scored rows; first keys: {missing[:3]}")
    return raw_index


def write_format_sidecar(path: Path, rows: dict[tuple[str, int, str], dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "org", "task_type", "model", "raw_source", "raw_available", "raw_nonempty",
              "ans_open_count", "ans_close_count", "ans_block_count", "trailing_text",
              "wrapper_present", "format_compliant"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(rows):
            row = rows[key]
            writer.writerow({
                "id": row["id"], "org": row["org"], "task_type": row["task_type"], "model": row["model"],
                "raw_source": row["source"], **{field: row[field] for field in fields[5:]},
            })


def read_format_sidecar(path: Path | None) -> dict[tuple[str, int, str], dict[str, Any]] | None:
    if path is None or not path.exists():
        return None
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["model"], int(row["id"]), row["task_type"])
            for field in ("raw_available", "raw_nonempty", "trailing_text", "wrapper_present", "format_compliant"):
                row[field] = row.get(field, "").lower() == "true"
            for field in ("ans_open_count", "ans_close_count", "ans_block_count"):
                row[field] = int(row.get(field, 0) or 0)
            rows[key] = row
    return rows


def build_format_summary(rows: list[dict[str, Any]], sidecar: dict[tuple[str, int, str], dict[str, Any]] | None) -> list[dict[str, Any]]:
    if sidecar is None:
        return []
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        key = (str(row["model"]), str(row["task_type"]))
        flags = sidecar[(key[0], int(row["id"]), key[1])]
        grouped[key]["total"] += 1
        grouped[key]["wrapper_present"] += int(flags["wrapper_present"])
        grouped[key]["format_compliant"] += int(flags["format_compliant"])
        grouped[key]["trailing_text"] += int(flags["trailing_text"])
        grouped[key]["multiple_blocks"] += int(flags["ans_block_count"] > 1)
    output = []
    for (model, task_type), counts in sorted(grouped.items()):
        total = counts["total"]
        output.append({
            "model": model, "task_type": task_type, "total": total,
            "wrapper_present": counts["wrapper_present"],
            "format_compliant": counts["format_compliant"],
            "trailing_text": counts["trailing_text"],
            "multiple_blocks": counts["multiple_blocks"],
            "wrapper_present_pct": 100 * counts["wrapper_present"] / total if total else 0.0,
            "format_compliant_pct": 100 * counts["format_compliant"] / total if total else 0.0,
        })
    return output


def build_textbook_by_format(rows: list[dict[str, Any]], sidecar: dict[tuple[str, int, str], dict[str, Any]] | None) -> list[dict[str, Any]]:
    statuses = ("all", "format_compliant", "format_noncompliant") if sidecar is not None else ("all",)
    totals: dict[tuple[str, str, str, str], Counter[str]] = defaultdict(Counter)
    orgs: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        if row.get("family") != "textbook_analysis":
            continue
        model = str(row["model"])
        category = str(row.get("textbook_category") or "")
        if category not in TARGETS:
            raise ValueError(f"Missing/unknown textbook category for {row.get('org')}: {category}")
        status_list = ["all"]
        if sidecar is not None:
            compliant = bool(sidecar[(model, int(row["id"]), str(row["task_type"]))]["format_compliant"])
            status_list.append("format_compliant" if compliant else "format_noncompliant")
        for status in status_list:
            for group, fields, key in (
                ("role", ROLE_FIELDS, "textbook_role_metrics"),
                ("condition", CONDITION_FIELDS, "textbook_condition_metrics"),
            ):
                for field in fields:
                    is_target = (group, field) in TARGETS[category]
                    scope = "reference_record_agreement" if is_target else "image_transcription"
                    if scope == "image_transcription" and group == "condition" and field == "operation":
                        continue
                    label = category if is_target else field
                    values = (row.get(key) or {}).get(field) or {}
                    index = (model, status, scope, label)
                    add_counts(totals[index], values)
                    orgs[index].add(str(row["org"]))
    output = []
    for index in sorted(totals):
        model, status, scope, label = index
        counts = totals[index]
        output.append({
            "model": model, "format_status": status, "scope": scope, "label": label,
            "reaction_support": len(orgs[index]), **counts, **ratios(counts),
        })
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=package_root(), help="Project root or MORQ release root")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--raw-root", type=Path, help="Workspace root containing raw response archives")
    parser.add_argument("--format-sidecar", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    # Source mode points at release/MORQ_release; public-copy mode points at its package root.
    if (root / "release" / "MORQ_release").exists():
        release = root / "release" / "MORQ_release"
        project_root = root
    else:
        release = root
        project_root = root.parent.parent if root.parent.name == "release" else root
    predictions = (args.predictions or (release / "evaluation" / "scored_predictions" / "all_model_scored_predictions.jsonl")).resolve()
    output_dir = (args.output_dir or (release / "evaluation" / "canonical_summary")).resolve()
    question_root = project_root if (project_root / "release" / "MORQ_release" / "data" / "questions" / "questions.jsonl").exists() else release
    questions = load_questions(question_root / "release" / "MORQ_release" if (question_root / "release" / "MORQ_release").exists() else question_root)
    rows = read_jsonl(predictions)
    if len(rows) != 28770:
        raise ValueError(f"Expected 28,770 scored rows, found {len(rows)}")
    if {str(row["model"]) for row in rows} != set(MODELS):
        raise ValueError("Scored snapshot model set does not match the six released models")
    if len({str(row["task_type"]) for row in rows}) != 15:
        raise ValueError("Scored snapshot does not contain 15 task types")
    rows_by_model = Counter(str(row["model"]) for row in rows)
    expected_by_model = Counter({model: 4795 for model in MODELS})
    if rows_by_model != expected_by_model:
        raise ValueError(f"Scored snapshot model coverage mismatch: {dict(rows_by_model)}")
    textbook_rows = [row for row in rows if row.get("family") == "textbook_analysis"]
    if len(textbook_rows) != 1182 or {str(row.get("stimulus_version")) for row in textbook_rows} != {"public_redraw_v1"}:
        raise ValueError("Expected 1,182 public_redraw_v1 textbook rows")
    question_keys = {(int(row["id"]), str(row["task_type"])) for row in rows}
    if not question_keys.issubset(questions):
        raise ValueError("Scored snapshot contains question keys absent from questions.jsonl")

    sidecar_path = (args.format_sidecar or (release / "evaluation" / "scored_predictions" / "format_compliance_by_prediction.csv")).resolve()
    sidecar = None
    if args.raw_root:
        sidecar = build_format_sidecar(rows, questions, args.raw_root.resolve())
        write_format_sidecar(sidecar_path, sidecar)
    else:
        sidecar = read_format_sidecar(sidecar_path)

    task_rows = build_task_summary(rows, sidecar)
    format_rows = build_format_summary(rows, sidecar)
    scoped_rows = build_textbook_by_format(rows, sidecar)
    write_csv(output_dir / "unified_task_metrics.csv", task_rows, list(task_rows[0].keys()))
    write_csv(output_dir / "format_compliance_by_model_task.csv", format_rows, list(format_rows[0].keys()) if format_rows else ["model", "task_type", "total"])
    write_csv(output_dir / "textbook_scope_metrics_by_format_status.csv", scoped_rows, list(scoped_rows[0].keys()))

    manifest = {
        "scored_rows": len(rows), "models": sorted({str(row["model"]) for row in rows}),
        "task_types": sorted({str(row["task_type"]) for row in rows}),
        "task_type_count": len({str(row["task_type"]) for row in rows}),
        "predictions_sha256": hashlib.sha256(predictions.read_bytes()).hexdigest(),
        "format_sidecar": sidecar_path.relative_to(release).as_posix() if sidecar_path.is_relative_to(release) else str(sidecar_path),
        "format_definition": "One opening <ANS> tag, one closing </ANS> tag, one matched block, and no non-whitespace text after the closing tag.",
        "format_source": "Derived from archived raw responses when --raw-root is supplied; raw responses are not copied into the public release.",
        "textbook_scope_definition": "Category-designated fields are reference-record agreement; non-target fields are image transcription; operation is omitted from the plotted transcription scope.",
    }
    (output_dir / "unified_evaluation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "task_types": 15, "format_sidecar": str(sidecar_path), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()


