#!/usr/bin/env python3
"""Recompute textbook field metrics from a released MORQ package.

The script accepts the released extracted-answer snapshot; it never calls a
model provider or requires provider raw completions. It matches every
``<step_n>`` prediction only with the record's same-indexed reaction step.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger


RDLogger.DisableLog("rdApp.*")
ROLES = ("reactants", "reagents", "products")
CONDITION_FIELDS = ("temperature", "time", "operation")
HISTORICAL_STIMULUS_VERSION = "historical_internal_textbook_v1"
PUBLIC_STIMULUS_VERSION = "public_redraw_v1"


def package_root() -> Path:
    script = Path(__file__).resolve()
    # Source lives under src/dataset_tools, while the public copy lives under
    # the package's code directory.
    return script.parents[1] if script.parent.name == "code" else script.parents[2]


def extract_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text or "", re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def canonical_components(value: Any) -> tuple[Counter[str], int]:
    """Return valid canonical components and the count of unparsable parts."""
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"none", "null", "nan", "n/a"}:
        return Counter(), 0
    components: Counter[str] = Counter()
    invalid = 0
    for plus_part in re.split(r"\s+\+\s+", raw):
        for part in plus_part.split("."):
            part = part.strip()
            if not part or part.lower() in {"none", "null", "nan", "n/a"}:
                continue
            molecule = Chem.MolFromSmiles(part)
            if molecule is None:
                invalid += 1
            else:
                components[Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)] += 1
    return components, invalid


def precision_recall_f1(match: int, predicted: int, gold: int) -> tuple[float, float, float]:
    precision = match / predicted if predicted else (1.0 if gold == 0 else 0.0)
    recall = match / gold if gold else (1.0 if predicted == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def normalize_condition(value: Any, field: str) -> str:
    """Normalize only documented notation variants; never infer synonymy."""
    text = unicodedata.normalize("NFKC", str(value or "")).lower().strip()
    if not text or text in {"null", "none", "nan", "n/a"}:
        return ""
    compact = re.sub(r"\s+", "", text)
    if field == "temperature":
        compact = compact.replace("roomtemperature", "rt").replace("室温", "rt")
        compact = compact.replace("r.t.", "rt").replace("r.t", "rt")
        compact = compact.replace("→", "~").replace("->", "~").replace("to", "~")
        tokens = re.findall(r"-?\d+(?:\.\d+)?\s*°?c|(?<![a-z])rt(?![a-z])", compact)
        normal = []
        for token in tokens:
            token = re.sub(r"\s+", "", token)
            if token == "rt":
                normal.append("rt")
            else:
                number = re.match(r"(-?\d+(?:\.\d+)?)", token)
                normal.append(f"{float(number.group(1)):g}c" if number else token)
        return "~".join(normal) if normal else compact
    if field == "time":
        compact = compact.replace("hours", "h").replace("hour", "h").replace("hrs", "h").replace("hr", "h")
        compact = compact.replace("minutes", "min").replace("minute", "min").replace("mins", "min")
        if compact == "overnight":
            return "overnight"
        normal = []
        for token in re.findall(r"\d+(?:\.\d+)?(?:h|min)", compact):
            amount = float(re.match(r"\d+(?:\.\d+)?", token).group(0))
            normal.append(f"{amount * 60 if token.endswith('h') else amount:g}min")
        return "~".join(normal) if normal else compact
    return {"waterseparation": "waterseparator", "4ams": "4ams", "ms4a": "4ams"}.get(compact, compact)


def metric(gold: Counter[str], predicted: Counter[str], invalid: int = 0) -> dict[str, int]:
    match = sum((gold & predicted).values())
    return {
        "match": match,
        "gold_total": sum(gold.values()),
        "pred_total": sum(predicted.values()) + invalid,
        "extra_count": sum((predicted - gold).values()) + invalid,
        "missing_count": sum((gold - predicted).values()),
        "pred_invalid_count": invalid,
    }


def parse_prediction(extracted_answer: str) -> dict[int, dict[str, dict[str, list[str]]]]:
    answer = extract_tag(extracted_answer, "ANS")
    if not answer and "<step_" in (extracted_answer or "").lower():
        answer = extracted_answer.strip()
    matches = list(re.finditer(r"<step_(\d+)>\s*(.*?)\s*</step_\1>", answer, re.DOTALL | re.IGNORECASE))
    blocks = [(int(match.group(1)), match.group(2)) for match in matches] or ([(1, answer)] if answer else [])
    parsed = {}
    for step, block in blocks:
        roles = {role: [] for role in ROLES}
        conditions = {field: [] for field in CONDITION_FIELDS}
        for role in ROLES:
            value = extract_tag(block, role)
            if value:
                roles[role].append(value)
        condition_block = extract_tag(block, "conditions") or block
        for field in CONDITION_FIELDS:
            value = extract_tag(condition_block, field)
            if value:
                conditions[field].append(value)
        parsed[step] = {"roles": roles, "conditions": conditions}
    return parsed


def truth_root(root: Path, stimulus_version: str) -> Path:
    if stimulus_version == HISTORICAL_STIMULUS_VERSION:
        return root / "data" / "reactions"
    if stimulus_version == PUBLIC_STIMULUS_VERSION:
        return root / "data" / "textbook_redraw_truth"
    raise ValueError(f"Unknown textbook stimulus version: {stimulus_version}")


def load_truth(root: Path, org: str, stimulus_version: str) -> dict[int, dict[str, dict[str, list[str]]]]:
    path = truth_root(root, stimulus_version) / org / "reaction.json"
    records = json.loads(path.read_text(encoding="utf-8"))
    steps: dict[int, dict[str, dict[str, list[str]]]] = defaultdict(
        lambda: {"roles": {role: [] for role in ROLES}, "conditions": {field: [] for field in CONDITION_FIELDS}}
    )
    for record in records:
        for role in ROLES:
            for molecule in record.get(role, []) or []:
                if isinstance(molecule, dict) and molecule.get("smiles"):
                    steps[int(molecule.get("step") or 1)]["roles"][role].append(str(molecule["smiles"]))
        for condition in record.get("reaction_conditions", []) or []:
            if isinstance(condition, dict):
                step = int(condition.get("step") or 1)
                for field in CONDITION_FIELDS:
                    value = condition.get(field)
                    if value not in (None, ""):
                        steps[step]["conditions"][field].append(str(value))
    return dict(steps)


def score_record(
    root: Path,
    row: dict[str, Any],
    stimulus_version: str = HISTORICAL_STIMULUS_VERSION,
) -> dict[tuple[str, str], dict[str, int]]:
    truth = load_truth(root, str(row["org"]), stimulus_version)
    prediction = parse_prediction(str(row.get("extracted_answer") or ""))
    fields: dict[tuple[str, str], dict[str, int]] = {}
    for role in ROLES:
        totals = Counter()
        for step in sorted(set(truth) | set(prediction)):
            gold, _ = canonical_components(".".join(truth.get(step, {"roles": {}})["roles"].get(role, [])))
            predicted, invalid = canonical_components(".".join(prediction.get(step, {"roles": {}})["roles"].get(role, [])))
            totals.update(metric(gold, predicted, invalid))
        fields[("role", role)] = dict(totals)
    for field in CONDITION_FIELDS:
        totals = Counter()
        for step in sorted(set(truth) | set(prediction)):
            gold = Counter(filter(None, (normalize_condition(value, field) for value in truth.get(step, {"conditions": {}})["conditions"].get(field, []))))
            predicted = Counter(filter(None, (normalize_condition(value, field) for value in prediction.get(step, {"conditions": {}})["conditions"].get(field, []))))
            totals.update(metric(gold, predicted))
        fields[("condition", field)] = dict(totals)
    return fields


def defaults(root: Path) -> tuple[Path, Path]:
    return root / "evaluation" / "scored_predictions" / "all_model_scored_predictions.jsonl", root / "evaluation" / "canonical_summary" / "textbook_field_metrics_by_model.csv"


def evaluate(root: Path, predictions: Path, stimulus_version: str) -> list[dict[str, Any]]:
    totals: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    with predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "textbook" not in str(row.get("task_type") or row.get("type") or "").lower():
                continue
            row_version = str(row.get("stimulus_version") or stimulus_version)
            if row_version != stimulus_version:
                raise ValueError(
                    f"Prediction stimulus version {row_version!r} does not match requested "
                    f"version {stimulus_version!r} for question {row.get('id')}"
                )
            for (group, field), values in score_record(root, row, stimulus_version).items():
                totals[(str(row.get("model") or ""), group, field)].update(values)
    rows = []
    for (model, group, field), values in sorted(totals.items()):
        precision, recall, f1 = precision_recall_f1(values["match"], values["pred_total"], values["gold_total"])
        rows.append({"model": model, "field_group": group, "field": field, **values,
                     "precision_pct": f"{100 * precision:.2f}", "recall_pct": f"{100 * recall:.2f}", "f1_pct": f"{100 * f1:.2f}"})
    return rows


def self_test() -> None:
    assert normalize_condition("2 h", "time") == normalize_condition("120 min", "time")
    assert normalize_condition("2 h", "time") != normalize_condition("12 h", "time")
    assert normalize_condition("25 C", "temperature") == normalize_condition("25°C", "temperature")
    assert normalize_condition("room temperature", "temperature") == normalize_condition("RT", "temperature")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=package_root())
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stimulus-version",
        choices=[HISTORICAL_STIMULUS_VERSION, PUBLIC_STIMULUS_VERSION],
        default=PUBLIC_STIMULUS_VERSION,
        help="Truth namespace matching the prediction inputs. The public release uses public_redraw_v1.",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("self-test passed")
        return
    predictions, output = defaults(args.root)
    predictions, output = args.predictions or predictions, args.output or output
    rows = evaluate(args.root, predictions, args.stimulus_version)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "field_group", "field", "match", "gold_total", "pred_total", "extra_count", "missing_count", "pred_invalid_count", "precision_pct", "recall_pct", "f1_pct"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
