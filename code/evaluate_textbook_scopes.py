#!/usr/bin/env python3
"""Recompute task-conditioned textbook metrics for a named stimulus version."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from evaluate_textbook_fields import (
    HISTORICAL_STIMULUS_VERSION,
    PUBLIC_STIMULUS_VERSION,
    package_root,
    precision_recall_f1,
    score_record,
)


TARGETS = {
    "Missing Reactants": {"roles": {"reactants"}, "conditions": set()},
    "Missing Products": {"roles": {"products"}, "conditions": set()},
    "Missing Conditions": {"roles": set(), "conditions": {"temperature", "time", "operation"}},
}
ALL_FIELDS = [("role", field) for field in ("reactants", "reagents", "products")] + [
    ("condition", field) for field in ("temperature", "time", "operation")
]


def load_categories(root: Path, stimulus_version: str) -> dict[str, str]:
    filename = (
        "textbook_task_categories.csv"
        if stimulus_version == PUBLIC_STIMULUS_VERSION
        else "historical_textbook_task_categories.csv"
    )
    with (root / "metadata" / filename).open("r", encoding="utf-8", newline="") as handle:
        return {row["org"]: row["category_name"] for row in csv.DictReader(handle)}


def evaluate(root: Path, predictions: Path, stimulus_version: str) -> list[dict[str, object]]:
    categories = load_categories(root, stimulus_version)
    totals: dict[tuple[str, str, str], Counter] = defaultdict(Counter)
    supports: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    with predictions.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if "textbook" not in str(row.get("task_type") or row.get("type") or "").lower():
                continue
            row_version = str(row.get("stimulus_version") or stimulus_version)
            if row_version != stimulus_version:
                raise ValueError(f"Prediction stimulus version {row_version!r} does not match {stimulus_version!r}")
            org = str(row["org"])
            category = str(row.get("textbook_category") or categories[org])
            if category != categories[org] or category not in TARGETS:
                raise ValueError(f"Unknown or mismatched textbook category for {org}: {category}")
            target = TARGETS[category]
            scored = score_record(root, row, stimulus_version)
            model = str(row.get("model") or "")
            for group, field in ALL_FIELDS:
                is_target = (group == "role" and field in target["roles"]) or (
                    group == "condition" and field in target["conditions"]
                )
                scope = "reference_record_agreement" if is_target else "image_transcription"
                if scope == "image_transcription" and group == "condition" and field == "operation":
                    continue
                label = category if is_target else field
                key = (model, scope, label)
                totals[key].update(scored[(group, field)])
                supports[key].add(org)

    rows: list[dict[str, object]] = []
    for (model, scope, label), values in sorted(totals.items()):
        precision, recall, f1 = precision_recall_f1(values["match"], values["pred_total"], values["gold_total"])
        rows.append({
            "model": model,
            "scope": scope,
            "label": label,
            "reaction_support": len(supports[(model, scope, label)]),
            **values,
            "precision_pct": f"{100 * precision:.2f}",
            "recall_pct": f"{100 * recall:.2f}",
            "f1_pct": f"{100 * f1:.2f}",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=package_root())
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--stimulus-version",
        choices=[HISTORICAL_STIMULUS_VERSION, PUBLIC_STIMULUS_VERSION],
        default=HISTORICAL_STIMULUS_VERSION,
    )
    args = parser.parse_args()
    root = args.root.resolve()
    predictions = args.predictions or (root / "evaluation" / "scored_predictions" / "all_model_scored_predictions.jsonl")
    output = args.output or (root / "evaluation" / "canonical_summary" / "textbook_scope_metrics_by_model.csv")
    rows = evaluate(root, predictions, args.stimulus_version)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model", "scope", "label", "reaction_support", "match", "gold_total", "pred_total", "extra_count", "missing_count", "pred_invalid_count", "precision_pct", "recall_pct", "f1_pct"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
