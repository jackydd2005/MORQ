# Release helpers

`gold_resolver.py` resolves released reference answers. `validate_gold_layout.py` checks question, gold-record, reaction-record and image linkage. `validate_human_spotcheck.py` checks the public human-spot-check aggregates, crosswalk arithmetic and current question/gold linkages. `validate_checksums.py` verifies every file declared in `metadata/checksums.sha256` with the Python standard library. `evaluate_textbook_fields.py` recomputes the complete textbook field metrics from released extracted-answer records, while `evaluate_textbook_scopes.py` recomputes the conditioned image-transcription and reference-record-agreement metrics.

`evaluate_all_tasks.py` is the single coverage entry point for all 15 task types. It validates 4,598 non-textbook questions for each of six models plus 197 public-redraw textbook questions for each of three available models (28,179 scored rows in total), writes unified task summaries, and writes the textbook scope/format summaries. The derived `evaluation/scored_predictions/format_compliance_by_prediction.csv` contains no raw responses; it records only whether each fixed baseline response had exactly one complete `<ANS>...</ANS>` block with no trailing text. The evaluator requires Python and RDKit 2025.09.3; the supplied `environment.yml` declares the environment.

Run the evaluator from the package root with a separate destination, for example:

```bash
python -B code/evaluate_all_tasks.py --root . --output-dir ../morq-recomputed
```

Omitting `--output-dir` writes into `evaluation/canonical_summary/` and can replace the released canonical summary. Use `python -B` for release validation and recomputation so that bytecode caches are not created inside the package. `baseline_protocol.json` documents the fixed baseline request protocol.
