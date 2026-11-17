# MORQ

MORQ (Multimodal Organic Reaction Questions) is a manually constructed benchmark for evaluating language and vision-language models on organic-reaction understanding and structured reconstruction.

Version 1.0.0, dated 2026-11-18, contains 197 reaction records and 4,795 questions across 15 task types. The complete release includes public reference answers, 1,580 linked image assets, 28,770 fixed scored observations, evaluation summaries, validation utilities and aggregate evidence from a version-mapped historical internal dataset-quality check.

## Dataset access

The complete versioned dataset is deposited in Zenodo under DOI [10.5281/zenodo.24243813](https://doi.org/10.5281/zenodo.24243813).

This GitHub repository intentionally does not contain the 80.60 MB dataset archive. Zenodo is the version-of-record data host. Download these two files from the Zenodo record:

- `MORQ-v1.0.0.zip`
- `MORQ-v1.0.0.zip.sha256`

The expected SHA-256 value for the v1.0.0 ZIP is:

```text
4433b4b01568bfbf769a6a73877d56e0d164a903de0edb4f31b7f30d7811f949
```

## Repository contents

- `code/`: evaluators, answer resolution and public validation utilities.
- `data/schemas/`: JSON schemas for questions, reactions, gold records and scored outputs.
- `environment.yml`: minimal Conda environment for the released evaluators.
- `CITATION.cff`: preferred dataset citation.
- `LICENSE` and `NOTICE`: data, redraw, code and exclusion terms.

## Validate the downloaded release

Extract `MORQ-v1.0.0.zip`, enter the resulting `MORQ_release` directory and run:

```bash
conda env create -f environment.yml
conda activate morq-release
python -B code/validate_gold_layout.py
python -B code/validate_human_spotcheck.py
python -B code/evaluate_textbook_fields.py --self-test
python -B code/validate_checksums.py
```

To recompute the complete released evaluation without modifying the canonical snapshot:

```bash
python -B code/evaluate_all_tasks.py --root . --output-dir ../morq-recomputed
```

## Public textbook redraws

The public textbook-style subset contains 197 independently rendered stimuli generated from structured reaction records. The release includes 1,182 scored observations measured on these distributed redraws for Claude Opus 4.8, Claude Sonnet 5, GPT-4o, GPT-5.2, Gemini 4 Argon and Gemini 3.8 Flash. Historical textbook images and prediction rows derived from them are not distributed. Fifty-five terminal failures remain explicit zero-score observations.

## Citation

Please cite the Zenodo dataset record and the associated article. The preferred machine-readable citation is provided in `CITATION.cff`.

The author order is Jiaqi Dong, Chenyi Zhou, Ansheng Liu, Yunling Song and Qiang Zhang. Jiaqi Dong and Chenyi Zhou contributed equally; Yunling Song assisted with large-model testing; Qiang Zhang is the corresponding author.

## Licence

The dataset and independently rendered public redraws are released under the Creative Commons Attribution 4.0 International licence. Files under `code/` are released under the MIT License. See `LICENSE` and `NOTICE` for the complete terms and exclusions.
