"""Verify every SHA-256 entry in a MORQ release package."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(root: Path) -> dict[str, object]:
    root = root.resolve()
    manifest = root / "metadata" / "checksums.sha256"
    errors: list[str] = []
    checked = 0
    if not manifest.is_file():
        return {"status": "FAIL", "checked": 0, "errors": ["metadata/checksums.sha256 is missing"]}

    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"Malformed checksum entry on line {line_number}")
            continue
        path = root / Path(relative)
        if not path.is_file():
            errors.append(f"Missing file: {relative}")
            continue
        observed = sha256(path)
        checked += 1
        if observed != expected:
            errors.append(f"Checksum mismatch: {relative}")

    return {"status": "PASS" if not errors else "FAIL", "checked": checked, "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = validate(args.root)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
