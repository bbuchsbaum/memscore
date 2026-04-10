from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def build_manifest(
    scores_csv: Path,
    dataset_root: Path,
    output_path: Path,
    *,
    source_prefix: str = "images/",
    target_prefix: str = "object_images/",
    strict: bool = True,
) -> tuple[Path, int, int, int]:
    dataset_root = dataset_root.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    missing = 0
    invalid_scores = 0
    with scores_csv.expanduser().resolve().open("r", encoding="utf-8", newline="") as handle, output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as out_handle:
        reader = csv.DictReader(handle)
        writer = csv.DictWriter(
            out_handle,
            fieldnames=[
                "path",
                "score",
                "id",
                "concept",
                "image_name",
                "source",
            ],
        )
        writer.writeheader()

        for row in reader:
            rel_path = (row.get("file_path") or "").strip()
            image_name = (row.get("image_name") or "").strip()
            score = (row.get("cr") or "").strip()
            image_id = (row.get("image_nr") or image_name).strip()
            if not rel_path or not image_name or not score:
                continue
            try:
                score_value = float(score)
            except ValueError:
                invalid_scores += 1
                continue
            if not math.isfinite(score_value):
                invalid_scores += 1
                continue

            if source_prefix and rel_path.startswith(source_prefix):
                rel_path = target_prefix + rel_path[len(source_prefix) :]

            parts = Path(rel_path).parts
            concept = parts[-2] if len(parts) >= 2 else ""
            absolute_path = (dataset_root / rel_path).resolve()
            if not absolute_path.exists():
                missing += 1
                if strict:
                    raise FileNotFoundError(f"Missing THINGS image for manifest row: {absolute_path}")
                continue

            writer.writerow(
                {
                    "path": rel_path,
                    "score": f"{score_value:.8g}",
                    "id": image_id,
                    "concept": concept,
                    "image_name": image_name,
                    "source": "THINGS",
                }
            )
            kept += 1

    return output_path, kept, missing, invalid_scores


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a manifest CSV for the THINGS memorability dataset.")
    parser.add_argument(
        "--scores-csv",
        default="data/THINGS/THINGS_Memorability_Scores.csv",
        help="Path to THINGS_Memorability_Scores.csv from the official OSF release.",
    )
    parser.add_argument(
        "--dataset-root",
        default="data/THINGS",
        help="Root directory containing the extracted THINGS image tree.",
    )
    parser.add_argument(
        "--output-path",
        default="data/THINGS/things_memorability_manifest.csv",
        help="Where to write the manifest CSV.",
    )
    parser.add_argument(
        "--source-prefix",
        default="images/",
        help="Path prefix used in THINGS_Memorability_Scores.csv.",
    )
    parser.add_argument(
        "--target-prefix",
        default="object_images/",
        help="Path prefix used by the extracted image archive.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Skip rows whose image files are missing instead of failing.",
    )
    args = parser.parse_args()

    output_path, kept, missing, invalid_scores = build_manifest(
        Path(args.scores_csv),
        Path(args.dataset_root),
        Path(args.output_path),
        source_prefix=args.source_prefix,
        target_prefix=args.target_prefix,
        strict=not args.no_strict,
    )
    print(f"Wrote manifest to {output_path}")
    print(f"Kept {kept} rows")
    print(f"Missing images skipped: {missing}")
    print(f"Invalid scores skipped: {invalid_scores}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
