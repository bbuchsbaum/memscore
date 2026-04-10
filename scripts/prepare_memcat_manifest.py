from __future__ import annotations

import argparse
import csv
from pathlib import Path

from membench.memcat import load_memcat_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a generic MemCat manifest for downstream locked splits.")
    parser.add_argument("--memcat-root", required=True, help="Path to the MemCat dataset root.")
    parser.add_argument("--csv-path", help="Optional explicit path to memcat_image_data.csv.")
    parser.add_argument(
        "--score-column",
        default="memorability_w_fa_correction",
        help="MemCat score column to export.",
    )
    parser.add_argument(
        "--output-manifest",
        required=True,
        help="Destination CSV with path, score, id, category, and subcategory columns.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = load_memcat_dataset(
        args.memcat_root,
        csv_path=args.csv_path,
        score_column=args.score_column,
    )
    output_path = Path(args.output_manifest).expanduser().resolve()
    memcat_root = Path(args.memcat_root).expanduser().resolve()

    rows: list[dict[str, object]] = []
    for record in dataset.records:
        rel_path = record.image_path.relative_to(memcat_root / "MemCat")
        parts = rel_path.parts
        rows.append(
            {
                "path": str(rel_path),
                "score": record.score,
                "id": record.identifier,
                "category": parts[0] if len(parts) > 0 else "",
                "subcategory": parts[1] if len(parts) > 1 else "",
                "source": "MemCat",
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "score", "id", "category", "subcategory", "source"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} MemCat rows to {output_path}")


if __name__ == "__main__":
    main()
