from __future__ import annotations

import argparse
import csv
from pathlib import Path

from membench.figrim import load_figrim_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export FIGRIM to a generic scored-image manifest.")
    parser.add_argument("--figrim-root", required=True, help="Path to the FIGRIM download directory.")
    parser.add_argument(
        "--protocol",
        default="across",
        help="FIGRIM protocol: across, within, across_category, or within_category.",
    )
    parser.add_argument(
        "--score-type",
        default="corrected_hit_rate",
        help="FIGRIM target definition: corrected_hit_rate or hit_rate.",
    )
    parser.add_argument(
        "--output-manifest",
        required=True,
        help="Destination CSV manifest containing path, score, id, and category.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    dataset = load_figrim_dataset(args.figrim_root, protocol=args.protocol, score_type=args.score_type)
    root = Path(args.figrim_root).expanduser().resolve()
    output_path = Path(args.output_manifest).expanduser().resolve()

    rows = []
    for record, category in zip(dataset.records, dataset.categories):
        rows.append(
            {
                "path": str(record.image_path.relative_to(root)),
                "score": f"{record.score:.8f}",
                "id": record.identifier,
                "category": category,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "score", "id", "category"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote FIGRIM manifest to {output_path}")


if __name__ == "__main__":
    main()
