from __future__ import annotations

import argparse
import csv
from pathlib import Path

from membench.data import load_lamem_splits


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a locked manifest CSV from official LaMem split files.")
    parser.add_argument("--lamem-root", required=True, help="Path to the LaMem dataset root.")
    parser.add_argument("--splits-dir", required=True, help="Directory containing train_<fold>.txt etc.")
    parser.add_argument("--fold", type=int, default=1, help="LaMem split fold to export.")
    parser.add_argument("--train-file", help="Optional explicit train split file.")
    parser.add_argument("--val-file", help="Optional explicit validation split file.")
    parser.add_argument("--test-file", help="Optional explicit test split file.")
    parser.add_argument("--output-manifest", required=True, help="Destination manifest CSV.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    splits = load_lamem_splits(
        root=args.lamem_root,
        splits_dir=args.splits_dir,
        fold=args.fold,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
    )
    lamem_root = Path(args.lamem_root).expanduser().resolve()
    output_path = Path(args.output_manifest).expanduser().resolve()
    rows = []
    for split_name, records in (("train", splits.train), ("val", splits.val), ("test", splits.test)):
        for record in records:
            rows.append(
                {
                    "path": str(record.image_path.relative_to(lamem_root)),
                    "score": f"{record.score:.8f}",
                    "id": record.identifier,
                    "split": split_name,
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "score", "id", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote LaMem locked manifest to {output_path}")


if __name__ == "__main__":
    main()
