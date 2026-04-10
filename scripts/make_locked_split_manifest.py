from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a locked train/val/test manifest from a scored-image CSV.")
    parser.add_argument("--input-manifest", required=True, help="CSV containing at least path and score columns.")
    parser.add_argument("--output-manifest", required=True, help="Destination CSV with an added split column.")
    parser.add_argument("--path-column", default="path", help="Column containing image paths.")
    parser.add_argument("--score-column", default="score", help="Column containing memorability scores.")
    parser.add_argument("--id-column", default="id", help="Optional identifier column.")
    parser.add_argument("--label-column", help="Optional stratification label column.")
    parser.add_argument("--train-size", type=float, default=0.7, help="Fraction assigned to train.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Fraction assigned to validation.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction assigned to test.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for the split.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input_manifest).expanduser().resolve()
    output_path = Path(args.output_manifest).expanduser().resolve()

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row in {input_path}")
        required = {args.path_column, args.score_column}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"Input manifest must contain columns {sorted(required)}")
        if args.label_column and args.label_column not in reader.fieldnames:
            raise ValueError(f"Input manifest is missing label column {args.label_column!r}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {input_path}")

    indices = np.arange(len(rows))
    labels = None
    if args.label_column:
        labels = np.asarray([row[args.label_column] for row in rows], dtype=object)

    val_fraction_within_trainval = args.val_size / (args.train_size + args.val_size)
    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels if labels is not None else None,
    )
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=val_fraction_within_trainval,
        random_state=args.seed,
        stratify=labels[trainval_idx] if labels is not None else None,
    )

    split_map = {int(index): "train" for index in train_idx}
    split_map.update({int(index): "val" for index in val_idx})
    split_map.update({int(index): "test" for index in test_idx})

    output_rows = []
    for index, row in enumerate(rows):
        updated = dict(row)
        updated["split"] = split_map[index]
        output_rows.append(updated)

    fieldnames = list(reader.fieldnames)
    if "split" not in fieldnames:
        fieldnames.append("split")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote locked split manifest to {output_path}")


if __name__ == "__main__":
    main()
