from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union


@dataclass(frozen=True)
class Record:
    image_path: Path
    score: float
    identifier: str


@dataclass(frozen=True)
class DatasetSplits:
    train: list[Record]
    val: list[Record]
    test: list[Record]


@dataclass(frozen=True)
class RecordDataset:
    records: list[Record]
    labels: list[str]
    label_column: Optional[str]


PathLike = Union[Path, str]


def _resolve(root: PathLike) -> Path:
    return Path(root).expanduser().resolve()


def _normalize_records(records: Iterable[Record]) -> list[Record]:
    return list(records)


def load_lamem_split_file(root: PathLike, split_file: PathLike) -> list[Record]:
    dataset_root = _resolve(root)
    split_path = _resolve(split_file)
    records: list[Record] = []
    with split_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(f"Expected '<relative_path> <score>' in {split_path}, got: {line!r}")
            rel_path = parts[0]
            score = float(parts[1])
            records.append(
                Record(
                    image_path=(dataset_root / "images" / rel_path).resolve(),
                    score=score,
                    identifier=rel_path,
                )
            )
    return _normalize_records(records)


def resolve_lamem_split_files(
    splits_dir: PathLike,
    fold: int,
    train_file: Optional[PathLike] = None,
    val_file: Optional[PathLike] = None,
    test_file: Optional[PathLike] = None,
) -> Tuple[Path, Path, Path]:
    if train_file and val_file and test_file:
        return (_resolve(train_file), _resolve(val_file), _resolve(test_file))

    base = _resolve(splits_dir)
    candidates = [
        (
            base / f"train_{fold}.txt",
            base / f"val_{fold}.txt",
            base / f"test_{fold}.txt",
        ),
        (
            base / f"split_{fold}" / "train.txt",
            base / f"split_{fold}" / "val.txt",
            base / f"split_{fold}" / "test.txt",
        ),
        (
            base / f"split{fold}" / "train.txt",
            base / f"split{fold}" / "val.txt",
            base / f"split{fold}" / "test.txt",
        ),
    ]

    for train_path, val_path, test_path in candidates:
        if train_path.exists() and val_path.exists() and test_path.exists():
            return (train_path, val_path, test_path)

    raise FileNotFoundError(
        "Could not resolve LaMem split files. Pass --train-file/--val-file/--test-file explicitly or "
        f"use a splits directory containing train_{fold}.txt, val_{fold}.txt, and test_{fold}.txt."
    )


def load_lamem_splits(
    root: PathLike,
    splits_dir: PathLike,
    fold: int,
    train_file: Optional[PathLike] = None,
    val_file: Optional[PathLike] = None,
    test_file: Optional[PathLike] = None,
) -> DatasetSplits:
    train_path, val_path, test_path = resolve_lamem_split_files(
        splits_dir=splits_dir,
        fold=fold,
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
    )
    return DatasetSplits(
        train=load_lamem_split_file(root, train_path),
        val=load_lamem_split_file(root, val_path),
        test=load_lamem_split_file(root, test_path),
    )


def load_memcat_records(
    root: PathLike,
    csv_path: Optional[PathLike] = None,
    score_column: str = "memorability_w_fa_correction",
) -> list[Record]:
    dataset_root = _resolve(root)
    if csv_path:
        source_csv = _resolve(csv_path)
    else:
        # Try both common layouts: <root>/data/memcat_image_data.csv and <root>/memcat_image_data.csv
        candidate_data = dataset_root / "data" / "memcat_image_data.csv"
        candidate_root = dataset_root / "memcat_image_data.csv"
        if candidate_data.exists():
            source_csv = candidate_data
        elif candidate_root.exists():
            source_csv = candidate_root
        else:
            raise FileNotFoundError(
                f"Could not find memcat_image_data.csv in {dataset_root / 'data'} or {dataset_root}. "
                "Pass csv_path explicitly."
            )

    # Resolve image directory: try <root>/images then <root>/MemCat then <root>
    image_base: Path
    if (dataset_root / "images").is_dir():
        image_base = Path("images")
    elif (dataset_root / "MemCat").is_dir():
        image_base = Path("MemCat")
    else:
        image_base = Path(".")

    records: list[Record] = []
    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row in {source_csv}")

        fieldnames = set(reader.fieldnames)
        score_key = score_column if score_column in fieldnames else None
        for fallback in ("memorability", "memo_score", "score"):
            if score_key is None and fallback in fieldnames:
                score_key = fallback

        if score_key is None:
            raise KeyError(f"Could not find a memorability score column in {source_csv}")

        for row in reader:
            image_name = row.get("image") or row.get("image_file") or row.get("image_name") or row.get("img")
            category = row.get("category") or row.get("supercategory") or row.get("cat")
            subcategory = row.get("subcategory") or row.get("subcat") or row.get("scat")
            if not image_name or not category or not subcategory:
                raise ValueError(
                    "MemCat loader expects image/category/subcategory columns. "
                    "Use a custom manifest if your export differs from the published CSV."
                )
            rel_path = image_base / category / subcategory / image_name
            records.append(
                Record(
                    image_path=(dataset_root / rel_path).resolve(),
                    score=float(row[score_key]),
                    identifier=str(rel_path),
                )
            )
    return _normalize_records(records)


def load_manifest(csv_path: PathLike, root: Optional[PathLike] = None) -> DatasetSplits:
    manifest_path = _resolve(csv_path)
    base_root = _resolve(root) if root else manifest_path.parent
    buckets: dict[str, list[Record]] = {"train": [], "val": [], "test": []}
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"path", "score", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manifest must contain columns {sorted(required)}")
        for row in reader:
            split = row["split"].strip().lower()
            if split not in buckets:
                raise ValueError(f"Unknown split {split!r} in {manifest_path}")
            rel_path = row["path"].strip()
            buckets[split].append(
                Record(
                    image_path=(base_root / rel_path).resolve(),
                    score=float(row["score"]),
                    identifier=row.get("id", rel_path),
                )
            )
    return DatasetSplits(
        train=_normalize_records(buckets["train"]),
        val=_normalize_records(buckets["val"]),
        test=_normalize_records(buckets["test"]),
    )


def load_record_dataset(
    csv_path: PathLike,
    *,
    root: Optional[PathLike] = None,
    path_column: str = "path",
    score_column: str = "score",
    id_column: Optional[str] = "id",
    label_column: Optional[str] = None,
) -> RecordDataset:
    manifest_path = _resolve(csv_path)
    base_root = _resolve(root) if root else manifest_path.parent

    records: list[Record] = []
    labels: list[str] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row in {manifest_path}")
        required = {path_column, score_column}
        if not required.issubset(reader.fieldnames):
            raise ValueError(f"Dataset manifest must contain columns {sorted(required)}")
        if label_column and label_column not in reader.fieldnames:
            raise ValueError(f"Dataset manifest is missing requested label column {label_column!r}")
        if id_column and id_column not in reader.fieldnames:
            id_column = None

        for row in reader:
            rel_path = row[path_column].strip()
            records.append(
                Record(
                    image_path=(base_root / rel_path).resolve(),
                    score=float(row[score_column]),
                    identifier=row[id_column] if id_column else rel_path,
                )
            )
            labels.append(row[label_column] if label_column else "__all__")

    return RecordDataset(records=_normalize_records(records), labels=labels, label_column=label_column)
