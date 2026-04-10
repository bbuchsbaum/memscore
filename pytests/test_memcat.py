import csv
import tempfile
from pathlib import Path

import numpy as np

from membench.memcat import MemCatDataset, load_memcat_dataset


def _write_fake_memcat(root: Path, num_per_category: int = 6) -> Path:
    """Create a minimal MemCat-style directory and CSV for testing."""
    categories = {
        "animal": ["cat", "dog"],
        "food": ["fruit", "vegetable"],
        "vehicle": ["car", "truck"],
    }
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "memcat_image_data.csv"

    rng = np.random.default_rng(42)
    rows = []
    for cat, subcats in categories.items():
        for subcat in subcats:
            img_dir = root / "images" / cat / subcat
            img_dir.mkdir(parents=True, exist_ok=True)
            for i in range(num_per_category):
                img_name = f"{cat}_{subcat}_{i}.jpg"
                (img_dir / img_name).write_bytes(b"\xff\xd8\xff\xe0")  # minimal JPEG header
                rows.append({
                    "image": img_name,
                    "category": cat,
                    "subcategory": subcat,
                    "memorability_w_fa_correction": float(rng.uniform(0.3, 0.9)),
                })

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "category", "subcategory", "memorability_w_fa_correction"])
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def test_load_memcat_dataset_returns_correct_structure() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_fake_memcat(root, num_per_category=4)
        dataset = load_memcat_dataset(root)

        assert isinstance(dataset, MemCatDataset)
        assert dataset.name == "memcat"
        assert dataset.score_column == "memorability_w_fa_correction"
        # 3 categories * 2 subcategories * 4 images = 24
        assert len(dataset.records) == 24
        assert len(dataset.categories) == 24
        assert set(dataset.categories) == {"animal", "food", "vehicle"}


def test_load_memcat_dataset_scores_in_valid_range() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_fake_memcat(root, num_per_category=3)
        dataset = load_memcat_dataset(root)

        for record in dataset.records:
            assert 0.0 <= record.score <= 1.0


def test_load_memcat_dataset_resolves_image_paths() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        _write_fake_memcat(root, num_per_category=2)
        dataset = load_memcat_dataset(root)

        for record in dataset.records:
            assert record.image_path.is_absolute()
            assert record.image_path.exists()


def test_load_memcat_dataset_with_explicit_csv_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        csv_path = _write_fake_memcat(root, num_per_category=2)
        # Move CSV to a non-default location
        alt_csv = root / "alternate_data.csv"
        alt_csv.write_text(csv_path.read_text())
        dataset = load_memcat_dataset(root, csv_path=alt_csv)

        assert len(dataset.records) == 12  # 3 * 2 * 2


def test_load_memcat_dataset_fallback_score_column() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        data_dir = root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        img_dir = root / "images" / "animal" / "cat"
        img_dir.mkdir(parents=True, exist_ok=True)
        (img_dir / "img.jpg").write_bytes(b"\xff\xd8\xff\xe0")

        csv_path = data_dir / "memcat_image_data.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "category", "subcategory", "memorability"])
            writer.writeheader()
            writer.writerow({"image": "img.jpg", "category": "animal", "subcategory": "cat", "memorability": "0.75"})

        dataset = load_memcat_dataset(root)
        assert dataset.score_column == "memorability"
        assert len(dataset.records) == 1
        assert np.isclose(dataset.records[0].score, 0.75)
