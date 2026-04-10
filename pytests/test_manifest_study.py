from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from membench.clip_regression import EmbeddingBundle
from memscore import api
from memscore.cli import main


def _write_manifest(root: Path, manifest_path: Path) -> None:
    image_root = root / "images"
    image_root.mkdir(parents=True, exist_ok=True)
    rows = []
    categories = ["animal", "food", "vehicle"]
    for category_idx, category in enumerate(categories):
        for item_idx in range(8):
            filename = f"{category}_{item_idx}.jpg"
            Image.new("RGB", (8, 8), color=(category_idx * 60, item_idx * 20, 120)).save(image_root / filename)
            rows.append(
                {
                    "path": f"images/{filename}",
                    "score": 0.2 + (0.05 * category_idx) + (0.01 * item_idx),
                    "id": f"{category}-{item_idx}",
                    "category": category,
                }
            )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "score", "id", "category"])
        writer.writeheader()
        writer.writerows(rows)


class DummyScorer:
    def __init__(self, checkpoint_path, batch_size, device) -> None:
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.device = device

    def predict(self, records):
        return np.asarray([record.score + 0.02 for record in records], dtype=np.float32)


class DummyEmbedder:
    def __init__(self, model_name, pretrained, batch_size, device, tta_mode, tta_resize) -> None:
        self.model_name = model_name

    def embed_split(self, records, split_name, cache_dir=None):
        offset = 0.0 if self.model_name == "ViT-B-32" else 0.03
        features = []
        for index, record in enumerate(records):
            features.append([record.score + offset, record.score**2, float(index)])
        return EmbeddingBundle(
            features=np.asarray(features, dtype=np.float32),
            scores=np.asarray([record.score for record in records], dtype=np.float32),
            identifiers=[record.identifier for record in records],
            paths=[str(record.image_path) for record in records],
        )


def test_study_manifest_produces_summary(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "dataset.csv"
    _write_manifest(tmp_path, manifest_path)

    monkeypatch.setattr("membench.manifest_study.ensure_resmem_checkpoint", lambda cache_dir: tmp_path / "model.pt")
    monkeypatch.setattr("membench.manifest_study.ResMemScorer", DummyScorer)
    monkeypatch.setattr("membench.manifest_study.ClipEmbedder", DummyEmbedder)

    study = api.study_manifest(
        manifest_path,
        root=tmp_path,
        dataset_name="toyset",
        label_column="category",
        clip_models=("ViT-B-32", "RN50"),
        pca_dims=("none", "2"),
        num_splits=2,
        cache_dir=tmp_path / "cache",
        output_json=tmp_path / "study.json",
        output_csv=tmp_path / "study.csv",
    )

    dataset = study["studies"][0]
    methods = {row["method"] for row in study["summary_rows"]}
    features = {row["feature_source"] for row in study["summary_rows"]}

    assert study["metadata"]["dataset_name"] == "toyset"
    assert dataset["label_column"] == "category"
    assert dataset["num_records"] == 24
    assert "ridge" in methods
    assert "ensemble_mean" in methods
    assert "ViT-B-32" in features
    assert "ViT-B-32+RN50" in features
    assert "ensemble:ViT-B-32+RN50" in features
    assert (tmp_path / "study.json").exists()
    assert (tmp_path / "study.csv").exists()


def test_cli_study_manifest_routes_to_api(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(
        "memscore.cli.api.study_manifest",
        lambda *args, **kwargs: {"summary_rows": [{"feature_source": "ViT-B-32", "method": "ridge"}]},
    )

    exit_code = main(
        [
            "study-manifest",
            "--manifest",
            str(tmp_path / "dataset.csv"),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Wrote study JSON to" in output
    assert "Wrote study CSV to" in output
