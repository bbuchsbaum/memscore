from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from membench.clip_regression import EmbeddingBundle
from memscore import api
from memscore.cli import main


def _write_fake_lamem(root: Path, splits_dir: Path) -> None:
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    def _make_split(name: str, start: int, stop: int) -> None:
        lines = []
        for idx in range(start, stop):
            rel_path = f"{idx:05d}.jpg"
            Image.new("RGB", (8, 8), color=(idx % 255, 60, 120)).save(image_dir / rel_path)
            score = 0.15 + (idx * 0.03)
            lines.append(f"{rel_path} {score:.6f}\n")
        (splits_dir / name).write_text("".join(lines), encoding="utf-8")

    _make_split("train_1.txt", 0, 12)
    _make_split("val_1.txt", 12, 20)
    _make_split("test_1.txt", 20, 30)


class DummyScorer:
    def __init__(self, checkpoint_path, batch_size, device) -> None:
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size
        self.device = device

    def predict(self, records):
        predictions = []
        for index, record in enumerate(records):
            predictions.append(record.score + (0.01 if index % 2 == 0 else -0.01))
        return np.asarray(predictions, dtype=np.float32)


class DummyEmbedder:
    def __init__(self, model_name, pretrained, batch_size, device, tta_mode, tta_resize) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.device = device
        self.tta_mode = tta_mode
        self.tta_resize = tta_resize

    def embed_split(self, records, split_name, cache_dir=None):
        offset = 0.0 if self.model_name == "ViT-B-32" else 0.05
        features = []
        for index, record in enumerate(records):
            features.append(
                [
                    record.score + offset,
                    (record.score**2) + (0.001 * index),
                    float(index) / max(len(records), 1),
                ]
            )
        return EmbeddingBundle(
            features=np.asarray(features, dtype=np.float32),
            scores=np.asarray([record.score for record in records], dtype=np.float32),
            identifiers=[record.identifier for record in records],
            paths=[str(record.image_path) for record in records],
        )


def test_study_lamem_produces_subsampled_summary(monkeypatch, tmp_path: Path) -> None:
    lamem_root = tmp_path / "lamem"
    splits_dir = tmp_path / "splits"
    _write_fake_lamem(lamem_root, splits_dir)

    monkeypatch.setattr("membench.lamem.ensure_resmem_checkpoint", lambda cache_dir: tmp_path / "model.pt")
    monkeypatch.setattr("membench.lamem.ResMemScorer", DummyScorer)
    monkeypatch.setattr("membench.lamem.ClipEmbedder", DummyEmbedder)

    study = api.study_lamem(
        lamem_root,
        splits_dir=splits_dir,
        fold=1,
        clip_models=("ViT-B-32", "RN50"),
        pca_dims=("none", "2"),
        num_splits=2,
        limit_train=6,
        limit_val=4,
        limit_test=5,
        cache_dir=tmp_path / "cache",
        output_json=tmp_path / "study.json",
        output_csv=tmp_path / "study.csv",
    )

    metadata = study["metadata"]
    dataset = study["studies"][0]
    methods = {row["method"] for row in study["summary_rows"]}
    feature_sources = {row["feature_source"] for row in study["summary_rows"]}

    assert metadata["fold"] == 1
    assert metadata["limit_train"] == 6
    assert metadata["limit_val"] == 4
    assert metadata["limit_test"] == 5
    assert dataset["sampled_split_sizes"] == {"train": 6, "val": 4, "test": 5}
    assert dataset["union_split_sizes"]["train"] >= 6
    assert dataset["union_split_sizes"]["val"] >= 4
    assert dataset["union_split_sizes"]["test"] >= 5
    assert "ViT-B-32" in feature_sources
    assert "ViT-B-32+RN50" in feature_sources
    assert "ensemble:ViT-B-32+RN50" in feature_sources
    assert "resmem+ViT-B-32+RN50" in feature_sources
    assert "ridge" in methods
    assert "ensemble_hybrid_val_weighted" in methods
    assert (tmp_path / "study.json").exists()
    assert (tmp_path / "study.csv").exists()


def test_cli_study_lamem_routes_to_api(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(
        "memscore.cli.api.study_lamem",
        lambda *args, **kwargs: {"summary_rows": [{"feature_source": "ViT-B-32", "method": "ridge"}]},
    )

    exit_code = main(
        [
            "study-lamem",
            "--lamem-root",
            str(tmp_path / "lamem"),
            "--splits-dir",
            str(tmp_path / "splits"),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Wrote study JSON to" in output
    assert "Wrote study CSV to" in output
