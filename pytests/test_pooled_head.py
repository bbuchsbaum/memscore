from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from membench.clip_regression import EmbeddingBundle
from membench.lora_head import LoRALinear
from memscore import api
from memscore.cli import main


def _write_split_manifest(root: Path, stem: str, offset: float) -> Path:
    manifest_path = root / f"{stem}.csv"
    rows = []
    for index in range(10):
        if index < 6:
            split = "train"
        elif index < 8:
            split = "val"
        else:
            split = "test"
        rows.append(
            {
                "path": f"images/{stem}_{index}.jpg",
                "score": 0.2 + offset + (0.03 * index),
                "split": split,
                "id": f"{stem}-{index}",
            }
        )

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "score", "split", "id"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def _write_dataset_config(root: Path, manifests: dict[str, Path]) -> Path:
    config_path = root / "datasets.csv"
    with config_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "manifest"])
        writer.writeheader()
        for dataset, manifest_path in manifests.items():
            writer.writerow({"dataset": dataset, "manifest": str(manifest_path)})
    return config_path


class DummyEmbedder:
    def __init__(self, model_name, pretrained, batch_size, device, tta_mode, tta_resize) -> None:
        self.model_name = model_name

    def embed_split(self, records, split_name, cache_dir=None):
        features = []
        for index, record in enumerate(records):
            features.append(
                [
                    record.score,
                    record.score**2,
                    float(index) / max(1, len(records)),
                ]
            )
        return EmbeddingBundle(
            features=np.asarray(features, dtype=np.float32),
            scores=np.asarray([record.score for record in records], dtype=np.float32),
            identifiers=[record.identifier for record in records],
            paths=[str(record.image_path) for record in records],
        )


def test_train_pooled_head_returns_locked_test_report(monkeypatch, tmp_path: Path) -> None:
    manifests = {
        "isola": _write_split_manifest(tmp_path, "isola", 0.00),
        "figrim": _write_split_manifest(tmp_path, "figrim", 0.15),
    }
    config_path = _write_dataset_config(tmp_path, manifests)

    monkeypatch.setattr("membench.supervised_head.ClipEmbedder", DummyEmbedder)

    result = api.train_pooled_head(
        config_path,
        cache_dir=tmp_path / "cache",
        output_json=tmp_path / "pooled.json",
        epochs=4,
        patience=2,
        hidden_dim=16,
        head_batch_size=8,
        device="cpu",
        seed=0,
    )

    assert set(result["datasets"]) == {"isola", "figrim"}
    assert "test_summary" in result
    assert result["metadata"]["best_epoch"] >= 1
    assert (tmp_path / "pooled.json").exists()
    assert result["datasets"]["isola"]["sizes"] == {"train": 6, "val": 2, "test": 2}


def test_cli_train_pooled_head_routes_to_api(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(
        "memscore.cli.api.train_pooled_head",
        lambda *args, **kwargs: {"test_summary": {"mean_spearman": 0.5, "mean_mse": 0.1}},
    )

    exit_code = main(
        [
            "train-pooled-head",
            "--dataset-config",
            str(tmp_path / "datasets.csv"),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Wrote pooled-head report to" in output


def test_train_lodo_head_returns_target_reports(monkeypatch, tmp_path: Path) -> None:
    manifests = {
        "isola": _write_split_manifest(tmp_path, "isola", 0.00),
        "figrim": _write_split_manifest(tmp_path, "figrim", 0.15),
        "lamem": _write_split_manifest(tmp_path, "lamem", 0.30),
    }
    config_path = _write_dataset_config(tmp_path, manifests)

    monkeypatch.setattr("membench.supervised_head.ClipEmbedder", DummyEmbedder)

    result = api.train_lodo_head(
        config_path,
        cache_dir=tmp_path / "cache",
        output_json=tmp_path / "lodo.json",
        epochs=4,
        patience=2,
        hidden_dim=16,
        head_batch_size=8,
        device="cpu",
        seed=0,
    )

    assert set(result["targets"]) == {"isola", "figrim", "lamem"}
    assert "summary" in result
    assert (tmp_path / "lodo.json").exists()
    assert set(result["targets"]["isola"]["source_datasets"]) == {"figrim", "lamem"}


def test_cli_train_lodo_head_routes_to_api(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(
        "memscore.cli.api.train_lodo_head",
        lambda *args, **kwargs: {"summary": {"mean_target_test_spearman": 0.4, "mean_target_test_mse": 0.2}},
    )

    exit_code = main(
        [
            "train-lodo-head",
            "--dataset-config",
            str(tmp_path / "datasets.csv"),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Wrote leave-one-out report to" in output


def test_lora_linear_matches_base_before_training() -> None:
    base = torch.nn.Linear(3, 2)
    layer = LoRALinear(base, rank=2, alpha=4.0, dropout=0.0)
    inputs = torch.randn(5, 3)
    expected = base(inputs)
    actual = layer(inputs)
    assert torch.allclose(actual, expected)


def test_cli_train_lora_head_routes_to_api(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(
        "memscore.cli.api.train_lora_head",
        lambda *args, **kwargs: {"test_summary": {"mean_spearman": 0.6, "mean_mse": 0.1}},
    )

    exit_code = main(
        [
            "train-lora-head",
            "--dataset-config",
            str(tmp_path / "datasets.csv"),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Wrote LoRA report to" in output


def test_train_multihead_head_returns_locked_test_report(monkeypatch, tmp_path: Path) -> None:
    manifests = {
        "isola": _write_split_manifest(tmp_path, "isola", 0.00),
        "figrim": _write_split_manifest(tmp_path, "figrim", 0.15),
    }
    config_path = _write_dataset_config(tmp_path, manifests)

    monkeypatch.setattr("membench.supervised_head.ClipEmbedder", DummyEmbedder)

    result = api.train_multihead_head(
        config_path,
        cache_dir=tmp_path / "cache",
        output_json=tmp_path / "multihead.json",
        epochs=4,
        patience=2,
        hidden_dim=16,
        head_batch_size=8,
        device="cpu",
        seed=0,
    )

    assert set(result["datasets"]) == {"isola", "figrim"}
    assert "test_summary" in result
    assert result["metadata"]["num_dataset_heads"] == 2
    assert (tmp_path / "multihead.json").exists()


def test_cli_train_multihead_head_routes_to_api(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(
        "memscore.cli.api.train_multihead_head",
        lambda *args, **kwargs: {"test_summary": {"mean_spearman": 0.55, "mean_mse": 0.1}},
    )

    exit_code = main(
        [
            "train-multihead-head",
            "--dataset-config",
            str(tmp_path / "datasets.csv"),
        ]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Wrote multi-head report to" in output
