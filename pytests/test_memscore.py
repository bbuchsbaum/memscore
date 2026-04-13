from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from memscore import api
from memscore.api import Prediction
from memscore.cli import main


def _write_image(path: Path) -> None:
    Image.new("RGB", (8, 8), color=(120, 30, 200)).save(path)


def test_predict_paths_uses_explicit_files_in_order(monkeypatch, tmp_path: Path) -> None:
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.png"
    _write_image(first)
    _write_image(second)

    class DummyScorer:
        def __init__(self, checkpoint_path, batch_size, device) -> None:
            self.checkpoint_path = checkpoint_path
            self.batch_size = batch_size
            self.device = device

        def predict(self, records):
            assert [record.image_path for record in records] == [second.resolve(), first.resolve()]
            return np.asarray([0.9, 0.1], dtype=np.float32)

    monkeypatch.setattr(api, "ensure_resmem_checkpoint", lambda cache_dir: tmp_path / "model.pt")
    monkeypatch.setattr(api, "ResMemScorer", DummyScorer)

    predictions = api.predict_paths([second, first], cache_dir=tmp_path / "cache")

    assert [item.identifier for item in predictions] == ["second.png", "first.jpg"]
    assert [round(item.score, 3) for item in predictions] == [0.9, 0.1]


def test_collect_image_paths_recurses_only_when_requested(tmp_path: Path) -> None:
    top = tmp_path / "images"
    nested = top / "nested"
    nested.mkdir(parents=True)
    _write_image(top / "a.jpg")
    _write_image(nested / "b.jpg")

    flat = api.collect_image_paths([top], recursive=False)
    recursive = api.collect_image_paths([top], recursive=True)

    assert [path.name for path in flat] == ["a.jpg"]
    assert [path.name for path in recursive] == ["a.jpg", "b.jpg"]


def test_cli_predict_writes_csv(monkeypatch, tmp_path: Path, capsys) -> None:
    image_path = tmp_path / "sample.jpg"
    _write_image(image_path)

    monkeypatch.setattr(
        "memscore.cli.api.predict_paths",
        lambda *args, **kwargs: [Prediction(identifier="sample.jpg", image_path=image_path, score=0.42)],
    )

    output_path = tmp_path / "predictions.csv"
    exit_code = main(["predict", str(image_path), "--output", str(output_path)])

    assert exit_code == 0
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["id", "image_path", "memorability"]
    assert rows[1][0] == "sample.jpg"
    assert rows[1][2] == "0.42000000"
    assert "Wrote predictions to" in capsys.readouterr().out


def test_cli_predict_resmem_alias(monkeypatch, tmp_path: Path, capsys) -> None:
    image_path = tmp_path / "sample.jpg"
    _write_image(image_path)

    monkeypatch.setattr(
        "memscore.cli.api.predict_paths",
        lambda *args, **kwargs: [Prediction(identifier="sample.jpg", image_path=image_path, score=0.31)],
    )

    exit_code = main(["predict-resmem", str(image_path)])

    assert exit_code == 0
    assert '"memorability": 0.31' in capsys.readouterr().out


def test_cli_benchmark_standard_uses_packaged_defaults(monkeypatch, tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,score,split\nimage.jpg,0.5,test\n", encoding="utf-8")
    output_path = tmp_path / "standard.json"
    captured_kwargs = {}

    def fake_benchmark_standard_manifest(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return {
            "metadata": {"standard_ensemble": "ensemble_mean"},
            "standard_model": {"summary": {"mean_spearman": 0.75}},
        }

    monkeypatch.setattr("memscore.cli.api.benchmark_standard_manifest", fake_benchmark_standard_manifest)

    exit_code = main(["benchmark-standard", "--manifest", str(manifest), "--output-json", str(output_path)])

    assert exit_code == 0
    assert captured_kwargs["clip_models"] == ["ViT-B-32", "RN50", "ViT-B-16"]
    assert captured_kwargs["tta_mode"] == "fivecrop"
    assert captured_kwargs["tta_resize"] == 256
    assert output_path.exists()
    assert "Wrote standard benchmark summary" in capsys.readouterr().out


def test_cli_predict_standard_writes_csv(monkeypatch, tmp_path: Path, capsys) -> None:
    image_path = tmp_path / "standard.png"
    _write_image(image_path)
    model_path = tmp_path / "memscore-standard.pkl"
    model_path.write_bytes(b"fixture")

    monkeypatch.setattr(
        "memscore.cli.api.predict_standard_paths",
        lambda *args, **kwargs: [Prediction(identifier="standard.png", image_path=image_path, score=0.77)],
    )

    output_path = tmp_path / "standard_predictions.csv"
    exit_code = main(
        [
            "predict-standard",
            "--model",
            str(model_path),
            str(image_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    with output_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0] == ["id", "image_path", "memorability"]
    assert rows[1][0] == "standard.png"
    assert rows[1][2] == "0.77000000"
    assert "Wrote predictions to" in capsys.readouterr().out


def test_cli_train_standard_scorer_uses_defaults(monkeypatch, tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,score,split\nimage.jpg,0.5,train\n", encoding="utf-8")
    output_model = tmp_path / "memscore-standard.pkl"
    captured_kwargs = {}

    def fake_train_standard_scorer(**kwargs):
        captured_kwargs.update(kwargs)
        output_model.write_bytes(b"fixture")
        return {"artifact_path": str(output_model), "clip_models": kwargs["clip_models"]}

    monkeypatch.setattr("memscore.cli.api.train_standard_scorer", fake_train_standard_scorer)

    exit_code = main(
        [
            "train-standard-scorer",
            "--manifest",
            str(manifest),
            "--output-model",
            str(output_model),
        ]
    )

    assert exit_code == 0
    assert captured_kwargs["clip_models"] == ["ViT-B-32", "RN50", "ViT-B-16"]
    assert captured_kwargs["tta_mode"] == "fivecrop"
    assert captured_kwargs["tta_resize"] == 256
    assert captured_kwargs["include_test"] is False
    assert "Wrote standard scorer artifact" in capsys.readouterr().out
