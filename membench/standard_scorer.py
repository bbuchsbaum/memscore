from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from .clip_regression import ClipEmbedder, normalize_tta_mode
from .data import Record, load_manifest
from .figrim import build_ridge_pipeline
from .supervised_head import load_fixed_split_dataset_specs

PathLike = Union[Path, str]
ARTIFACT_VERSION = 1
TrainingGroup = tuple[str, list[Record]]


def _artifact_payload(
    *,
    models: dict[str, object],
    clip_pretrained: str,
    clip_tta_mode: str,
    clip_tta_resize: Optional[int],
    training_summary: dict[str, object],
) -> dict[str, object]:
    return {
        "artifact_version": ARTIFACT_VERSION,
        "model_type": "memscore-standard-ridge-mean",
        "clip_models": list(models),
        "clip_pretrained": clip_pretrained,
        "clip_tta_mode": clip_tta_mode,
        "clip_tta_resize": clip_tta_resize,
        "regressor": "ridge",
        "ensemble": "mean",
        "models": models,
        "training_summary": training_summary,
    }


def _load_artifact(path: PathLike) -> dict[str, object]:
    artifact_path = Path(path).expanduser().resolve()
    with artifact_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid memscore scorer artifact: {artifact_path}")
    if payload.get("artifact_version") != ARTIFACT_VERSION:
        raise ValueError(
            f"Unsupported scorer artifact version {payload.get('artifact_version')!r}; "
            f"expected {ARTIFACT_VERSION}"
        )
    if payload.get("model_type") != "memscore-standard-ridge-mean":
        raise ValueError(f"Unsupported scorer artifact type {payload.get('model_type')!r}")
    return payload


def _records_from_manifest(
    manifest_path: PathLike,
    root: Optional[PathLike],
    *,
    include_test: bool,
) -> tuple[list[TrainingGroup], dict[str, int]]:
    splits = load_manifest(manifest_path, root=root)
    groups: list[TrainingGroup] = [
        ("train", splits.train),
        ("val", splits.val),
    ]
    if include_test:
        groups.append(("test", splits.test))
    records = [record for _, group_records in groups for record in group_records]
    return groups, {
        "train": len(splits.train),
        "val": len(splits.val),
        "test": len(splits.test),
        "used": len(records),
    }


def _load_training_records(
    *,
    manifest_path: Optional[PathLike],
    root: Optional[PathLike],
    dataset_config: Optional[PathLike],
    include_test: bool,
) -> tuple[list[TrainingGroup], dict[str, object]]:
    if manifest_path is None and dataset_config is None:
        raise ValueError("Pass either manifest_path or dataset_config")
    if manifest_path is not None and dataset_config is not None:
        raise ValueError("Pass only one of manifest_path or dataset_config")

    if manifest_path is not None:
        groups, sizes = _records_from_manifest(manifest_path, root, include_test=include_test)
        return groups, {
            "source": "manifest",
            "manifest": str(Path(manifest_path).expanduser().resolve()),
            "root": str(Path(root).expanduser().resolve()) if root is not None else None,
            "sizes": sizes,
            "include_test": include_test,
        }

    assert dataset_config is not None
    groups: list[TrainingGroup] = []
    datasets: list[dict[str, object]] = []
    for spec in load_fixed_split_dataset_specs(dataset_config):
        spec_groups, sizes = _records_from_manifest(spec.manifest_path, spec.root, include_test=include_test)
        for split_name, split_records in spec_groups:
            if split_records:
                groups.append((f"{spec.name}_{split_name}", split_records))
        datasets.append(
            {
                "dataset": spec.name,
                "manifest": str(spec.manifest_path),
                "root": str(spec.root) if spec.root is not None else None,
                "sizes": sizes,
            }
        )
    records = [record for _, group_records in groups for record in group_records]
    return groups, {
        "source": "dataset_config",
        "dataset_config": str(Path(dataset_config).expanduser().resolve()),
        "datasets": datasets,
        "total_used": len(records),
        "include_test": include_test,
    }


def train_standard_scorer(
    *,
    output_path: PathLike,
    manifest_path: Optional[PathLike] = None,
    root: Optional[PathLike] = None,
    dataset_config: Optional[PathLike] = None,
    clip_models: Sequence[str],
    clip_pretrained: str,
    clip_tta_mode: str,
    clip_tta_resize: Optional[int],
    batch_size: int,
    device: str,
    cache_dir: PathLike,
    include_test: bool = False,
) -> dict[str, object]:
    """Train and save the standard memscore ridge-head ensemble."""

    groups, training_summary = _load_training_records(
        manifest_path=manifest_path,
        root=root,
        dataset_config=dataset_config,
        include_test=include_test,
    )
    records = [record for _, group_records in groups for record in group_records]
    if len(records) < 2:
        raise ValueError("At least two labeled training records are required")

    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    normalized_tta = normalize_tta_mode(clip_tta_mode)
    scores = np.asarray([record.score for record in records], dtype=np.float32)

    models: dict[str, object] = {}
    for model_name in clip_models:
        embedder = ClipEmbedder(
            model_name=model_name,
            pretrained=clip_pretrained,
            batch_size=batch_size,
            device=device,
            tta_mode=normalized_tta,
            tta_resize=clip_tta_resize,
        )
        feature_blocks = []
        for split_name, group_records in groups:
            if not group_records:
                continue
            bundle = embedder.embed_split(
                group_records,
                split_name=split_name,
                cache_dir=cache_root / "clip",
            )
            feature_blocks.append(bundle.features)
        features = np.concatenate(feature_blocks, axis=0)
        model = build_ridge_pipeline(pca_dim=None)
        model.fit(features, scores)
        models[model_name] = model

    payload = _artifact_payload(
        models=models,
        clip_pretrained=clip_pretrained,
        clip_tta_mode=normalized_tta,
        clip_tta_resize=clip_tta_resize,
        training_summary=training_summary,
    )
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "artifact_path": str(destination),
        "artifact_version": ARTIFACT_VERSION,
        "model_type": payload["model_type"],
        "clip_models": payload["clip_models"],
        "clip_pretrained": payload["clip_pretrained"],
        "clip_tta_mode": payload["clip_tta_mode"],
        "clip_tta_resize": payload["clip_tta_resize"],
        "regressor": payload["regressor"],
        "ensemble": payload["ensemble"],
        "training_summary": training_summary,
    }


def predict_with_standard_scorer(
    *,
    model_path: PathLike,
    records: list[Record],
    batch_size: int,
    device: str,
    cache_dir: Optional[PathLike] = None,
    clip_scores: bool = True,
) -> np.ndarray:
    """Predict memorability for records with a saved standard scorer artifact."""

    if not records:
        return np.asarray([], dtype=np.float32)

    payload = _load_artifact(model_path)
    cache_root = Path(cache_dir).expanduser().resolve() if cache_dir is not None else None
    member_predictions: list[np.ndarray] = []
    for model_name in payload["clip_models"]:
        embedder = ClipEmbedder(
            model_name=model_name,
            pretrained=str(payload["clip_pretrained"]),
            batch_size=batch_size,
            device=device,
            tta_mode=str(payload["clip_tta_mode"]),
            tta_resize=payload["clip_tta_resize"],
        )
        bundle = embedder.embed_split(
            records,
            split_name="standard_scorer_predict",
            cache_dir=cache_root / "clip" if cache_root is not None else None,
        )
        model = payload["models"][model_name]
        member_predictions.append(model.predict(bundle.features).astype(np.float32))

    predictions = np.mean(member_predictions, axis=0).astype(np.float32)
    if clip_scores:
        predictions = np.clip(predictions, 0.0, 1.0)
    return predictions
