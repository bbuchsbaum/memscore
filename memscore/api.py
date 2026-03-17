from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from membench.clip_regression import run_clip_regression
from membench.data import DatasetSplits, Record, load_lamem_splits, load_manifest
from membench.figrim import parse_pca_dims, parse_positive_dims, run_figrim_study, write_study_outputs
from membench.metrics import compute_metrics
from membench.resmem_baseline import ResMemScorer, ensure_resmem_checkpoint

PathLike = Union[str, Path]
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Prediction:
    """One memorability score returned by :func:`predict_paths`."""

    identifier: str
    image_path: Path
    score: float


@dataclass(frozen=True)
class BenchmarkRun:
    """Artifacts returned by a single benchmark run."""

    summary: dict[str, object]
    test_records: list[Record]
    resmem_predictions: np.ndarray
    challenger_predictions: np.ndarray


def collect_image_paths(paths: Sequence[PathLike], recursive: bool = False) -> list[Path]:
    """Resolve image files and directories into a unique list of paths.

    Parameters
    ----------
    paths:
        One or more image files or directories.
    recursive:
        When ``True``, recurse into subdirectories for directory inputs.

    Returns
    -------
    list of pathlib.Path
        Unique image paths in a stable order.
    """

    collected: list[Path] = []
    seen: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input path does not exist: {path}")
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            for candidate in sorted(iterator):
                if not candidate.is_file() or candidate.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                if candidate not in seen:
                    collected.append(candidate)
                    seen.add(candidate)
            continue

        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image extension for {path.name!r}")
        if path not in seen:
            collected.append(path)
            seen.add(path)

    if not collected:
        raise ValueError("No images found. Pass one or more image files or directories containing images.")
    return collected


def predict_paths(
    paths: Sequence[PathLike],
    *,
    recursive: bool = False,
    checkpoint_path: Optional[PathLike] = None,
    batch_size: int = 32,
    device: str = "cpu",
    cache_dir: PathLike = ".cache/memscore",
) -> list[Prediction]:
    """Score one or more images with the released ``ResMem`` checkpoint.

    Parameters
    ----------
    paths:
        Image files or directories to score.
    recursive:
        Recurse into subdirectories for directory inputs.
    checkpoint_path:
        Optional explicit checkpoint path. When omitted, ``memscore`` downloads
        the released ``ResMem`` weights into ``cache_dir``.
    batch_size:
        Batch size for model inference.
    device:
        Torch device string such as ``"cpu"`` or ``"cuda"``.
    cache_dir:
        Location used for downloaded model weights.

    Returns
    -------
    list of Prediction
        One score per resolved image.
    """

    image_paths = collect_image_paths(paths, recursive=recursive)
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    resolved_checkpoint = (
        Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path
        else ensure_resmem_checkpoint(cache_root / "resmem")
    )
    scorer = ResMemScorer(checkpoint_path=resolved_checkpoint, batch_size=batch_size, device=device)
    records = [
        Record(
            image_path=image_path,
            score=0.0,
            identifier=image_path.name,
        )
        for image_path in image_paths
    ]
    scores = scorer.predict(records)
    return [
        Prediction(identifier=record.identifier, image_path=record.image_path, score=float(score))
        for record, score in zip(records, scores)
    ]


def write_prediction_csv(predictions: Sequence[Prediction], output_path: PathLike) -> Path:
    """Write :class:`Prediction` objects to a CSV file."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "image_path", "memorability"])
        for prediction in predictions:
            writer.writerow([prediction.identifier, str(prediction.image_path), f"{prediction.score:.8f}"])
    return destination


def _subset_records(records: Sequence[Record], limit: Optional[int], seed: int) -> list[Record]:
    if limit is None or limit <= 0 or len(records) <= limit:
        return list(records)
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(records), size=limit, replace=False))
    return [records[index] for index in indices.tolist()]


def _benchmark_dataset(
    splits: DatasetSplits,
    *,
    dataset_name: str,
    clip_models: Sequence[str],
    clip_pretrained: str,
    regressors: Sequence[str],
    batch_size: int,
    device: str,
    cache_dir: PathLike,
    checkpoint_path: Optional[PathLike] = None,
    random_state: int = 0,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
    limit_test: Optional[int] = None,
    tta_mode: str = "none",
    tta_resize: Optional[int] = None,
) -> BenchmarkRun:
    trimmed = DatasetSplits(
        train=_subset_records(splits.train, limit_train, random_state),
        val=_subset_records(splits.val, limit_val, random_state + 1),
        test=_subset_records(splits.test, limit_test, random_state + 2),
    )

    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    resolved_checkpoint = (
        Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path
        else ensure_resmem_checkpoint(cache_root / "resmem")
    )
    scorer = ResMemScorer(checkpoint_path=resolved_checkpoint, batch_size=batch_size, device=device)
    resmem_predictions = scorer.predict(trimmed.test)
    test_truth = np.asarray([record.score for record in trimmed.test], dtype=np.float32)
    resmem_metrics = compute_metrics(test_truth, resmem_predictions)

    clip_runs = []
    best_clip = None
    for clip_model in clip_models:
        clip_result = run_clip_regression(
            train_records=trimmed.train,
            val_records=trimmed.val,
            test_records=trimmed.test,
            clip_model=clip_model,
            clip_pretrained=clip_pretrained,
            batch_size=batch_size,
            cache_dir=cache_root / "clip",
            device=device,
            regressors=list(regressors),
            random_state=random_state,
            tta_mode=tta_mode,
            tta_resize=tta_resize,
        )
        clip_runs.append(clip_result)
        if best_clip is None or clip_result.val_metrics.spearman > best_clip.val_metrics.spearman:
            best_clip = clip_result

    assert best_clip is not None

    summary = {
        "dataset": dataset_name,
        "sizes": {
            "train": len(trimmed.train),
            "val": len(trimmed.val),
            "test": len(trimmed.test),
        },
        "resmem": {
            "checkpoint_path": str(resolved_checkpoint),
            "test_metrics": resmem_metrics.as_dict(),
        },
        "clip_search": [result.as_dict() for result in clip_runs],
        "best_clip": best_clip.as_dict(),
        "delta_vs_resmem": {
            "spearman": best_clip.test_metrics.spearman - resmem_metrics.spearman,
            "mse": best_clip.test_metrics.mse - resmem_metrics.mse,
        },
    }
    return BenchmarkRun(
        summary=summary,
        test_records=trimmed.test,
        resmem_predictions=resmem_predictions,
        challenger_predictions=best_clip.test_predictions,
    )


def benchmark_manifest(
    manifest_path: PathLike,
    *,
    root: Optional[PathLike] = None,
    clip_models: Sequence[str] = ("ViT-B-32",),
    clip_pretrained: str = "openai",
    regressors: Sequence[str] = ("ridge", "mlp"),
    batch_size: int = 32,
    device: str = "cpu",
    cache_dir: PathLike = ".cache/memscore",
    checkpoint_path: Optional[PathLike] = None,
    random_state: int = 0,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
    limit_test: Optional[int] = None,
    tta_mode: str = "none",
    tta_resize: Optional[int] = None,
) -> BenchmarkRun:
    """Benchmark frozen CLIP challengers against ``ResMem`` from a manifest.

    The manifest CSV must contain ``path``, ``score``, and ``split`` columns
    and may optionally contain ``id``.
    """

    splits = load_manifest(manifest_path, root=root)
    return _benchmark_dataset(
        splits,
        dataset_name="manifest",
        clip_models=clip_models,
        clip_pretrained=clip_pretrained,
        regressors=regressors,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
        checkpoint_path=checkpoint_path,
        random_state=random_state,
        limit_train=limit_train,
        limit_val=limit_val,
        limit_test=limit_test,
        tta_mode=tta_mode,
        tta_resize=tta_resize,
    )


def benchmark_lamem(
    lamem_root: PathLike,
    *,
    splits_dir: PathLike,
    fold: int = 1,
    train_file: Optional[PathLike] = None,
    val_file: Optional[PathLike] = None,
    test_file: Optional[PathLike] = None,
    clip_models: Sequence[str] = ("ViT-B-32",),
    clip_pretrained: str = "openai",
    regressors: Sequence[str] = ("ridge", "mlp"),
    batch_size: int = 32,
    device: str = "cpu",
    cache_dir: PathLike = ".cache/memscore",
    checkpoint_path: Optional[PathLike] = None,
    random_state: int = 0,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
    limit_test: Optional[int] = None,
    tta_mode: str = "none",
    tta_resize: Optional[int] = None,
) -> BenchmarkRun:
    """Benchmark frozen CLIP challengers against ``ResMem`` on LaMem."""

    splits = load_lamem_splits(
        root=lamem_root,
        splits_dir=splits_dir,
        fold=fold,
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
    )
    return _benchmark_dataset(
        splits,
        dataset_name="lamem",
        clip_models=clip_models,
        clip_pretrained=clip_pretrained,
        regressors=regressors,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
        checkpoint_path=checkpoint_path,
        random_state=random_state,
        limit_train=limit_train,
        limit_val=limit_val,
        limit_test=limit_test,
        tta_mode=tta_mode,
        tta_resize=tta_resize,
    )


def write_benchmark_predictions(run: BenchmarkRun, output_path: PathLike) -> Path:
    """Write side-by-side benchmark predictions for the test split."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "image_path", "ground_truth", "resmem_prediction", "clip_prediction"])
        for record, resmem_pred, clip_pred in zip(
            run.test_records,
            run.resmem_predictions,
            run.challenger_predictions,
        ):
            writer.writerow(
                [
                    record.identifier,
                    str(record.image_path),
                    f"{record.score:.8f}",
                    f"{float(resmem_pred):.8f}",
                    f"{float(clip_pred):.8f}",
                ]
            )
    return destination


def study_figrim(
    figrim_root: PathLike,
    *,
    protocols: Sequence[str] = ("across", "within"),
    score_types: Sequence[str] = ("corrected_hit_rate",),
    clip_models: Sequence[str] = ("ViT-B-32", "RN50"),
    clip_pretrained: str = "openai",
    clip_tta_mode: str = "none",
    clip_tta_resize: Optional[int] = None,
    pca_dims: Sequence[str] = ("none", "128", "64", "32"),
    pls_dims: Sequence[str] = (),
    num_splits: int = 10,
    batch_size: int = 32,
    device: str = "cpu",
    cache_dir: PathLike = ".cache/memscore",
    random_state: int = 0,
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
    ensemble_weight_step: float = 0.05,
    checkpoint_path: Optional[PathLike] = None,
    output_json: Optional[PathLike] = None,
    output_csv: Optional[PathLike] = None,
) -> dict[str, object]:
    """Run the repeated FIGRIM study used for ``ResMem`` comparisons.

    Parameters
    ----------
    figrim_root:
        Root directory of the FIGRIM download.
    protocols:
        Protocols to evaluate. Short aliases such as ``"across"`` and
        ``"within"`` are accepted.
    score_types:
        Target definitions to evaluate, for example ``"corrected_hit_rate"``.
    clip_models:
        One or more frozen CLIP vision backbones.
    clip_pretrained:
        ``open_clip`` pretrained tag.
    clip_tta_mode:
        CLIP embedding test-time augmentation mode.
    clip_tta_resize:
        Optional resize target for test-time augmentation.
    pca_dims:
        PCA dimensions to evaluate before ridge regression. Use ``"none"`` for
        full embeddings.
    pls_dims:
        Optional PLS component counts to evaluate.
    num_splits:
        Number of repeated stratified train/val/test splits.
    batch_size:
        Inference batch size for both ``ResMem`` and CLIP feature extraction.
    device:
        Torch device string such as ``"cpu"`` or ``"cuda"``.
    cache_dir:
        Cache directory for checkpoints and embeddings.
    random_state:
        Starting random seed used to derive repeated split seeds.
    train_size, val_size, test_size:
        Fractions assigned to each split.
    ensemble_weight_step:
        Step size used when searching validation-weighted ensembles.
    checkpoint_path:
        Optional explicit ``ResMem`` checkpoint path.
    output_json, output_csv:
        Optional output files written when both are provided.

    Returns
    -------
    dict
        A JSON-serializable study object containing split-level results and a
        summary table.
    """

    study = run_figrim_study(
        figrim_root=figrim_root,
        protocols=list(protocols),
        score_types=list(score_types),
        clip_models=list(clip_models),
        clip_pretrained=clip_pretrained,
        clip_tta_mode=clip_tta_mode,
        clip_tta_resize=clip_tta_resize,
        pca_dims=parse_pca_dims(list(pca_dims)),
        pls_dims=parse_positive_dims(list(pls_dims)),
        num_splits=num_splits,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
        random_state=random_state,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        ensemble_weight_step=ensemble_weight_step,
        resmem_checkpoint=checkpoint_path,
    )
    if output_json and output_csv:
        write_study_outputs(study, output_json=output_json, output_csv=output_csv)
    return study
