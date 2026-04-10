from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from .clip_regression import ClipEmbedder, normalize_tta_mode
from .data import load_manifest
from .figrim import (
    SplitIndices,
    compare_against_resmem,
    evaluate_mean_ensemble_on_splits,
    evaluate_rank_mean_ensemble_on_splits,
    evaluate_ridge_on_splits,
    evaluate_stacked_ridge_ensemble_on_splits,
    evaluate_weighted_ensemble_on_splits,
    summarize_split_metrics,
)
from .metrics import compute_metrics
from .resmem_baseline import ResMemScorer, ensure_resmem_checkpoint

PathLike = Union[Path, str]


def _strip_internal_fields(evaluation: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in evaluation.items() if not key.startswith("_")}


def run_locked_ensemble_benchmark(
    *,
    manifest_path: PathLike,
    root: Optional[PathLike],
    dataset_name: Optional[str],
    clip_models: list[str],
    clip_pretrained: str,
    clip_tta_mode: str,
    clip_tta_resize: Optional[int],
    batch_size: int,
    device: str,
    cache_dir: PathLike,
    ensemble_weight_step: float = 0.05,
    resmem_checkpoint: Optional[PathLike] = None,
) -> dict[str, object]:
    """Evaluate ridge heads and their ensembles on a locked train/val/test manifest."""

    manifest = Path(manifest_path).expanduser().resolve()
    image_root = Path(root).expanduser().resolve() if root is not None else None
    name = dataset_name or manifest.stem
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    normalized_tta = normalize_tta_mode(clip_tta_mode)

    splits = load_manifest(manifest, root=image_root)
    all_records = [*splits.train, *splits.val, *splits.test]
    score_array = np.asarray([record.score for record in all_records], dtype=np.float32)
    split = SplitIndices(
        train=np.arange(0, len(splits.train), dtype=np.int64),
        val=np.arange(len(splits.train), len(splits.train) + len(splits.val), dtype=np.int64),
        test=np.arange(len(splits.train) + len(splits.val), len(all_records), dtype=np.int64),
        seed=0,
    )
    split_list = [split]

    checkpoint_path = (
        Path(resmem_checkpoint).expanduser().resolve()
        if resmem_checkpoint is not None
        else ensure_resmem_checkpoint(cache_root / "resmem")
    )
    resmem = ResMemScorer(checkpoint_path=checkpoint_path, batch_size=batch_size, device=device)
    resmem_predictions = resmem.predict(all_records)
    resmem_evaluations = [
        {
            "seed": 0,
            "test_metrics": compute_metrics(score_array[split.test], resmem_predictions[split.test]).as_dict(),
        }
    ]

    member_evaluations: list[list[dict[str, object]]] = []
    single_models: dict[str, object] = {}
    for model_name in clip_models:
        embedder = ClipEmbedder(
            model_name=model_name,
            pretrained=clip_pretrained,
            batch_size=batch_size,
            device=device,
            tta_mode=normalized_tta,
            tta_resize=clip_tta_resize,
        )
        bundle = embedder.embed_split(
            all_records,
            split_name=f"{name}_all",
            cache_dir=cache_root / "clip",
        )
        evaluations = evaluate_ridge_on_splits(
            features=bundle.features,
            scores=score_array,
            splits=split_list,
            pca_dim=None,
            keep_predictions=True,
        )
        member_evaluations.append(evaluations)
        single_models[model_name] = {
            "summary": summarize_split_metrics(evaluations),
            "comparison_vs_resmem": compare_against_resmem(resmem_evaluations, evaluations),
            "split_results": [_strip_internal_fields(evaluations[0])],
        }

    ensemble_evaluations = {
        "ensemble_mean": evaluate_mean_ensemble_on_splits(member_evaluations, scores=score_array, splits=split_list),
        "ensemble_rank_mean": evaluate_rank_mean_ensemble_on_splits(
            member_evaluations,
            scores=score_array,
            splits=split_list,
        ),
        "ensemble_stacked_ridge": evaluate_stacked_ridge_ensemble_on_splits(
            member_evaluations,
            scores=score_array,
            splits=split_list,
        ),
    }
    if len(member_evaluations) > 1:
        ensemble_evaluations["ensemble_weighted"] = evaluate_weighted_ensemble_on_splits(
            member_evaluations,
            scores=score_array,
            splits=split_list,
            step=ensemble_weight_step,
        )

    ensembles: dict[str, object] = {}
    for method, evaluations in ensemble_evaluations.items():
        ensembles[method] = {
            "summary": summarize_split_metrics(evaluations),
            "comparison_vs_resmem": compare_against_resmem(resmem_evaluations, evaluations),
            "split_results": [_strip_internal_fields(evaluations[0])],
        }

    return {
        "metadata": {
            "manifest": str(manifest),
            "root": str(image_root) if image_root is not None else None,
            "dataset_name": name,
            "clip_models": list(clip_models),
            "clip_pretrained": clip_pretrained,
            "clip_tta_mode": normalized_tta,
            "clip_tta_resize": clip_tta_resize,
            "regressor": "ridge",
            "standard_ensemble": "ensemble_mean",
            "batch_size": batch_size,
            "device": device,
            "cache_dir": str(cache_root),
            "ensemble_weight_step": ensemble_weight_step,
        },
        "sizes": {"train": len(splits.train), "val": len(splits.val), "test": len(splits.test)},
        "resmem": {
            "test_metrics": resmem_evaluations[0]["test_metrics"],
            "checkpoint_path": str(checkpoint_path),
        },
        "single_models": single_models,
        "ensembles": ensembles,
        "standard_model": ensembles["ensemble_mean"],
    }
