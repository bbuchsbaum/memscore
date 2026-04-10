from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .clip_regression import ClipEmbedder, normalize_tta_mode
from .data import RecordDataset, load_record_dataset
from .figrim import (
    _add_config_result,
    _load_prediction_cache,
    _paths_key,
    _write_prediction_cache,
    evaluate_mean_ensemble_on_splits,
    evaluate_pls_on_splits,
    evaluate_rank_mean_ensemble_on_splits,
    evaluate_rank_weighted_ensemble_on_splits,
    evaluate_resmem_on_splits,
    evaluate_ridge_on_splits,
    evaluate_stacked_ridge_ensemble_on_splits,
    evaluate_weighted_ensemble_on_splits,
    generate_repeated_splits,
    summarize_split_metrics,
)
from .resmem_baseline import ResMemScorer, ensure_resmem_checkpoint

PathLike = Union[Path, str]


@dataclass(frozen=True)
class ManifestStudyDataset:
    name: str
    records: list
    labels: list[str]
    label_column: Optional[str]


def load_manifest_study_dataset(
    manifest_path: PathLike,
    *,
    root: Optional[PathLike] = None,
    dataset_name: Optional[str] = None,
    path_column: str = "path",
    score_column: str = "score",
    id_column: Optional[str] = "id",
    label_column: Optional[str] = None,
) -> ManifestStudyDataset:
    dataset = load_record_dataset(
        manifest_path,
        root=root,
        path_column=path_column,
        score_column=score_column,
        id_column=id_column,
        label_column=label_column,
    )
    name = dataset_name or Path(manifest_path).expanduser().resolve().stem
    return ManifestStudyDataset(
        name=name,
        records=dataset.records,
        labels=dataset.labels,
        label_column=dataset.label_column,
    )


def run_manifest_study(
    manifest_path: PathLike,
    *,
    root: Optional[PathLike] = None,
    dataset_name: Optional[str] = None,
    path_column: str = "path",
    score_column: str = "score",
    id_column: Optional[str] = "id",
    label_column: Optional[str] = None,
    clip_models: list[str],
    clip_pretrained: str,
    clip_tta_mode: str,
    clip_tta_resize: Optional[int],
    pca_dims: list[Optional[int]],
    pls_dims: list[int],
    num_splits: int,
    batch_size: int,
    device: str,
    cache_dir: PathLike,
    random_state: int = 0,
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
    ensemble_weight_step: float = 0.05,
    resmem_checkpoint: Optional[PathLike] = None,
    limit_records: Optional[int] = None,
) -> dict[str, object]:
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    normalized_tta_mode = normalize_tta_mode(clip_tta_mode)

    checkpoint_path = (
        Path(resmem_checkpoint).expanduser().resolve()
        if resmem_checkpoint
        else ensure_resmem_checkpoint(cache_root / "resmem")
    )
    resmem_scorer = ResMemScorer(checkpoint_path=checkpoint_path, batch_size=batch_size, device=device)

    dataset = load_manifest_study_dataset(
        manifest_path,
        root=root,
        dataset_name=dataset_name,
        path_column=path_column,
        score_column=score_column,
        id_column=id_column,
        label_column=label_column,
    )

    if limit_records is not None and limit_records > 0 and len(dataset.records) > limit_records:
        rng = np.random.default_rng(random_state)
        chosen = np.sort(rng.choice(len(dataset.records), size=limit_records, replace=False))
        records = [dataset.records[idx] for idx in chosen.tolist()]
        labels = [dataset.labels[idx] for idx in chosen.tolist()]
    else:
        records = dataset.records
        labels = dataset.labels

    print(
        f"[manifest-study] dataset={dataset.name} records={len(records)} "
        f"label_column={dataset.label_column or 'none'} num_splits={num_splits}"
    )

    splits = generate_repeated_splits(
        labels,
        num_splits=num_splits,
        random_state=random_state,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )
    score_array = np.asarray([record.score for record in records], dtype=np.float32)
    paths_key = _paths_key(records)

    prediction_cache_path = cache_root / "resmem" / f"{dataset.name}_predictions.npz"
    cached_predictions = _load_prediction_cache(prediction_cache_path, records)
    if cached_predictions is None:
        print(f"[manifest-study] scoring ResMem on {dataset.name}")
        cached_predictions = resmem_scorer.predict(records)
        _write_prediction_cache(prediction_cache_path, records, cached_predictions)
    else:
        print(f"[manifest-study] loaded cached ResMem predictions for {dataset.name}")
    resmem_evaluations = evaluate_resmem_on_splits(cached_predictions, score_array, splits)

    dataset_result: dict[str, object] = {
        "protocol": dataset.name,
        "score_type": score_column,
        "num_records": len(records),
        "label_column": dataset.label_column,
        "splits": [split.as_dict() for split in splits],
        "clip_tta_mode": normalized_tta_mode,
        "clip_tta_resize": clip_tta_resize,
        "resmem": {
            "summary": summarize_split_metrics(resmem_evaluations),
            "split_results": resmem_evaluations,
        },
        "model_configs": [],
    }

    summary_rows: list[dict[str, object]] = []
    single_model_no_reduction: dict[str, list[dict[str, object]]] = {}
    feature_map: dict[str, np.ndarray] = {}
    embedding_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}

    for clip_model in clip_models:
        embedding_key = (
            f"{clip_model}::{clip_pretrained}::{normalized_tta_mode}::{clip_tta_resize}",
            paths_key,
        )
        if embedding_key not in embedding_cache:
            print(f"[manifest-study] extracting CLIP embeddings model={clip_model} pretrained={clip_pretrained}")
            embedder = ClipEmbedder(
                model_name=clip_model,
                pretrained=clip_pretrained,
                batch_size=batch_size,
                device=device,
                tta_mode=normalized_tta_mode,
                tta_resize=clip_tta_resize,
            )
            bundle = embedder.embed_split(
                records,
                split_name=f"{dataset.name}_all",
                cache_dir=cache_root / "clip",
            )
            embedding_cache[embedding_key] = bundle.features

        features = embedding_cache[embedding_key]
        feature_map[clip_model] = features
        for pca_dim in pca_dims:
            print(
                f"[manifest-study] evaluating model={clip_model} ridge "
                f"pca_dim={'none' if pca_dim is None else pca_dim}"
            )
            evaluations = evaluate_ridge_on_splits(
                features=features,
                scores=score_array,
                splits=splits,
                pca_dim=pca_dim,
                keep_predictions=pca_dim is None,
            )
            if pca_dim is None:
                single_model_no_reduction[clip_model] = evaluations
            config_result = _add_config_result(
                dataset_result,
                summary_rows,
                resmem_evaluations,
                feature_source=clip_model,
                method="ridge",
                reducer="none" if pca_dim is None else "pca",
                components=pca_dim,
                evaluations=evaluations,
            )
            print(
                f"[manifest-study] done model={clip_model} ridge pca_dim={'none' if pca_dim is None else pca_dim} "
                f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
            )

        for pls_dim in pls_dims:
            print(f"[manifest-study] evaluating model={clip_model} pls components={pls_dim}")
            evaluations = evaluate_pls_on_splits(
                features=features,
                scores=score_array,
                splits=splits,
                n_components=pls_dim,
            )
            config_result = _add_config_result(
                dataset_result,
                summary_rows,
                resmem_evaluations,
                feature_source=clip_model,
                method="pls",
                reducer="pls",
                components=pls_dim,
                evaluations=evaluations,
            )
            print(
                f"[manifest-study] done model={clip_model} pls components={pls_dim} "
                f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
            )

    if len(clip_models) > 1:
        fused_name = "+".join(clip_models)
        fused_features = np.concatenate([feature_map[model_name] for model_name in clip_models], axis=1)
        for pca_dim in pca_dims:
            print(
                f"[manifest-study] evaluating fusion={fused_name} ridge "
                f"pca_dim={'none' if pca_dim is None else pca_dim}"
            )
            evaluations = evaluate_ridge_on_splits(
                features=fused_features,
                scores=score_array,
                splits=splits,
                pca_dim=pca_dim,
            )
            config_result = _add_config_result(
                dataset_result,
                summary_rows,
                resmem_evaluations,
                feature_source=fused_name,
                method="ridge",
                reducer="none" if pca_dim is None else "pca",
                components=pca_dim,
                evaluations=evaluations,
            )
            print(
                f"[manifest-study] done fusion={fused_name} ridge pca_dim={'none' if pca_dim is None else pca_dim} "
                f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
            )

        for pls_dim in pls_dims:
            print(f"[manifest-study] evaluating fusion={fused_name} pls components={pls_dim}")
            evaluations = evaluate_pls_on_splits(
                features=fused_features,
                scores=score_array,
                splits=splits,
                n_components=pls_dim,
            )
            config_result = _add_config_result(
                dataset_result,
                summary_rows,
                resmem_evaluations,
                feature_source=fused_name,
                method="pls",
                reducer="pls",
                components=pls_dim,
                evaluations=evaluations,
            )
            print(
                f"[manifest-study] done fusion={fused_name} pls components={pls_dim} "
                f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
            )

        ensemble_members = [
            single_model_no_reduction[model_name]
            for model_name in clip_models
            if model_name in single_model_no_reduction
        ]
        if len(ensemble_members) >= 2:
            for method_name, evaluator in (
                ("ensemble_mean", lambda: evaluate_mean_ensemble_on_splits(ensemble_members, scores=score_array, splits=splits)),
                (
                    "ensemble_val_weighted",
                    lambda: evaluate_weighted_ensemble_on_splits(
                        ensemble_members,
                        scores=score_array,
                        splits=splits,
                        step=ensemble_weight_step,
                    ),
                ),
                ("ensemble_rank_mean", lambda: evaluate_rank_mean_ensemble_on_splits(ensemble_members, scores=score_array, splits=splits)),
                (
                    "ensemble_rank_weighted",
                    lambda: evaluate_rank_weighted_ensemble_on_splits(
                        ensemble_members,
                        scores=score_array,
                        splits=splits,
                        step=ensemble_weight_step,
                    ),
                ),
                ("ensemble_stacked_ridge", lambda: evaluate_stacked_ridge_ensemble_on_splits(ensemble_members, scores=score_array, splits=splits)),
            ):
                print(f"[manifest-study] evaluating {method_name} ensemble={' + '.join(clip_models)}")
                evaluations = evaluator()
                config_result = _add_config_result(
                    dataset_result,
                    summary_rows,
                    resmem_evaluations,
                    feature_source="ensemble:" + "+".join(clip_models),
                    method=method_name,
                    reducer="n/a",
                    components=None,
                    evaluations=evaluations,
                )
                print(
                    f"[manifest-study] done {method_name} ensemble={' + '.join(clip_models)} "
                    f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                    f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
                )

    return {
        "metadata": {
            "manifest_path": str(Path(manifest_path).expanduser().resolve()),
            "root": str(Path(root).expanduser().resolve()) if root else None,
            "dataset_name": dataset.name,
            "path_column": path_column,
            "score_column": score_column,
            "id_column": id_column,
            "label_column": label_column,
            "clip_models": clip_models,
            "clip_pretrained": clip_pretrained,
            "clip_tta_mode": normalized_tta_mode,
            "clip_tta_resize": clip_tta_resize,
            "pca_dims": ["none" if dim is None else int(dim) for dim in pca_dims],
            "pls_dims": pls_dims,
            "num_splits": num_splits,
            "batch_size": batch_size,
            "device": device,
            "random_state": random_state,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "ensemble_weight_step": ensemble_weight_step,
            "checkpoint_path": str(checkpoint_path),
            "limit_records": limit_records,
        },
        "studies": [dataset_result],
        "summary_rows": summary_rows,
    }
