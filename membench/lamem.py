from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .clip_regression import ClipEmbedder, normalize_tta_mode
from .data import Record, resolve_lamem_split_files
from .figrim import (
    SplitIndices,
    _add_config_result,
    _load_prediction_cache,
    _write_prediction_cache,
    evaluate_mean_ensemble_on_splits,
    evaluate_pls_on_splits,
    evaluate_rank_mean_ensemble_on_splits,
    evaluate_rank_weighted_ensemble_on_splits,
    evaluate_resmem_on_splits,
    evaluate_ridge_on_splits,
    evaluate_stacked_ridge_ensemble_on_splits,
    evaluate_weighted_ensemble_on_splits,
    summarize_split_metrics,
)
from .metrics import compute_metrics
from .resmem_baseline import ResMemScorer, ensure_resmem_checkpoint

PathLike = Union[Path, str]


@dataclass(frozen=True)
class LaMemStudyData:
    records: list[Record]
    scores: np.ndarray
    splits: list[SplitIndices]
    original_sizes: dict[str, int]
    union_sizes: dict[str, int]
    sampled_sizes: dict[str, int]


def _sample_indices(total: int, limit: Optional[int], rng: np.random.Generator) -> np.ndarray:
    if limit is None or limit <= 0 or limit >= total:
        return np.arange(total, dtype=np.int64)
    return np.sort(rng.choice(total, size=limit, replace=False).astype(np.int64))


def _build_position_map(selected_indices: np.ndarray) -> np.ndarray:
    mapping = np.full(int(selected_indices.max()) + 1, -1, dtype=np.int64)
    mapping[selected_indices] = np.arange(selected_indices.size, dtype=np.int64)
    return mapping


def _merge_split_records(
    train_records: list[Record],
    val_records: list[Record],
    test_records: list[Record],
    *,
    limit_train: Optional[int],
    limit_val: Optional[int],
    limit_test: Optional[int],
    num_splits: int,
    random_state: int,
) -> LaMemStudyData:
    if num_splits <= 0:
        raise ValueError("num_splits must be positive")

    sampled_train: list[np.ndarray] = []
    sampled_val: list[np.ndarray] = []
    sampled_test: list[np.ndarray] = []

    for split_idx in range(num_splits):
        rng = np.random.default_rng(random_state + split_idx)
        sampled_train.append(_sample_indices(len(train_records), limit_train, rng))
        sampled_val.append(_sample_indices(len(val_records), limit_val, rng))
        sampled_test.append(_sample_indices(len(test_records), limit_test, rng))

    train_union = np.unique(np.concatenate(sampled_train, axis=0))
    val_union = np.unique(np.concatenate(sampled_val, axis=0))
    test_union = np.unique(np.concatenate(sampled_test, axis=0))

    train_pos = _build_position_map(train_union)
    val_pos = _build_position_map(val_union)
    test_pos = _build_position_map(test_union)

    train_offset = 0
    val_offset = int(train_union.size)
    test_offset = int(train_union.size + val_union.size)

    merged_records = (
        [train_records[idx] for idx in train_union.tolist()]
        + [val_records[idx] for idx in val_union.tolist()]
        + [test_records[idx] for idx in test_union.tolist()]
    )
    score_array = np.asarray([record.score for record in merged_records], dtype=np.float32)

    splits: list[SplitIndices] = []
    for split_idx, (train_idx, val_idx, test_idx) in enumerate(zip(sampled_train, sampled_val, sampled_test)):
        splits.append(
            SplitIndices(
                train=train_offset + train_pos[train_idx],
                val=val_offset + val_pos[val_idx],
                test=test_offset + test_pos[test_idx],
                seed=random_state + split_idx,
            )
        )

    return LaMemStudyData(
        records=merged_records,
        scores=score_array,
        splits=splits,
        original_sizes={
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        union_sizes={
            "train": int(train_union.size),
            "val": int(val_union.size),
            "test": int(test_union.size),
        },
        sampled_sizes={
            "train": len(train_records) if limit_train is None or limit_train <= 0 else min(limit_train, len(train_records)),
            "val": len(val_records) if limit_val is None or limit_val <= 0 else min(limit_val, len(val_records)),
            "test": len(test_records) if limit_test is None or limit_test <= 0 else min(limit_test, len(test_records)),
        },
    )


def _prediction_member_evaluations(
    predictions: np.ndarray,
    scores: np.ndarray,
    splits: list[SplitIndices],
) -> list[dict[str, object]]:
    evaluations: list[dict[str, object]] = []
    for split in splits:
        val_pred = predictions[split.val].astype(np.float32)
        test_pred = predictions[split.test].astype(np.float32)
        evaluations.append(
            {
                "seed": split.seed,
                "val_metrics": compute_metrics(scores[split.val], val_pred).as_dict(),
                "test_metrics": compute_metrics(scores[split.test], test_pred).as_dict(),
                "_val_predictions": val_pred,
                "_test_predictions": test_pred,
            }
        )
    return evaluations


def run_lamem_study(
    lamem_root: PathLike,
    *,
    splits_dir: PathLike,
    fold: int = 1,
    train_file: Optional[PathLike] = None,
    val_file: Optional[PathLike] = None,
    test_file: Optional[PathLike] = None,
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
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
    limit_test: Optional[int] = None,
    ensemble_weight_step: float = 0.05,
    resmem_checkpoint: Optional[PathLike] = None,
    include_resmem_ensembles: bool = True,
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

    train_path, val_path, test_path = resolve_lamem_split_files(
        splits_dir=splits_dir,
        fold=fold,
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
    )

    from .data import load_lamem_split_file

    train_records = load_lamem_split_file(lamem_root, train_path)
    val_records = load_lamem_split_file(lamem_root, val_path)
    test_records = load_lamem_split_file(lamem_root, test_path)

    study_data = _merge_split_records(
        train_records,
        val_records,
        test_records,
        limit_train=limit_train,
        limit_val=limit_val,
        limit_test=limit_test,
        num_splits=num_splits,
        random_state=random_state,
    )
    print(
        f"[lamem-study] fold={fold} num_splits={num_splits} "
        f"sampled=train:{study_data.sampled_sizes['train']} val:{study_data.sampled_sizes['val']} test:{study_data.sampled_sizes['test']} "
        f"union=train:{study_data.union_sizes['train']} val:{study_data.union_sizes['val']} test:{study_data.union_sizes['test']}"
    )

    prediction_cache_path = cache_root / "resmem" / f"lamem_fold{fold}_study_predictions.npz"
    cached_predictions = _load_prediction_cache(prediction_cache_path, study_data.records)
    if cached_predictions is None:
        print("[lamem-study] scoring ResMem on sampled LaMem union")
        cached_predictions = resmem_scorer.predict(study_data.records)
        _write_prediction_cache(prediction_cache_path, study_data.records, cached_predictions)
    else:
        print("[lamem-study] loaded cached ResMem predictions for sampled LaMem union")

    resmem_evaluations = evaluate_resmem_on_splits(cached_predictions, study_data.scores, study_data.splits)
    resmem_member = _prediction_member_evaluations(cached_predictions, study_data.scores, study_data.splits)

    dataset_result: dict[str, object] = {
        "protocol": f"lamem_fold_{fold}",
        "score_type": "memorability",
        "num_records": len(study_data.records),
        "original_split_sizes": study_data.original_sizes,
        "sampled_split_sizes": study_data.sampled_sizes,
        "union_split_sizes": study_data.union_sizes,
        "split_files": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
        "splits": [split.as_dict() for split in study_data.splits],
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

    for clip_model in clip_models:
        print(f"[lamem-study] extracting CLIP embeddings model={clip_model} pretrained={clip_pretrained}")
        embedder = ClipEmbedder(
            model_name=clip_model,
            pretrained=clip_pretrained,
            batch_size=batch_size,
            device=device,
            tta_mode=normalized_tta_mode,
            tta_resize=clip_tta_resize,
        )
        bundle = embedder.embed_split(
            study_data.records,
            split_name=f"lamem_fold{fold}_study",
            cache_dir=cache_root / "clip",
        )
        features = bundle.features
        feature_map[clip_model] = features

        for pca_dim in pca_dims:
            print(
                f"[lamem-study] evaluating model={clip_model} ridge "
                f"pca_dim={'none' if pca_dim is None else pca_dim}"
            )
            evaluations = evaluate_ridge_on_splits(
                features=features,
                scores=study_data.scores,
                splits=study_data.splits,
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
                f"[lamem-study] done model={clip_model} ridge pca_dim={'none' if pca_dim is None else pca_dim} "
                f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
            )

        for pls_dim in pls_dims:
            print(f"[lamem-study] evaluating model={clip_model} pls components={pls_dim}")
            evaluations = evaluate_pls_on_splits(
                features=features,
                scores=study_data.scores,
                splits=study_data.splits,
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
                f"[lamem-study] done model={clip_model} pls components={pls_dim} "
                f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
            )

    ensemble_members = [single_model_no_reduction[model_name] for model_name in clip_models if model_name in single_model_no_reduction]
    if len(clip_models) > 1:
        fused_name = "+".join(clip_models)
        fused_features = np.concatenate([feature_map[model_name] for model_name in clip_models], axis=1)
        for pca_dim in pca_dims:
            print(
                f"[lamem-study] evaluating fusion={fused_name} ridge "
                f"pca_dim={'none' if pca_dim is None else pca_dim}"
            )
            evaluations = evaluate_ridge_on_splits(
                features=fused_features,
                scores=study_data.scores,
                splits=study_data.splits,
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
                f"[lamem-study] done fusion={fused_name} ridge pca_dim={'none' if pca_dim is None else pca_dim} "
                f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
            )

        for pls_dim in pls_dims:
            print(f"[lamem-study] evaluating fusion={fused_name} pls components={pls_dim}")
            evaluations = evaluate_pls_on_splits(
                features=fused_features,
                scores=study_data.scores,
                splits=study_data.splits,
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
                f"[lamem-study] done fusion={fused_name} pls components={pls_dim} "
                f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
            )

    if len(ensemble_members) >= 2:
        for method_name, evaluator in (
            ("ensemble_mean", lambda: evaluate_mean_ensemble_on_splits(ensemble_members, scores=study_data.scores, splits=study_data.splits)),
            (
                "ensemble_val_weighted",
                lambda: evaluate_weighted_ensemble_on_splits(
                    ensemble_members,
                    scores=study_data.scores,
                    splits=study_data.splits,
                    step=ensemble_weight_step,
                ),
            ),
            ("ensemble_rank_mean", lambda: evaluate_rank_mean_ensemble_on_splits(ensemble_members, scores=study_data.scores, splits=study_data.splits)),
            (
                "ensemble_rank_weighted",
                lambda: evaluate_rank_weighted_ensemble_on_splits(
                    ensemble_members,
                    scores=study_data.scores,
                    splits=study_data.splits,
                    step=ensemble_weight_step,
                ),
            ),
            ("ensemble_stacked_ridge", lambda: evaluate_stacked_ridge_ensemble_on_splits(ensemble_members, scores=study_data.scores, splits=study_data.splits)),
        ):
            print(f"[lamem-study] evaluating {method_name} ensemble={' + '.join(clip_models)}")
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
                f"[lamem-study] done {method_name} ensemble={' + '.join(clip_models)} "
                f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
            )

    if include_resmem_ensembles and ensemble_members:
        hybrid_members = [resmem_member] + ensemble_members
        hybrid_name = "resmem+" + "+".join(clip_models)
        for method_name, evaluator in (
            ("ensemble_hybrid_mean", lambda: evaluate_mean_ensemble_on_splits(hybrid_members, scores=study_data.scores, splits=study_data.splits)),
            (
                "ensemble_hybrid_val_weighted",
                lambda: evaluate_weighted_ensemble_on_splits(
                    hybrid_members,
                    scores=study_data.scores,
                    splits=study_data.splits,
                    step=ensemble_weight_step,
                ),
            ),
            ("ensemble_hybrid_rank_mean", lambda: evaluate_rank_mean_ensemble_on_splits(hybrid_members, scores=study_data.scores, splits=study_data.splits)),
            (
                "ensemble_hybrid_rank_weighted",
                lambda: evaluate_rank_weighted_ensemble_on_splits(
                    hybrid_members,
                    scores=study_data.scores,
                    splits=study_data.splits,
                    step=ensemble_weight_step,
                ),
            ),
            ("ensemble_hybrid_stacked_ridge", lambda: evaluate_stacked_ridge_ensemble_on_splits(hybrid_members, scores=study_data.scores, splits=study_data.splits)),
        ):
            print(f"[lamem-study] evaluating {method_name} ensemble={hybrid_name}")
            evaluations = evaluator()
            config_result = _add_config_result(
                dataset_result,
                summary_rows,
                resmem_evaluations,
                feature_source=hybrid_name,
                method=method_name,
                reducer="n/a",
                components=None,
                evaluations=evaluations,
            )
            print(
                f"[lamem-study] done {method_name} ensemble={hybrid_name} "
                f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
            )

    return {
        "metadata": {
            "lamem_root": str(Path(lamem_root).expanduser().resolve()),
            "splits_dir": str(Path(splits_dir).expanduser().resolve()),
            "fold": fold,
            "train_file": str(train_path),
            "val_file": str(val_path),
            "test_file": str(test_path),
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
            "limit_train": limit_train,
            "limit_val": limit_val,
            "limit_test": limit_test,
            "ensemble_weight_step": ensemble_weight_step,
            "checkpoint_path": str(checkpoint_path),
            "include_resmem_ensembles": include_resmem_ensembles,
        },
        "studies": [dataset_result],
        "summary_rows": summary_rows,
    }
