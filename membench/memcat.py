from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .clip_regression import ClipEmbedder, normalize_tta_mode
from .data import Record, load_memcat_records
from .figrim import (
    SplitIndices,
    _add_config_result,
    _load_prediction_cache,
    _paths_key,
    _strip_prediction_fields,
    _write_prediction_cache,
    build_ridge_pipeline,
    compare_against_resmem,
    evaluate_mean_ensemble_on_splits,
    evaluate_pls_on_splits,
    evaluate_rank_mean_ensemble_on_splits,
    evaluate_rank_weighted_ensemble_on_splits,
    evaluate_resmem_on_splits,
    evaluate_ridge_on_splits,
    evaluate_stacked_ridge_ensemble_on_splits,
    evaluate_weighted_ensemble_on_splits,
    generate_repeated_splits,
    parse_pca_dims,
    parse_positive_dims,
    summarize_split_metrics,
    write_study_outputs,
)
from .metrics import compute_metrics
from .resmem_baseline import ResMemScorer, ensure_resmem_checkpoint

PathLike = Union[Path, str]


@dataclass(frozen=True)
class MemCatDataset:
    name: str
    score_column: str
    records: list[Record]
    categories: list[str]


def load_memcat_dataset(
    root: PathLike,
    csv_path: Optional[PathLike] = None,
    score_column: str = "memorability_w_fa_correction",
) -> MemCatDataset:
    """Load MemCat records and extract per-image categories for stratification."""
    dataset_root = Path(root).expanduser().resolve()
    if csv_path:
        source_csv = Path(csv_path).expanduser().resolve()
    else:
        candidate_data = dataset_root / "data" / "memcat_image_data.csv"
        candidate_root = dataset_root / "memcat_image_data.csv"
        if candidate_data.exists():
            source_csv = candidate_data
        elif candidate_root.exists():
            source_csv = candidate_root
        else:
            raise FileNotFoundError(
                f"Could not find memcat_image_data.csv in {dataset_root / 'data'} or {dataset_root}. "
                "Pass csv_path explicitly."
            )

    # Resolve image directory: try <root>/images then <root>/MemCat then <root>
    if (dataset_root / "images").is_dir():
        image_base = Path("images")
    elif (dataset_root / "MemCat").is_dir():
        image_base = Path("MemCat")
    else:
        image_base = Path(".")

    records: list[Record] = []
    categories: list[str] = []

    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row in {source_csv}")

        fieldnames = set(reader.fieldnames)
        score_key = score_column if score_column in fieldnames else None
        for fallback in ("memorability", "memo_score", "score"):
            if score_key is None and fallback in fieldnames:
                score_key = fallback

        if score_key is None:
            raise KeyError(f"Could not find a memorability score column in {source_csv}")

        for row in reader:
            image_name = row.get("image") or row.get("image_file") or row.get("image_name") or row.get("img")
            category = row.get("category") or row.get("supercategory") or row.get("cat")
            subcategory = row.get("subcategory") or row.get("subcat") or row.get("scat")
            if not image_name or not category or not subcategory:
                raise ValueError(
                    "MemCat loader expects image/category/subcategory columns. "
                    "Use a custom manifest if your export differs from the published CSV."
                )
            rel_path = image_base / category / subcategory / image_name
            records.append(
                Record(
                    image_path=(dataset_root / rel_path).resolve(),
                    score=float(row[score_key]),
                    identifier=str(rel_path),
                )
            )
            categories.append(category)

    return MemCatDataset(
        name="memcat",
        score_column=score_key,
        records=records,
        categories=categories,
    )


def run_memcat_study(
    memcat_root: PathLike,
    *,
    csv_path: Optional[PathLike] = None,
    score_column: str = "memorability_w_fa_correction",
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
) -> dict[str, object]:
    """Run a repeated-splits MemCat study comparing CLIP challengers to ResMem."""

    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    normalized_tta_mode = normalize_tta_mode(clip_tta_mode)

    checkpoint_path = (
        Path(resmem_checkpoint).expanduser().resolve()
        if resmem_checkpoint
        else ensure_resmem_checkpoint(cache_root / "resmem")
    )
    resmem_scorer = ResMemScorer(checkpoint_path=checkpoint_path, batch_size=batch_size, device=device)

    dataset = load_memcat_dataset(memcat_root, csv_path=csv_path, score_column=score_column)
    print(
        f"[memcat-study] score_column={dataset.score_column} "
        f"records={len(dataset.records)} num_splits={num_splits}"
    )

    splits = generate_repeated_splits(
        dataset.categories,
        num_splits=num_splits,
        random_state=random_state,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )
    score_array = np.asarray([record.score for record in dataset.records], dtype=np.float32)
    paths_key = _paths_key(dataset.records)

    # --- ResMem baseline ---
    prediction_cache_path = cache_root / "resmem" / "memcat_predictions.npz"
    cached_predictions = _load_prediction_cache(prediction_cache_path, dataset.records)
    if cached_predictions is None:
        print("[memcat-study] scoring ResMem on full MemCat set")
        cached_predictions = resmem_scorer.predict(dataset.records)
        _write_prediction_cache(prediction_cache_path, dataset.records, cached_predictions)
    else:
        print("[memcat-study] loaded cached ResMem predictions for MemCat")
    resmem_evaluations = evaluate_resmem_on_splits(cached_predictions, score_array, splits)

    dataset_result: dict[str, object] = {
        "protocol": "memcat",
        "score_type": dataset.score_column,
        "num_records": len(dataset.records),
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

    # --- Per-CLIP-model evaluation ---
    for clip_model in clip_models:
        print(f"[memcat-study] extracting CLIP embeddings model={clip_model} pretrained={clip_pretrained}")
        embedder = ClipEmbedder(
            model_name=clip_model,
            pretrained=clip_pretrained,
            batch_size=batch_size,
            device=device,
            tta_mode=normalized_tta_mode,
            tta_resize=clip_tta_resize,
        )
        bundle = embedder.embed_split(
            dataset.records,
            split_name="memcat_all",
            cache_dir=cache_root / "clip",
        )
        features = bundle.features
        feature_map[clip_model] = features

        for pca_dim in pca_dims:
            print(
                f"[memcat-study] evaluating model={clip_model} ridge "
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
                f"[memcat-study] done model={clip_model} ridge pca_dim={'none' if pca_dim is None else pca_dim} "
                f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
            )

        for pls_dim in pls_dims:
            print(f"[memcat-study] evaluating model={clip_model} pls components={pls_dim}")
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
                f"[memcat-study] done model={clip_model} pls components={pls_dim} "
                f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
            )

    # --- Multi-model: fusion + ensembles ---
    if len(clip_models) > 1:
        fused_name = "+".join(clip_models)
        fused_features = np.concatenate([feature_map[model_name] for model_name in clip_models], axis=1)
        for pca_dim in pca_dims:
            print(
                f"[memcat-study] evaluating fusion={fused_name} ridge "
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
                f"[memcat-study] done fusion={fused_name} ridge pca_dim={'none' if pca_dim is None else pca_dim} "
                f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
            )

        for pls_dim in pls_dims:
            print(f"[memcat-study] evaluating fusion={fused_name} pls components={pls_dim}")
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
                f"[memcat-study] done fusion={fused_name} pls components={pls_dim} "
                f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
            )

        ensemble_members = [
            single_model_no_reduction[model_name]
            for model_name in clip_models
            if model_name in single_model_no_reduction
        ]
        if len(ensemble_members) >= 2:
            print(f"[memcat-study] evaluating mean ensemble={' + '.join(clip_models)}")
            evaluations = evaluate_mean_ensemble_on_splits(
                member_evaluations=ensemble_members,
                scores=score_array,
                splits=splits,
            )
            config_result = _add_config_result(
                dataset_result,
                summary_rows,
                resmem_evaluations,
                feature_source="ensemble:" + "+".join(clip_models),
                method="ensemble_mean",
                reducer="n/a",
                components=None,
                evaluations=evaluations,
            )
            print(
                f"[memcat-study] done mean ensemble={' + '.join(clip_models)} "
                f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
            )

            print(f"[memcat-study] evaluating val-weighted ensemble={' + '.join(clip_models)}")
            evaluations = evaluate_weighted_ensemble_on_splits(
                member_evaluations=ensemble_members,
                scores=score_array,
                splits=splits,
                step=ensemble_weight_step,
            )
            config_result = _add_config_result(
                dataset_result,
                summary_rows,
                resmem_evaluations,
                feature_source="ensemble_weighted:" + "+".join(clip_models),
                method="ensemble_val_weighted",
                reducer="n/a",
                components=None,
                evaluations=evaluations,
            )
            print(
                f"[memcat-study] done val-weighted ensemble={' + '.join(clip_models)} "
                f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
            )

            print(f"[memcat-study] evaluating rank-mean ensemble={' + '.join(clip_models)}")
            evaluations = evaluate_rank_mean_ensemble_on_splits(
                member_evaluations=ensemble_members,
                scores=score_array,
                splits=splits,
            )
            config_result = _add_config_result(
                dataset_result,
                summary_rows,
                resmem_evaluations,
                feature_source="ensemble_rank:" + "+".join(clip_models),
                method="ensemble_rank_mean",
                reducer="n/a",
                components=None,
                evaluations=evaluations,
            )
            print(
                f"[memcat-study] done rank-mean ensemble={' + '.join(clip_models)} "
                f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
            )

            print(f"[memcat-study] evaluating rank-weighted ensemble={' + '.join(clip_models)}")
            evaluations = evaluate_rank_weighted_ensemble_on_splits(
                member_evaluations=ensemble_members,
                scores=score_array,
                splits=splits,
                step=ensemble_weight_step,
            )
            config_result = _add_config_result(
                dataset_result,
                summary_rows,
                resmem_evaluations,
                feature_source="ensemble_rank_weighted:" + "+".join(clip_models),
                method="ensemble_rank_weighted",
                reducer="n/a",
                components=None,
                evaluations=evaluations,
            )
            print(
                f"[memcat-study] done rank-weighted ensemble={' + '.join(clip_models)} "
                f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
            )

            print(f"[memcat-study] evaluating stacked-ridge ensemble={' + '.join(clip_models)}")
            evaluations = evaluate_stacked_ridge_ensemble_on_splits(
                member_evaluations=ensemble_members,
                scores=score_array,
                splits=splits,
            )
            config_result = _add_config_result(
                dataset_result,
                summary_rows,
                resmem_evaluations,
                feature_source="ensemble_stacked:" + "+".join(clip_models),
                method="ensemble_stacked_ridge",
                reducer="n/a",
                components=None,
                evaluations=evaluations,
            )
            print(
                f"[memcat-study] done stacked-ridge ensemble={' + '.join(clip_models)} "
                f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
            )

    return {
        "metadata": {
            "memcat_root": str(Path(memcat_root).expanduser().resolve()),
            "score_column": dataset.score_column,
            "clip_models": clip_models,
            "clip_pretrained": clip_pretrained,
            "clip_tta_mode": normalized_tta_mode,
            "clip_tta_resize": clip_tta_resize,
            "pca_dims": ["none" if dim is None else int(dim) for dim in pca_dims],
            "pls_dims": pls_dims,
            "num_splits": num_splits,
            "batch_size": batch_size,
            "device": device,
            "ensemble_weight_step": ensemble_weight_step,
            "checkpoint_path": str(checkpoint_path),
        },
        "studies": [dataset_result],
        "summary_rows": summary_rows,
    }
