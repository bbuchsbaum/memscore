from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

from .clip_regression import ClipEmbedder, normalize_tta_mode
from .data import Record
from .metrics import compute_metrics
from .resmem_baseline import ResMemScorer, ensure_resmem_checkpoint

PathLike = Union[Path, str]


@dataclass(frozen=True)
class FigrimDataset:
    name: str
    score_type: str
    records: list[Record]
    categories: list[str]


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    seed: int

    def as_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "sizes": {
                "train": int(self.train.size),
                "val": int(self.val.size),
                "test": int(self.test.size),
            },
        }


def _normalize_protocol(protocol: str) -> str:
    normalized = protocol.strip().lower().replace("-", "_")
    aliases = {
        "across": "across_category",
        "within": "within_category",
    }
    return aliases.get(normalized, normalized)


def _score_from_row(row: dict[str, str], score_type: str) -> float:
    hits = float(row["hits"])
    misses = float(row["misses"])
    false_alarms = float(row["false_alarms"])
    correct_rejections = float(row["correct_rejections"])

    hit_rate = hits / (hits + misses)
    if score_type == "hit_rate":
        return hit_rate

    if score_type == "corrected_hit_rate":
        false_alarm_rate = false_alarms / (false_alarms + correct_rejections)
        corrected = 0.5 + 0.5 * (hit_rate - false_alarm_rate)
        return float(np.clip(corrected, 0.0, 1.0))

    raise KeyError(f"Unsupported FIGRIM score type {score_type!r}")


def load_figrim_dataset(root: PathLike, protocol: str, score_type: str = "corrected_hit_rate") -> FigrimDataset:
    dataset_root = Path(root).expanduser().resolve()
    protocol_name = _normalize_protocol(protocol)
    source_csv = dataset_root / f"{protocol_name}.csv"
    image_root = dataset_root / "images" / "SCENES_TARGETS"
    if not source_csv.exists():
        raise FileNotFoundError(f"Missing FIGRIM metadata CSV: {source_csv}")
    if not image_root.exists():
        raise FileNotFoundError(f"Missing FIGRIM image directory: {image_root}")

    records: list[Record] = []
    categories: list[str] = []
    with source_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header row in {source_csv}")
        for row in reader:
            category = row["scene category"]
            image_path = (image_root / category / row["filename"]).resolve()
            records.append(
                Record(
                    image_path=image_path,
                    score=_score_from_row(row, score_type=score_type),
                    identifier=row["image number"],
                )
            )
            categories.append(category)

    return FigrimDataset(
        name=protocol_name,
        score_type=score_type,
        records=records,
        categories=categories,
    )


def generate_repeated_splits(
    categories: list[str],
    num_splits: int,
    random_state: int = 0,
    train_size: float = 0.7,
    val_size: float = 0.1,
    test_size: float = 0.2,
) -> list[SplitIndices]:
    if num_splits <= 0:
        raise ValueError("num_splits must be positive")

    expected_total = train_size + val_size + test_size
    if not np.isclose(expected_total, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    indices = np.arange(len(categories))
    label_array = np.asarray(categories)
    val_fraction_within_trainval = val_size / (train_size + val_size)

    splits: list[SplitIndices] = []
    for split_idx in range(num_splits):
        seed = random_state + split_idx
        trainval_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=label_array,
        )
        train_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=val_fraction_within_trainval,
            random_state=seed,
            stratify=label_array[trainval_idx],
        )
        splits.append(
            SplitIndices(
                train=np.sort(train_idx),
                val=np.sort(val_idx),
                test=np.sort(test_idx),
                seed=seed,
            )
        )
    return splits


def build_ridge_pipeline(pca_dim: Optional[int]) -> Pipeline:
    steps = [("scale", StandardScaler())]
    if pca_dim is not None:
        steps.append(("pca", PCA(n_components=pca_dim, svd_solver="full")))
    steps.append(("model", RidgeCV(alphas=np.logspace(-4, 4, 17))))
    return Pipeline(steps)


def evaluate_ridge_on_splits(
    features: np.ndarray,
    scores: np.ndarray,
    splits: list[SplitIndices],
    pca_dim: Optional[int] = None,
    keep_predictions: bool = False,
) -> list[dict[str, object]]:
    evaluations: list[dict[str, object]] = []
    for split in splits:
        train_X = features[split.train]
        val_X = features[split.val]
        test_X = features[split.test]

        train_y = scores[split.train]
        val_y = scores[split.val]
        test_y = scores[split.test]

        val_model = build_ridge_pipeline(pca_dim)
        val_model.fit(train_X, train_y)
        val_pred = val_model.predict(val_X).astype(np.float32)
        val_metrics = compute_metrics(val_y, val_pred)

        final_model = clone(val_model)
        trainval_X = np.concatenate([train_X, val_X], axis=0)
        trainval_y = np.concatenate([train_y, val_y], axis=0)
        final_model.fit(trainval_X, trainval_y)
        test_pred = final_model.predict(test_X).astype(np.float32)
        test_metrics = compute_metrics(test_y, test_pred)

        alpha = float(final_model.named_steps["model"].alpha_)
        result = {
            "seed": split.seed,
            "val_metrics": val_metrics.as_dict(),
            "test_metrics": test_metrics.as_dict(),
            "ridge_alpha": alpha,
        }
        if keep_predictions:
            result["_val_predictions"] = val_pred
            result["_test_predictions"] = test_pred
        evaluations.append(result)
    return evaluations


def evaluate_resmem_on_splits(
    predictions: np.ndarray,
    scores: np.ndarray,
    splits: list[SplitIndices],
) -> list[dict[str, object]]:
    evaluations: list[dict[str, object]] = []
    for split in splits:
        test_metrics = compute_metrics(scores[split.test], predictions[split.test])
        evaluations.append({"seed": split.seed, "test_metrics": test_metrics.as_dict()})
    return evaluations


def evaluate_pls_on_splits(
    features: np.ndarray,
    scores: np.ndarray,
    splits: list[SplitIndices],
    n_components: int,
    keep_predictions: bool = False,
) -> list[dict[str, object]]:
    evaluations: list[dict[str, object]] = []
    for split in splits:
        train_X = features[split.train]
        val_X = features[split.val]
        test_X = features[split.test]

        train_y = scores[split.train]
        val_y = scores[split.val]
        test_y = scores[split.test]

        max_components = int(min(train_X.shape[0] - 1, train_X.shape[1]))
        actual_components = int(min(n_components, max_components))
        if actual_components <= 0:
            raise ValueError(f"PLS requires at least one component, got {actual_components}")

        val_model = PLSRegression(n_components=actual_components, scale=True)
        val_model.fit(train_X, train_y)
        val_pred = np.asarray(val_model.predict(val_X)).reshape(-1).astype(np.float32)
        val_metrics = compute_metrics(val_y, val_pred)

        final_model = PLSRegression(n_components=actual_components, scale=True)
        trainval_X = np.concatenate([train_X, val_X], axis=0)
        trainval_y = np.concatenate([train_y, val_y], axis=0)
        final_model.fit(trainval_X, trainval_y)
        test_pred = np.asarray(final_model.predict(test_X)).reshape(-1).astype(np.float32)
        test_metrics = compute_metrics(test_y, test_pred)

        result = {
            "seed": split.seed,
            "val_metrics": val_metrics.as_dict(),
            "test_metrics": test_metrics.as_dict(),
            "pls_components": actual_components,
        }
        if keep_predictions:
            result["_val_predictions"] = val_pred
            result["_test_predictions"] = test_pred
        evaluations.append(result)
    return evaluations


def evaluate_mean_ensemble_on_splits(
    member_evaluations: list[list[dict[str, object]]],
    scores: np.ndarray,
    splits: list[SplitIndices],
) -> list[dict[str, object]]:
    if not member_evaluations:
        raise ValueError("Ensemble requires at least one member")

    evaluations: list[dict[str, object]] = []
    for split_idx, split in enumerate(splits):
        val_predictions = np.mean(
            [np.asarray(member[split_idx]["_val_predictions"], dtype=np.float32) for member in member_evaluations],
            axis=0,
        )
        test_predictions = np.mean(
            [np.asarray(member[split_idx]["_test_predictions"], dtype=np.float32) for member in member_evaluations],
            axis=0,
        )
        result = {
            "seed": split.seed,
            "val_metrics": compute_metrics(scores[split.val], val_predictions).as_dict(),
            "test_metrics": compute_metrics(scores[split.test], test_predictions).as_dict(),
        }
        evaluations.append(result)
    return evaluations


def _normalize_rank_predictions(predictions: np.ndarray) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D prediction vector, got shape {values.shape}")
    if values.size <= 1:
        return np.zeros_like(values, dtype=np.float32)
    ranks = rankdata(values, method="average").astype(np.float32)
    return (ranks - 1.0) / float(values.size - 1)


def evaluate_rank_mean_ensemble_on_splits(
    member_evaluations: list[list[dict[str, object]]],
    scores: np.ndarray,
    splits: list[SplitIndices],
) -> list[dict[str, object]]:
    if not member_evaluations:
        raise ValueError("Rank ensemble requires at least one member")

    evaluations: list[dict[str, object]] = []
    for split_idx, split in enumerate(splits):
        val_predictions = np.mean(
            [
                _normalize_rank_predictions(
                    np.asarray(member[split_idx]["_val_predictions"], dtype=np.float32)
                )
                for member in member_evaluations
            ],
            axis=0,
        )
        test_predictions = np.mean(
            [
                _normalize_rank_predictions(
                    np.asarray(member[split_idx]["_test_predictions"], dtype=np.float32)
                )
                for member in member_evaluations
            ],
            axis=0,
        )
        evaluations.append(
            {
                "seed": split.seed,
                "val_metrics": compute_metrics(scores[split.val], val_predictions).as_dict(),
                "test_metrics": compute_metrics(scores[split.test], test_predictions).as_dict(),
            }
        )
    return evaluations


def summarize_split_metrics(
    evaluations: list[dict[str, object]],
    metric_key: str = "test_metrics",
) -> dict[str, float]:
    spearman = np.asarray([item[metric_key]["spearman"] for item in evaluations], dtype=np.float64)
    mse = np.asarray([item[metric_key]["mse"] for item in evaluations], dtype=np.float64)
    return {
        "mean_spearman": float(spearman.mean()),
        "std_spearman": float(spearman.std(ddof=0)),
        "mean_mse": float(mse.mean()),
        "std_mse": float(mse.std(ddof=0)),
    }


def compare_against_resmem(
    resmem_evaluations: list[dict[str, object]],
    model_evaluations: list[dict[str, object]],
) -> dict[str, float]:
    resmem_spearman = np.asarray([item["test_metrics"]["spearman"] for item in resmem_evaluations], dtype=np.float64)
    model_spearman = np.asarray([item["test_metrics"]["spearman"] for item in model_evaluations], dtype=np.float64)
    resmem_mse = np.asarray([item["test_metrics"]["mse"] for item in resmem_evaluations], dtype=np.float64)
    model_mse = np.asarray([item["test_metrics"]["mse"] for item in model_evaluations], dtype=np.float64)

    delta_spearman = model_spearman - resmem_spearman
    delta_mse = model_mse - resmem_mse
    return {
        "mean_delta_spearman": float(delta_spearman.mean()),
        "mean_delta_mse": float(delta_mse.mean()),
        "spearman_win_rate": float(np.mean(delta_spearman > 0.0)),
        "mse_win_rate": float(np.mean(delta_mse < 0.0)),
        "min_delta_spearman": float(delta_spearman.min()),
        "max_delta_spearman": float(delta_spearman.max()),
    }


def parse_pca_dims(values: list[str]) -> list[Optional[int]]:
    dims: list[Optional[int]] = []
    for value in values:
        lowered = value.strip().lower()
        if lowered in {"none", "full"}:
            dims.append(None)
            continue
        dim = int(lowered)
        if dim <= 0:
            raise ValueError(f"PCA dimensions must be positive, got {value!r}")
        dims.append(dim)
    return dims


def parse_positive_dims(values: list[str]) -> list[int]:
    dims: list[int] = []
    for value in values:
        dim = int(value.strip())
        if dim <= 0:
            raise ValueError(f"Dimensions must be positive, got {value!r}")
        dims.append(dim)
    return dims


def _paths_key(records: list[Record]) -> tuple[str, ...]:
    return tuple(str(record.image_path) for record in records)


def _strip_prediction_fields(evaluations: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{key: value for key, value in item.items() if not key.startswith("_")} for item in evaluations]


def _load_prediction_cache(cache_path: Path, records: list[Record]) -> Optional[np.ndarray]:
    if not cache_path.exists():
        return None
    payload = np.load(cache_path, allow_pickle=True)
    cached_paths = payload["paths"].tolist()
    expected_paths = [str(record.image_path) for record in records]
    if cached_paths != expected_paths:
        return None
    return payload["predictions"].astype(np.float32)


def _write_prediction_cache(cache_path: Path, records: list[Record], predictions: np.ndarray) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        predictions=np.asarray(predictions, dtype=np.float32),
        paths=np.asarray([str(record.image_path) for record in records], dtype=object),
    )


def _simplex_weight_grid(num_members: int, step: float) -> list[np.ndarray]:
    if num_members < 2:
        raise ValueError("Weighted ensembles require at least two members")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError(f"ensemble step {step!r} must evenly divide 1.0")

    def _compositions(total: int, parts: int):
        if parts == 1:
            yield (total,)
            return
        for head in range(total + 1):
            for tail in _compositions(total - head, parts - 1):
                yield (head,) + tail

    weights = []
    for combo in _compositions(units, num_members):
        weights.append(np.asarray(combo, dtype=np.float32) / float(units))
    return weights


def _add_config_result(
    dataset_result: dict[str, object],
    summary_rows: list[dict[str, object]],
    resmem_evaluations: list[dict[str, object]],
    *,
    feature_source: str,
    method: str,
    reducer: str,
    components: Optional[int],
    evaluations: list[dict[str, object]],
) -> dict[str, object]:
    comparison = compare_against_resmem(resmem_evaluations, evaluations)
    config_result = {
        "feature_source": feature_source,
        "method": method,
        "reducer": reducer,
        "components": components,
        "summary": summarize_split_metrics(evaluations),
        "comparison_vs_resmem": comparison,
        "split_results": _strip_prediction_fields(evaluations),
    }
    dataset_result["model_configs"].append(config_result)
    summary_rows.append(
        {
            "protocol": dataset_result["protocol"],
            "score_type": dataset_result["score_type"],
            "clip_tta_mode": dataset_result["clip_tta_mode"],
            "clip_tta_resize": dataset_result["clip_tta_resize"],
            "feature_source": feature_source,
            "method": method,
            "reducer": reducer,
            "components": "none" if components is None else int(components),
            "resmem_mean_spearman": dataset_result["resmem"]["summary"]["mean_spearman"],
            "model_mean_spearman": config_result["summary"]["mean_spearman"],
            "mean_delta_spearman": comparison["mean_delta_spearman"],
            "spearman_win_rate": comparison["spearman_win_rate"],
            "resmem_mean_mse": dataset_result["resmem"]["summary"]["mean_mse"],
            "model_mean_mse": config_result["summary"]["mean_mse"],
            "mean_delta_mse": comparison["mean_delta_mse"],
            "mse_win_rate": comparison["mse_win_rate"],
        }
    )
    return config_result


def evaluate_weighted_ensemble_on_splits(
    member_evaluations: list[list[dict[str, object]]],
    scores: np.ndarray,
    splits: list[SplitIndices],
    step: float = 0.05,
) -> list[dict[str, object]]:
    if not member_evaluations:
        raise ValueError("Weighted ensemble requires at least one member")
    weights_grid = _simplex_weight_grid(len(member_evaluations), step=step)

    evaluations: list[dict[str, object]] = []
    for split_idx, split in enumerate(splits):
        val_predictions = np.asarray(
            [member[split_idx]["_val_predictions"] for member in member_evaluations],
            dtype=np.float32,
        )
        test_predictions = np.asarray(
            [member[split_idx]["_test_predictions"] for member in member_evaluations],
            dtype=np.float32,
        )

        best_weights = None
        best_val_metrics = None
        for weights in weights_grid:
            blended_val = np.sum(val_predictions * weights[:, None], axis=0)
            val_metrics = compute_metrics(scores[split.val], blended_val)
            if best_val_metrics is None:
                best_weights = weights
                best_val_metrics = val_metrics
                continue
            if val_metrics.spearman > best_val_metrics.spearman or (
                np.isclose(val_metrics.spearman, best_val_metrics.spearman) and val_metrics.mse < best_val_metrics.mse
            ):
                best_weights = weights
                best_val_metrics = val_metrics

        assert best_weights is not None
        assert best_val_metrics is not None
        blended_test = np.sum(test_predictions * best_weights[:, None], axis=0)
        evaluations.append(
            {
                "seed": split.seed,
                "val_metrics": best_val_metrics.as_dict(),
                "test_metrics": compute_metrics(scores[split.test], blended_test).as_dict(),
                "ensemble_weights": best_weights.tolist(),
            }
        )
    return evaluations


def evaluate_rank_weighted_ensemble_on_splits(
    member_evaluations: list[list[dict[str, object]]],
    scores: np.ndarray,
    splits: list[SplitIndices],
    step: float = 0.05,
) -> list[dict[str, object]]:
    if not member_evaluations:
        raise ValueError("Weighted rank ensemble requires at least one member")
    weights_grid = _simplex_weight_grid(len(member_evaluations), step=step)

    evaluations: list[dict[str, object]] = []
    for split_idx, split in enumerate(splits):
        val_predictions = np.asarray(
            [
                _normalize_rank_predictions(np.asarray(member[split_idx]["_val_predictions"], dtype=np.float32))
                for member in member_evaluations
            ],
            dtype=np.float32,
        )
        test_predictions = np.asarray(
            [
                _normalize_rank_predictions(np.asarray(member[split_idx]["_test_predictions"], dtype=np.float32))
                for member in member_evaluations
            ],
            dtype=np.float32,
        )

        best_weights = None
        best_val_metrics = None
        for weights in weights_grid:
            blended_val = np.sum(val_predictions * weights[:, None], axis=0)
            val_metrics = compute_metrics(scores[split.val], blended_val)
            if best_val_metrics is None:
                best_weights = weights
                best_val_metrics = val_metrics
                continue
            if val_metrics.spearman > best_val_metrics.spearman or (
                np.isclose(val_metrics.spearman, best_val_metrics.spearman) and val_metrics.mse < best_val_metrics.mse
            ):
                best_weights = weights
                best_val_metrics = val_metrics

        assert best_weights is not None
        assert best_val_metrics is not None
        blended_test = np.sum(test_predictions * best_weights[:, None], axis=0)
        evaluations.append(
            {
                "seed": split.seed,
                "val_metrics": best_val_metrics.as_dict(),
                "test_metrics": compute_metrics(scores[split.test], blended_test).as_dict(),
                "ensemble_weights": best_weights.tolist(),
            }
        )
    return evaluations


def evaluate_stacked_ridge_ensemble_on_splits(
    member_evaluations: list[list[dict[str, object]]],
    scores: np.ndarray,
    splits: list[SplitIndices],
) -> list[dict[str, object]]:
    if not member_evaluations:
        raise ValueError("Stacked ensemble requires at least one member")

    evaluations: list[dict[str, object]] = []
    for split_idx, split in enumerate(splits):
        val_predictions = np.stack(
            [np.asarray(member[split_idx]["_val_predictions"], dtype=np.float32) for member in member_evaluations],
            axis=1,
        )
        test_predictions = np.stack(
            [np.asarray(member[split_idx]["_test_predictions"], dtype=np.float32) for member in member_evaluations],
            axis=1,
        )
        meta_model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-4, 4, 17))),
            ]
        )
        meta_model.fit(val_predictions, scores[split.val])
        blended_val = meta_model.predict(val_predictions).astype(np.float32)
        blended_test = meta_model.predict(test_predictions).astype(np.float32)
        evaluations.append(
            {
                "seed": split.seed,
                "val_metrics": compute_metrics(scores[split.val], blended_val).as_dict(),
                "test_metrics": compute_metrics(scores[split.test], blended_test).as_dict(),
                "ridge_alpha": float(meta_model.named_steps["model"].alpha_),
            }
        )
    return evaluations


def run_figrim_study(
    figrim_root: PathLike,
    protocols: list[str],
    score_types: list[str],
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
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    normalized_tta_mode = normalize_tta_mode(clip_tta_mode)

    checkpoint_path = (
        Path(resmem_checkpoint).expanduser().resolve()
        if resmem_checkpoint
        else ensure_resmem_checkpoint(cache_root / "resmem")
    )
    resmem_scorer = ResMemScorer(checkpoint_path=checkpoint_path, batch_size=batch_size, device=device)

    prediction_cache: dict[tuple[str, ...], np.ndarray] = {}
    embedding_cache: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}

    studies: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for protocol in protocols:
        for score_type in score_types:
            dataset = load_figrim_dataset(figrim_root, protocol=protocol, score_type=score_type)
            print(
                f"[figrim-study] protocol={dataset.name} score_type={dataset.score_type} "
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

            if paths_key not in prediction_cache:
                prediction_cache_path = cache_root / "resmem" / "figrim_targets_predictions.npz"
                cached_predictions = _load_prediction_cache(prediction_cache_path, dataset.records)
                if cached_predictions is None:
                    print("[figrim-study] scoring ResMem on full FIGRIM target set")
                    cached_predictions = resmem_scorer.predict(dataset.records)
                    _write_prediction_cache(prediction_cache_path, dataset.records, cached_predictions)
                else:
                    print("[figrim-study] loaded cached ResMem predictions for FIGRIM target set")
                prediction_cache[paths_key] = cached_predictions
            resmem_evaluations = evaluate_resmem_on_splits(prediction_cache[paths_key], score_array, splits)

            dataset_result = {
                "protocol": dataset.name,
                "score_type": dataset.score_type,
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

            single_model_no_reduction: dict[str, list[dict[str, object]]] = {}
            feature_map: dict[str, np.ndarray] = {}
            for clip_model in clip_models:
                embedding_key = (
                    f"{clip_model}::{clip_pretrained}::{normalized_tta_mode}::{clip_tta_resize}",
                    paths_key,
                )
                if embedding_key not in embedding_cache:
                    print(f"[figrim-study] extracting CLIP embeddings model={clip_model} pretrained={clip_pretrained}")
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
                        split_name="figrim_targets_all",
                        cache_dir=cache_root / "clip",
                    )
                    embedding_cache[embedding_key] = bundle.features

                features = embedding_cache[embedding_key]
                feature_map[clip_model] = features
                for pca_dim in pca_dims:
                    print(
                        f"[figrim-study] evaluating model={clip_model} ridge "
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
                        f"[figrim-study] done model={clip_model} ridge pca_dim={'none' if pca_dim is None else pca_dim} "
                        f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                        f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
                    )

                for pls_dim in pls_dims:
                    print(f"[figrim-study] evaluating model={clip_model} pls components={pls_dim}")
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
                        f"[figrim-study] done model={clip_model} pls components={pls_dim} "
                        f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                        f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
                    )

            if len(clip_models) > 1:
                fused_name = "+".join(clip_models)
                fused_features = np.concatenate([feature_map[model_name] for model_name in clip_models], axis=1)
                for pca_dim in pca_dims:
                    print(
                        f"[figrim-study] evaluating fusion={fused_name} ridge "
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
                        f"[figrim-study] done fusion={fused_name} ridge pca_dim={'none' if pca_dim is None else pca_dim} "
                        f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                        f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
                    )

                for pls_dim in pls_dims:
                    print(f"[figrim-study] evaluating fusion={fused_name} pls components={pls_dim}")
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
                        f"[figrim-study] done fusion={fused_name} pls components={pls_dim} "
                        f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                        f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
                    )

                ensemble_members = [single_model_no_reduction[model_name] for model_name in clip_models if model_name in single_model_no_reduction]
                if len(ensemble_members) >= 2:
                    print(f"[figrim-study] evaluating mean ensemble={' + '.join(clip_models)}")
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
                        f"[figrim-study] done mean ensemble={' + '.join(clip_models)} "
                        f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                        f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
                    )
                    print(f"[figrim-study] evaluating val-weighted ensemble={' + '.join(clip_models)}")
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
                        f"[figrim-study] done val-weighted ensemble={' + '.join(clip_models)} "
                        f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                        f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
                    )

                    print(f"[figrim-study] evaluating rank-mean ensemble={' + '.join(clip_models)}")
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
                        f"[figrim-study] done rank-mean ensemble={' + '.join(clip_models)} "
                        f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                        f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
                    )

                    print(f"[figrim-study] evaluating rank-weighted ensemble={' + '.join(clip_models)}")
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
                        f"[figrim-study] done rank-weighted ensemble={' + '.join(clip_models)} "
                        f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                        f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
                    )

                    print(f"[figrim-study] evaluating stacked-ridge ensemble={' + '.join(clip_models)}")
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
                        f"[figrim-study] done stacked-ridge ensemble={' + '.join(clip_models)} "
                        f"mean_delta_spearman={config_result['comparison_vs_resmem']['mean_delta_spearman']:.4f} "
                        f"win_rate={config_result['comparison_vs_resmem']['spearman_win_rate']:.2f}"
                    )
            studies.append(dataset_result)

    return {
        "metadata": {
            "figrim_root": str(Path(figrim_root).expanduser().resolve()),
            "protocols": [_normalize_protocol(protocol) for protocol in protocols],
            "score_types": score_types,
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
        "studies": studies,
        "summary_rows": summary_rows,
    }


def write_study_outputs(study: dict[str, object], output_json: PathLike, output_csv: PathLike) -> None:
    output_json_path = Path(output_json).expanduser().resolve()
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(study, indent=2), encoding="utf-8")

    output_csv_path = Path(output_csv).expanduser().resolve()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = study["summary_rows"]
    if not rows:
        raise ValueError("Study did not produce any summary rows")
    fieldnames = list(rows[0].keys())
    with output_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
