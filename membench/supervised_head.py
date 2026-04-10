from __future__ import annotations

import csv
import copy
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .clip_regression import ClipEmbedder, normalize_tta_mode
from .data import DatasetSplits, Record, load_manifest
from .metrics import compute_metrics

PathLike = Union[Path, str]


@dataclass(frozen=True)
class FixedSplitDatasetSpec:
    name: str
    manifest_path: Path
    root: Optional[Path]


@dataclass(frozen=True)
class EmbeddedDataset:
    name: str
    train_features: np.ndarray
    train_scores: np.ndarray
    val_features: np.ndarray
    val_scores: np.ndarray
    test_features: np.ndarray
    test_scores: np.ndarray
    mean: float
    std: float
    sizes: dict[str, int]

    def normalize(self, scores: np.ndarray) -> np.ndarray:
        return (scores - self.mean) / self.std

    def denormalize(self, scores: np.ndarray) -> np.ndarray:
        return (scores * self.std) + self.mean


class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).squeeze(-1)


class DatasetSpecificMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, num_datasets: int) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleList(nn.Linear(hidden_dim, 1) for _ in range(num_datasets))

    def forward(self, inputs: torch.Tensor, dataset_indices: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(inputs)
        outputs = hidden.new_empty((hidden.shape[0],), dtype=hidden.dtype)
        for head_index, head in enumerate(self.heads):
            mask = dataset_indices == head_index
            if torch.any(mask):
                outputs[mask] = head(hidden[mask]).squeeze(-1)
        return outputs


def _resolve_relative(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base / path).resolve()


def load_fixed_split_dataset_specs(config_path: PathLike) -> list[FixedSplitDatasetSpec]:
    config_file = Path(config_path).expanduser().resolve()
    specs: list[FixedSplitDatasetSpec] = []
    with config_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset", "manifest"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Dataset config must contain columns {sorted(required)}")
        for row in reader:
            name = row["dataset"].strip()
            if not name:
                raise ValueError("Dataset config contains an empty dataset name")
            manifest_path = _resolve_relative(config_file.parent, row["manifest"].strip())
            root_text = (row.get("root") or "").strip()
            specs.append(
                FixedSplitDatasetSpec(
                    name=name,
                    manifest_path=manifest_path,
                    root=_resolve_relative(config_file.parent, root_text) if root_text else None,
                )
            )
    if not specs:
        raise ValueError(f"No datasets found in {config_file}")
    return specs


def _embed_split(
    embedder: ClipEmbedder,
    records: list[Record],
    *,
    dataset_name: str,
    split_name: str,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    bundle = embedder.embed_split(records, split_name=f"{dataset_name}_{split_name}", cache_dir=cache_dir)
    return bundle.features.astype(np.float32), bundle.scores.astype(np.float32)


def load_embedded_datasets(
    specs: list[FixedSplitDatasetSpec],
    *,
    clip_model: str,
    clip_pretrained: str,
    clip_tta_mode: str,
    clip_tta_resize: Optional[int],
    batch_size: int,
    device: str,
    cache_dir: PathLike,
) -> list[EmbeddedDataset]:
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    embedder = ClipEmbedder(
        model_name=clip_model,
        pretrained=clip_pretrained,
        batch_size=batch_size,
        device=device,
        tta_mode=clip_tta_mode,
        tta_resize=clip_tta_resize,
    )

    datasets: list[EmbeddedDataset] = []
    for spec in specs:
        splits: DatasetSplits = load_manifest(spec.manifest_path, root=spec.root)
        train_features, train_scores = _embed_split(
            embedder,
            splits.train,
            dataset_name=spec.name,
            split_name="train",
            cache_dir=cache_root / "clip",
        )
        val_features, val_scores = _embed_split(
            embedder,
            splits.val,
            dataset_name=spec.name,
            split_name="val",
            cache_dir=cache_root / "clip",
        )
        test_features, test_scores = _embed_split(
            embedder,
            splits.test,
            dataset_name=spec.name,
            split_name="test",
            cache_dir=cache_root / "clip",
        )
        mean = float(np.mean(train_scores))
        std = float(np.std(train_scores))
        if std < 1e-8:
            std = 1.0
        datasets.append(
            EmbeddedDataset(
                name=spec.name,
                train_features=train_features,
                train_scores=train_scores,
                val_features=val_features,
                val_scores=val_scores,
                test_features=test_features,
                test_scores=test_scores,
                mean=mean,
                std=std,
                sizes={"train": len(train_scores), "val": len(val_scores), "test": len(test_scores)},
            )
        )
    return datasets


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_balanced_epoch_arrays(
    datasets: list[EmbeddedDataset],
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    max_train = max(dataset.train_features.shape[0] for dataset in datasets)
    feature_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    for dataset in datasets:
        train_size = dataset.train_features.shape[0]
        indices = rng.choice(train_size, size=max_train, replace=train_size < max_train)
        feature_batches.append(dataset.train_features[indices])
        target_batches.append(dataset.normalize(dataset.train_scores[indices]))
    return (
        np.concatenate(feature_batches, axis=0).astype(np.float32),
        np.concatenate(target_batches, axis=0).astype(np.float32),
    )


def _make_balanced_epoch_arrays_with_dataset_ids(
    datasets: list[EmbeddedDataset],
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_train = max(dataset.train_features.shape[0] for dataset in datasets)
    feature_batches: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    dataset_batches: list[np.ndarray] = []
    for dataset_index, dataset in enumerate(datasets):
        train_size = dataset.train_features.shape[0]
        indices = rng.choice(train_size, size=max_train, replace=train_size < max_train)
        feature_batches.append(dataset.train_features[indices])
        target_batches.append(dataset.normalize(dataset.train_scores[indices]))
        dataset_batches.append(np.full(max_train, dataset_index, dtype=np.int64))
    return (
        np.concatenate(feature_batches, axis=0).astype(np.float32),
        np.concatenate(target_batches, axis=0).astype(np.float32),
        np.concatenate(dataset_batches, axis=0),
    )


def _pairwise_rank_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if predictions.numel() < 2:
        return predictions.new_tensor(0.0)
    target_diff = targets[:, None] - targets[None, :]
    sign = torch.sign(target_diff)
    upper = torch.triu(torch.ones_like(sign, dtype=torch.bool), diagonal=1)
    mask = upper & (sign != 0)
    if not torch.any(mask):
        return predictions.new_tensor(0.0)
    pred_diff = predictions[:, None] - predictions[None, :]
    return F.softplus(-sign[mask] * pred_diff[mask]).mean()


def _evaluate_model(
    model: EmbeddingMLP,
    datasets: list[EmbeddedDataset],
    *,
    split_name: str,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    by_dataset: dict[str, object] = {}
    spearmans: list[float] = []
    mses: list[float] = []
    with torch.inference_mode():
        for dataset in datasets:
            if split_name == "val":
                features = dataset.val_features
                truth = dataset.val_scores
            elif split_name == "test":
                features = dataset.test_features
                truth = dataset.test_scores
            else:
                raise ValueError(f"Unsupported split {split_name!r}")
            tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            predictions = model(tensor).detach().cpu().numpy().astype(np.float32)
            restored = dataset.denormalize(predictions)
            metrics = compute_metrics(truth, restored)
            by_dataset[dataset.name] = {
                "metrics": metrics.as_dict(),
                "size": int(truth.shape[0]),
            }
            spearmans.append(metrics.spearman)
            mses.append(metrics.mse)
    return {
        "datasets": by_dataset,
        "summary": {
            "mean_spearman": float(np.mean(spearmans)),
            "mean_mse": float(np.mean(mses)),
        },
    }


def _evaluate_dataset_specific_model(
    model: DatasetSpecificMLP,
    datasets: list[EmbeddedDataset],
    *,
    split_name: str,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    by_dataset: dict[str, object] = {}
    spearmans: list[float] = []
    mses: list[float] = []
    with torch.inference_mode():
        for dataset_index, dataset in enumerate(datasets):
            if split_name == "val":
                features = dataset.val_features
                truth = dataset.val_scores
            elif split_name == "test":
                features = dataset.test_features
                truth = dataset.test_scores
            else:
                raise ValueError(f"Unsupported split {split_name!r}")
            tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
            ids = torch.full((features.shape[0],), dataset_index, dtype=torch.long, device=device)
            predictions = model(tensor, ids).detach().cpu().numpy().astype(np.float32)
            restored = dataset.denormalize(predictions)
            metrics = compute_metrics(truth, restored)
            by_dataset[dataset.name] = {
                "metrics": metrics.as_dict(),
                "size": int(truth.shape[0]),
            }
            spearmans.append(metrics.spearman)
            mses.append(metrics.mse)
    return {
        "datasets": by_dataset,
        "summary": {
            "mean_spearman": float(np.mean(spearmans)),
            "mean_mse": float(np.mean(mses)),
        },
    }


def _train_embedding_head(
    datasets: list[EmbeddedDataset],
    *,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    epochs: int = 25,
    patience: int = 6,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    head_batch_size: int = 256,
    rank_loss_weight: float = 0.2,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[EmbeddingMLP, list[dict[str, object]], int, dict[str, object]]:
    _set_seed(seed)
    if not datasets:
        raise ValueError("No datasets available for pooled training")

    input_dim = int(datasets[0].train_features.shape[1])
    if any(dataset.train_features.shape[1] != input_dim for dataset in datasets):
        raise ValueError("All embedded datasets must have the same feature dimension")

    torch_device = torch.device(device)
    model = EmbeddingMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    best_key = (-math.inf, math.inf)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        train_features, train_targets = _make_balanced_epoch_arrays(datasets, rng=rng)
        loader = DataLoader(
            TensorDataset(
                torch.as_tensor(train_features, dtype=torch.float32),
                torch.as_tensor(train_targets, dtype=torch.float32),
            ),
            batch_size=head_batch_size,
            shuffle=True,
        )

        model.train()
        batch_losses: list[float] = []
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(torch_device)
            batch_targets = batch_targets.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features)
            mse_loss = F.mse_loss(predictions, batch_targets)
            rank_loss = _pairwise_rank_loss(predictions, batch_targets)
            loss = mse_loss + (rank_loss_weight * rank_loss)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        val_report = _evaluate_model(model, datasets, split_name="val", device=torch_device)
        val_key = (
            float(val_report["summary"]["mean_spearman"]),
            float(val_report["summary"]["mean_mse"]),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)) if batch_losses else 0.0,
                "val_summary": val_report["summary"],
            }
        )

        if val_key[0] > best_key[0] or (val_key[0] == best_key[0] and val_key[1] < best_key[1]):
            best_key = val_key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    final_val = _evaluate_model(model, datasets, split_name="val", device=torch_device)
    return model, history, best_epoch, final_val


def _train_dataset_specific_head(
    datasets: list[EmbeddedDataset],
    *,
    hidden_dim: int = 256,
    dropout: float = 0.1,
    epochs: int = 25,
    patience: int = 6,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    head_batch_size: int = 256,
    rank_loss_weight: float = 0.2,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[DatasetSpecificMLP, list[dict[str, object]], int, dict[str, object]]:
    _set_seed(seed)
    if not datasets:
        raise ValueError("No datasets available for multi-head training")

    input_dim = int(datasets[0].train_features.shape[1])
    if any(dataset.train_features.shape[1] != input_dim for dataset in datasets):
        raise ValueError("All embedded datasets must have the same feature dimension")

    torch_device = torch.device(device)
    model = DatasetSpecificMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        num_datasets=len(datasets),
    ).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    best_key = (-math.inf, math.inf)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        train_features, train_targets, train_dataset_ids = _make_balanced_epoch_arrays_with_dataset_ids(
            datasets,
            rng=rng,
        )
        loader = DataLoader(
            TensorDataset(
                torch.as_tensor(train_features, dtype=torch.float32),
                torch.as_tensor(train_targets, dtype=torch.float32),
                torch.as_tensor(train_dataset_ids, dtype=torch.long),
            ),
            batch_size=head_batch_size,
            shuffle=True,
        )

        model.train()
        batch_losses: list[float] = []
        for batch_features, batch_targets, batch_dataset_ids in loader:
            batch_features = batch_features.to(torch_device)
            batch_targets = batch_targets.to(torch_device)
            batch_dataset_ids = batch_dataset_ids.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(batch_features, batch_dataset_ids)
            mse_loss = F.mse_loss(predictions, batch_targets)
            rank_loss = _pairwise_rank_loss(predictions, batch_targets)
            loss = mse_loss + (rank_loss_weight * rank_loss)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        val_report = _evaluate_dataset_specific_model(model, datasets, split_name="val", device=torch_device)
        val_key = (
            float(val_report["summary"]["mean_spearman"]),
            float(val_report["summary"]["mean_mse"]),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)) if batch_losses else 0.0,
                "val_summary": val_report["summary"],
            }
        )

        if val_key[0] > best_key[0] or (val_key[0] == best_key[0] and val_key[1] < best_key[1]):
            best_key = val_key
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    final_val = _evaluate_dataset_specific_model(model, datasets, split_name="val", device=torch_device)
    return model, history, best_epoch, final_val


def run_pooled_embedding_head(
    dataset_config: PathLike,
    *,
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "openai",
    clip_tta_mode: str = "none",
    clip_tta_resize: Optional[int] = None,
    batch_size: int = 32,
    device: str = "cpu",
    cache_dir: PathLike = ".cache/memscore",
    hidden_dim: int = 256,
    dropout: float = 0.1,
    epochs: int = 25,
    patience: int = 6,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    head_batch_size: int = 256,
    rank_loss_weight: float = 0.2,
    seed: int = 0,
    output_json: Optional[PathLike] = None,
) -> dict[str, object]:
    specs = load_fixed_split_dataset_specs(dataset_config)
    datasets = load_embedded_datasets(
        specs,
        clip_model=clip_model,
        clip_pretrained=clip_pretrained,
        clip_tta_mode=normalize_tta_mode(clip_tta_mode),
        clip_tta_resize=clip_tta_resize,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
    )
    model, history, best_epoch, final_val = _train_embedding_head(
        datasets,
        hidden_dim=hidden_dim,
        dropout=dropout,
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        head_batch_size=head_batch_size,
        rank_loss_weight=rank_loss_weight,
        seed=seed,
        device=device,
    )
    torch_device = torch.device(device)
    final_test = _evaluate_model(model, datasets, split_name="test", device=torch_device)

    result = {
        "metadata": {
            "dataset_config": str(Path(dataset_config).expanduser().resolve()),
            "clip_model": clip_model,
            "clip_pretrained": clip_pretrained,
            "clip_tta_mode": normalize_tta_mode(clip_tta_mode),
            "clip_tta_resize": clip_tta_resize,
            "batch_size": batch_size,
            "device": device,
            "cache_dir": str(Path(cache_dir).expanduser().resolve()),
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "epochs_requested": epochs,
            "epochs_trained": len(history),
            "patience": patience,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "head_batch_size": head_batch_size,
            "rank_loss_weight": rank_loss_weight,
            "seed": seed,
            "best_epoch": best_epoch,
        },
        "datasets": {
            dataset.name: {
                "sizes": dataset.sizes,
                "train_score_mean": dataset.mean,
                "train_score_std": dataset.std,
                "val_metrics": final_val["datasets"][dataset.name]["metrics"],
                "test_metrics": final_test["datasets"][dataset.name]["metrics"],
            }
            for dataset in datasets
        },
        "validation_summary": final_val["summary"],
        "test_summary": final_test["summary"],
        "training_history": history,
    }

    if output_json:
        destination = Path(output_json).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_lodo_embedding_head(
    dataset_config: PathLike,
    *,
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "openai",
    clip_tta_mode: str = "none",
    clip_tta_resize: Optional[int] = None,
    batch_size: int = 32,
    device: str = "cpu",
    cache_dir: PathLike = ".cache/memscore",
    hidden_dim: int = 256,
    dropout: float = 0.1,
    epochs: int = 25,
    patience: int = 6,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    head_batch_size: int = 256,
    rank_loss_weight: float = 0.2,
    seed: int = 0,
    output_json: Optional[PathLike] = None,
) -> dict[str, object]:
    specs = load_fixed_split_dataset_specs(dataset_config)
    datasets = load_embedded_datasets(
        specs,
        clip_model=clip_model,
        clip_pretrained=clip_pretrained,
        clip_tta_mode=normalize_tta_mode(clip_tta_mode),
        clip_tta_resize=clip_tta_resize,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
    )
    if len(datasets) < 2:
        raise ValueError("LODO training requires at least two datasets")

    torch_device = torch.device(device)
    per_target: dict[str, object] = {}
    target_test_spearmans: list[float] = []
    target_test_mses: list[float] = []

    for target in datasets:
        source_datasets = [dataset for dataset in datasets if dataset.name != target.name]
        model, history, best_epoch, source_val = _train_embedding_head(
            source_datasets,
            hidden_dim=hidden_dim,
            dropout=dropout,
            epochs=epochs,
            patience=patience,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            head_batch_size=head_batch_size,
            rank_loss_weight=rank_loss_weight,
            seed=seed,
            device=device,
        )
        target_val = _evaluate_model(model, [target], split_name="val", device=torch_device)
        target_test = _evaluate_model(model, [target], split_name="test", device=torch_device)
        target_metrics = target_test["datasets"][target.name]["metrics"]
        target_test_spearmans.append(float(target_metrics["spearman"]))
        target_test_mses.append(float(target_metrics["mse"]))
        per_target[target.name] = {
            "target_sizes": target.sizes,
            "source_datasets": [dataset.name for dataset in source_datasets],
            "best_epoch": best_epoch,
            "epochs_trained": len(history),
            "source_validation_summary": source_val["summary"],
            "target_val_metrics": target_val["datasets"][target.name]["metrics"],
            "target_test_metrics": target_metrics,
            "training_history": history,
        }

    result = {
        "metadata": {
            "dataset_config": str(Path(dataset_config).expanduser().resolve()),
            "clip_model": clip_model,
            "clip_pretrained": clip_pretrained,
            "clip_tta_mode": normalize_tta_mode(clip_tta_mode),
            "clip_tta_resize": clip_tta_resize,
            "batch_size": batch_size,
            "device": device,
            "cache_dir": str(Path(cache_dir).expanduser().resolve()),
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "epochs_requested": epochs,
            "patience": patience,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "head_batch_size": head_batch_size,
            "rank_loss_weight": rank_loss_weight,
            "seed": seed,
        },
        "targets": per_target,
        "summary": {
            "mean_target_test_spearman": float(np.mean(target_test_spearmans)),
            "mean_target_test_mse": float(np.mean(target_test_mses)),
        },
    }
    if output_json:
        destination = Path(output_json).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_multihead_embedding_head(
    dataset_config: PathLike,
    *,
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "openai",
    clip_tta_mode: str = "none",
    clip_tta_resize: Optional[int] = None,
    batch_size: int = 32,
    device: str = "cpu",
    cache_dir: PathLike = ".cache/memscore",
    hidden_dim: int = 256,
    dropout: float = 0.1,
    epochs: int = 25,
    patience: int = 6,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    head_batch_size: int = 256,
    rank_loss_weight: float = 0.2,
    seed: int = 0,
    output_json: Optional[PathLike] = None,
) -> dict[str, object]:
    specs = load_fixed_split_dataset_specs(dataset_config)
    datasets = load_embedded_datasets(
        specs,
        clip_model=clip_model,
        clip_pretrained=clip_pretrained,
        clip_tta_mode=normalize_tta_mode(clip_tta_mode),
        clip_tta_resize=clip_tta_resize,
        batch_size=batch_size,
        device=device,
        cache_dir=cache_dir,
    )
    model, history, best_epoch, final_val = _train_dataset_specific_head(
        datasets,
        hidden_dim=hidden_dim,
        dropout=dropout,
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        head_batch_size=head_batch_size,
        rank_loss_weight=rank_loss_weight,
        seed=seed,
        device=device,
    )
    torch_device = torch.device(device)
    final_test = _evaluate_dataset_specific_model(model, datasets, split_name="test", device=torch_device)

    result = {
        "metadata": {
            "dataset_config": str(Path(dataset_config).expanduser().resolve()),
            "clip_model": clip_model,
            "clip_pretrained": clip_pretrained,
            "clip_tta_mode": normalize_tta_mode(clip_tta_mode),
            "clip_tta_resize": clip_tta_resize,
            "batch_size": batch_size,
            "device": device,
            "cache_dir": str(Path(cache_dir).expanduser().resolve()),
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "epochs_requested": epochs,
            "epochs_trained": len(history),
            "patience": patience,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "head_batch_size": head_batch_size,
            "rank_loss_weight": rank_loss_weight,
            "seed": seed,
            "best_epoch": best_epoch,
            "num_dataset_heads": len(datasets),
        },
        "datasets": {
            dataset.name: {
                "sizes": dataset.sizes,
                "train_score_mean": dataset.mean,
                "train_score_std": dataset.std,
                "val_metrics": final_val["datasets"][dataset.name]["metrics"],
                "test_metrics": final_test["datasets"][dataset.name]["metrics"],
            }
            for dataset in datasets
        },
        "validation_summary": final_val["summary"],
        "test_summary": final_test["summary"],
        "training_history": history,
    }

    if output_json:
        destination = Path(output_json).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
