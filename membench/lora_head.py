from __future__ import annotations

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
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .clip_regression import _import_open_clip
from .data import DatasetSplits, Record, load_manifest
from .metrics import compute_metrics
from .supervised_head import FixedSplitDatasetSpec, load_fixed_split_dataset_specs

PathLike = Union[Path, str]


@dataclass(frozen=True)
class ImageSplitDataset:
    name: str
    train: list[Record]
    val: list[Record]
    test: list[Record]
    mean: float
    std: float
    sizes: dict[str, int]

    def normalize(self, scores: np.ndarray) -> np.ndarray:
        return (scores - self.mean) / self.std

    def denormalize(self, scores: np.ndarray) -> np.ndarray:
        return (scores * self.std) + self.mean


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be >= 1")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = self.base(inputs)
        delta = self.lora_up(self.lora_down(self.dropout(inputs))) * self.scaling
        return base + delta


class RegressionHead(nn.Module):
    def __init__(self, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.proj(self.dropout(self.norm(inputs))).squeeze(-1)


class ClipLoRARegressor(nn.Module):
    def __init__(
        self,
        *,
        clip_model: str,
        clip_pretrained: str,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        num_lora_blocks: int,
        head_dropout: float,
    ) -> None:
        super().__init__()
        open_clip = _import_open_clip()
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            clip_model,
            pretrained=clip_pretrained,
        )
        for parameter in self.clip_model.parameters():
            parameter.requires_grad = False

        visual = getattr(self.clip_model, "visual", None)
        transformer = getattr(visual, "transformer", None)
        resblocks = getattr(transformer, "resblocks", None)
        if visual is None or transformer is None or resblocks is None:
            raise ValueError(f"{clip_model!r} does not expose a ViT-style visual transformer")
        if num_lora_blocks < 1 or num_lora_blocks > len(resblocks):
            raise ValueError(f"num_lora_blocks must be in [1, {len(resblocks)}]")

        self.patched_modules: list[str] = []
        for index in range(len(resblocks) - num_lora_blocks, len(resblocks)):
            block = resblocks[index]
            block.mlp.c_fc = LoRALinear(
                block.mlp.c_fc,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            block.mlp.c_proj = LoRALinear(
                block.mlp.c_proj,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            self.patched_modules.extend(
                [
                    f"visual.transformer.resblocks.{index}.mlp.c_fc",
                    f"visual.transformer.resblocks.{index}.mlp.c_proj",
                ]
            )

        embed_dim = int(getattr(visual, "output_dim", 0) or getattr(visual, "width"))
        self.head = RegressionHead(embed_dim=embed_dim, dropout=head_dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.clip_model.encode_image(images)
        features = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return self.head(features)

    def parameter_groups(
        self,
        *,
        lora_learning_rate: float,
        head_learning_rate: float,
        weight_decay: float,
    ) -> list[dict[str, object]]:
        lora_params: list[nn.Parameter] = []
        head_params: list[nn.Parameter] = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("clip_model."):
                lora_params.append(parameter)
            else:
                head_params.append(parameter)
        return [
            {"params": lora_params, "lr": lora_learning_rate, "weight_decay": weight_decay},
            {"params": head_params, "lr": head_learning_rate, "weight_decay": weight_decay},
        ]

    def count_trainable_parameters(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad))


class ImageRegressionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, samples: list[tuple[Path, float]], preprocess) -> None:
        self.samples = samples
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, target = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = self.preprocess(image)
        return tensor, torch.tensor(target, dtype=torch.float32)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def _load_image_split_datasets(specs: list[FixedSplitDatasetSpec]) -> list[ImageSplitDataset]:
    datasets: list[ImageSplitDataset] = []
    for spec in specs:
        splits: DatasetSplits = load_manifest(spec.manifest_path, root=spec.root)
        train_scores = np.asarray([record.score for record in splits.train], dtype=np.float32)
        mean = float(np.mean(train_scores))
        std = float(np.std(train_scores))
        if std < 1e-8:
            std = 1.0
        datasets.append(
            ImageSplitDataset(
                name=spec.name,
                train=splits.train,
                val=splits.val,
                test=splits.test,
                mean=mean,
                std=std,
                sizes={"train": len(splits.train), "val": len(splits.val), "test": len(splits.test)},
            )
        )
    return datasets


def _make_balanced_train_samples(
    datasets: list[ImageSplitDataset],
    *,
    rng: np.random.Generator,
    train_samples_per_dataset: Optional[int],
) -> list[tuple[Path, float]]:
    if not datasets:
        return []
    per_dataset = train_samples_per_dataset or max(len(dataset.train) for dataset in datasets)
    samples: list[tuple[Path, float]] = []
    for dataset in datasets:
        train_size = len(dataset.train)
        indices = rng.choice(train_size, size=per_dataset, replace=train_size < per_dataset)
        for index in indices.tolist():
            record = dataset.train[int(index)]
            normalized = float(dataset.normalize(np.asarray([record.score], dtype=np.float32))[0])
            samples.append((record.image_path, normalized))
    rng.shuffle(samples)
    return samples


def _make_eval_samples(dataset: ImageSplitDataset, split_name: str) -> tuple[list[tuple[Path, float]], np.ndarray]:
    records = getattr(dataset, split_name)
    truth = np.asarray([record.score for record in records], dtype=np.float32)
    samples = [
        (
            record.image_path,
            float(dataset.normalize(np.asarray([record.score], dtype=np.float32))[0]),
        )
        for record in records
    ]
    return samples, truth


def _evaluate_model(
    model: ClipLoRARegressor,
    datasets: list[ImageSplitDataset],
    *,
    split_name: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, object]:
    model.eval()
    by_dataset: dict[str, object] = {}
    spearmans: list[float] = []
    mses: list[float] = []
    with torch.inference_mode():
        for dataset in datasets:
            samples, truth = _make_eval_samples(dataset, split_name)
            loader = DataLoader(ImageRegressionDataset(samples, model.preprocess), batch_size=batch_size, shuffle=False)
            predicted_batches: list[np.ndarray] = []
            for images, _targets in loader:
                predictions = model(images.to(device)).detach().cpu().numpy().astype(np.float32)
                predicted_batches.append(predictions)
            restored = dataset.denormalize(np.concatenate(predicted_batches, axis=0))
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


def run_pooled_lora_head(
    dataset_config: PathLike,
    *,
    clip_model: str = "ViT-B-32",
    clip_pretrained: str = "openai",
    device: str = "cpu",
    batch_size: int = 16,
    eval_batch_size: int = 32,
    epochs: int = 6,
    patience: int = 2,
    lora_rank: int = 4,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    num_lora_blocks: int = 2,
    head_dropout: float = 0.1,
    lora_learning_rate: float = 1e-4,
    head_learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    rank_loss_weight: float = 0.1,
    train_samples_per_dataset: Optional[int] = 2048,
    max_grad_norm: float = 1.0,
    seed: int = 0,
    output_json: Optional[PathLike] = None,
) -> dict[str, object]:
    specs = load_fixed_split_dataset_specs(dataset_config)
    datasets = _load_image_split_datasets(specs)
    if not datasets:
        raise ValueError("No datasets available for LoRA training")

    _set_seed(seed)
    torch_device = torch.device(device)
    model = ClipLoRARegressor(
        clip_model=clip_model,
        clip_pretrained=clip_pretrained,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        num_lora_blocks=num_lora_blocks,
        head_dropout=head_dropout,
    ).to(torch_device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(
            lora_learning_rate=lora_learning_rate,
            head_learning_rate=head_learning_rate,
            weight_decay=weight_decay,
        )
    )
    rng = np.random.default_rng(seed)

    best_key = (-math.inf, math.inf)
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        train_samples = _make_balanced_train_samples(
            datasets,
            rng=rng,
            train_samples_per_dataset=train_samples_per_dataset,
        )
        loader = DataLoader(
            ImageRegressionDataset(train_samples, model.preprocess),
            batch_size=batch_size,
            shuffle=True,
        )
        model.train()
        batch_losses: list[float] = []
        for images, targets in loader:
            images = images.to(torch_device)
            targets = targets.to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(images)
            mse_loss = F.mse_loss(predictions, targets)
            rank_loss = _pairwise_rank_loss(predictions, targets)
            loss = mse_loss + (rank_loss_weight * rank_loss)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu().item()))

        val_report = _evaluate_model(model, datasets, split_name="val", device=torch_device, batch_size=eval_batch_size)
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
    final_val = _evaluate_model(model, datasets, split_name="val", device=torch_device, batch_size=eval_batch_size)
    final_test = _evaluate_model(model, datasets, split_name="test", device=torch_device, batch_size=eval_batch_size)
    result = {
        "metadata": {
            "dataset_config": str(Path(dataset_config).expanduser().resolve()),
            "clip_model": clip_model,
            "clip_pretrained": clip_pretrained,
            "device": device,
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "epochs_requested": epochs,
            "epochs_trained": len(history),
            "patience": patience,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "num_lora_blocks": num_lora_blocks,
            "head_dropout": head_dropout,
            "lora_learning_rate": lora_learning_rate,
            "head_learning_rate": head_learning_rate,
            "weight_decay": weight_decay,
            "rank_loss_weight": rank_loss_weight,
            "train_samples_per_dataset": train_samples_per_dataset,
            "max_grad_norm": max_grad_norm,
            "seed": seed,
            "best_epoch": best_epoch,
            "trainable_parameters": model.count_trainable_parameters(),
            "patched_modules": model.patched_modules,
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
