from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import functional as TF

from .data import Record
from .metrics import RegressionMetrics, compute_metrics

PathLike = Union[Path, str]


def _import_open_clip():
    try:
        import open_clip  # type: ignore

        return open_clip
    except ImportError:
        repo_root = Path(__file__).resolve().parents[1]
        for candidate in (repo_root / "_vendor", repo_root.parent / "_vendor"):
            if candidate.exists():
                path_text = str(candidate)
                if path_text not in sys.path:
                    sys.path.append(path_text)
                import open_clip  # type: ignore

                return open_clip
    raise RuntimeError(
        "open_clip_torch is not installed. Install requirements-benchmark.txt or place a workspace-local "
        "_vendor directory containing open_clip_torch and its dependencies."
    )


def _sanitize(text: str) -> str:
    return text.lower().replace("/", "-").replace("@", "-").replace(" ", "-")


def normalize_tta_mode(mode: Optional[str]) -> str:
    normalized = (mode or "none").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "none": "none",
        "fivecrop": "fivecrop",
        "5crop": "fivecrop",
        "tencrop": "tencrop",
        "10crop": "tencrop",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported CLIP TTA mode {mode!r}")
    return aliases[normalized]


def tta_num_views(mode: str) -> int:
    return {"none": 1, "fivecrop": 5, "tencrop": 10}[mode]


def _regressor_candidates(random_state: int) -> dict[str, Pipeline]:
    return {
        "ridge": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-4, 4, 17))),
            ]
        ),
        "mlp": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(256, 64),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        batch_size=256,
                        learning_rate_init=3e-4,
                        max_iter=60,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=8,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


@dataclass(frozen=True)
class EmbeddingBundle:
    features: np.ndarray
    scores: np.ndarray
    identifiers: list[str]
    paths: list[str]


@dataclass(frozen=True)
class ClipResult:
    clip_model: str
    clip_pretrained: str
    regressor: str
    val_metrics: RegressionMetrics
    test_metrics: RegressionMetrics
    test_predictions: np.ndarray

    def as_dict(self) -> dict[str, object]:
        return {
            "clip_model": self.clip_model,
            "clip_pretrained": self.clip_pretrained,
            "regressor": self.regressor,
            "val_metrics": self.val_metrics.as_dict(),
            "test_metrics": self.test_metrics.as_dict(),
        }


class ClipEmbedder:
    def __init__(
        self,
        model_name: str,
        pretrained: str,
        batch_size: int = 32,
        device: str = "cpu",
        tta_mode: str = "none",
        tta_resize: Optional[int] = None,
    ) -> None:
        open_clip = _import_open_clip()
        self.model_name = model_name
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.to(self.device)
        self.model.eval()
        self.tta_mode = normalize_tta_mode(tta_mode)
        self.input_size = self._infer_input_size()
        self.tta_resize = self._resolve_tta_resize(tta_resize)
        self.num_views = tta_num_views(self.tta_mode)
        self._resize_interpolation = getattr(self.preprocess.transforms[0], "interpolation", None)

    def embed_split(
        self,
        records: list[Record],
        split_name: str,
        cache_dir: Optional[PathLike] = None,
    ) -> EmbeddingBundle:
        cache_path = None
        if cache_dir:
            cache_root = Path(cache_dir)
            cache_root.mkdir(parents=True, exist_ok=True)
            cache_path = cache_root / self._cache_name(split_name)
            cached = self._load_cache(cache_path, records)
            if cached is not None:
                return cached

        features: list[np.ndarray] = []
        for start in range(0, len(records), self.batch_size):
            batch = records[start : start + self.batch_size]
            tensor_batch = torch.stack([self._load_image_views(record.image_path) for record in batch], dim=0)
            batch_size = tensor_batch.shape[0]
            flat_batch = tensor_batch.reshape(batch_size * self.num_views, *tensor_batch.shape[2:])
            with torch.inference_mode():
                encoded = self.model.encode_image(flat_batch.to(self.device))
                encoded = encoded / encoded.norm(dim=-1, keepdim=True)
                encoded = encoded.reshape(batch_size, self.num_views, -1).mean(dim=1)
                encoded = encoded / encoded.norm(dim=-1, keepdim=True)
            features.append(encoded.cpu().numpy().astype(np.float32))

        bundle = EmbeddingBundle(
            features=np.concatenate(features, axis=0),
            scores=np.asarray([record.score for record in records], dtype=np.float32),
            identifiers=[record.identifier for record in records],
            paths=[str(record.image_path) for record in records],
        )
        if cache_path:
            np.savez_compressed(
                cache_path,
                features=bundle.features,
                scores=bundle.scores,
                identifiers=np.asarray(bundle.identifiers, dtype=object),
                paths=np.asarray(bundle.paths, dtype=object),
                metadata=np.asarray(
                    json.dumps({"model_name": self.model_name, "pretrained": self.pretrained}),
                    dtype=object,
                ),
            )
        return bundle

    def _load_cache(self, cache_path: Path, records: list[Record]) -> EmbeddingBundle | None:
        if not cache_path.exists():
            return None
        payload = np.load(cache_path, allow_pickle=True)
        cached_paths = payload["paths"].tolist()
        expected_paths = [str(record.image_path) for record in records]
        if cached_paths != expected_paths:
            return None
        return EmbeddingBundle(
            features=payload["features"].astype(np.float32),
            scores=np.asarray([record.score for record in records], dtype=np.float32),
            identifiers=[record.identifier for record in records],
            paths=[str(value) for value in cached_paths],
        )

    def _cache_name(self, split_name: str) -> str:
        stem = f"{_sanitize(self.model_name)}_{_sanitize(self.pretrained)}"
        if self.tta_mode != "none":
            stem += f"_tta-{self.tta_mode}-{self.tta_resize}"
        return f"{stem}_{split_name}_embeddings.npz"

    def _infer_input_size(self) -> int:
        image_size = getattr(getattr(self.model, "visual", None), "image_size", None)
        if isinstance(image_size, (tuple, list)):
            return int(image_size[0])
        if image_size is not None:
            return int(image_size)
        for transform in getattr(self.preprocess, "transforms", []):
            if transform.__class__.__name__ == "CenterCrop":
                size = getattr(transform, "size", None)
                if isinstance(size, (tuple, list)):
                    return int(size[0])
                return int(size)
        raise ValueError("Could not infer CLIP image size from model or preprocess")

    def _resolve_tta_resize(self, tta_resize: Optional[int]) -> Optional[int]:
        if self.tta_mode == "none":
            return None
        if tta_resize is not None:
            return int(tta_resize)
        return int(round(self.input_size / 0.875))

    def _load_image_views(self, image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.tta_mode == "none":
                return self.preprocess(image).unsqueeze(0)

            resized = TF.resize(image, size=self.tta_resize, interpolation=self._resize_interpolation, antialias=True)
            crop_size = [self.input_size, self.input_size]
            if self.tta_mode == "fivecrop":
                crops = TF.five_crop(resized, size=crop_size)
            else:
                crops = TF.ten_crop(resized, size=crop_size)
            return torch.stack([self.preprocess(crop) for crop in crops], dim=0)


def run_clip_regression(
    train_records: list[Record],
    val_records: list[Record],
    test_records: list[Record],
    clip_model: str,
    clip_pretrained: str,
    batch_size: int,
    cache_dir: Optional[PathLike],
    device: str,
    regressors: list[str],
    random_state: int = 0,
    tta_mode: str = "none",
    tta_resize: Optional[int] = None,
) -> ClipResult:
    embedder = ClipEmbedder(
        model_name=clip_model,
        pretrained=clip_pretrained,
        batch_size=batch_size,
        device=device,
        tta_mode=tta_mode,
        tta_resize=tta_resize,
    )
    train_bundle = embedder.embed_split(train_records, "train", cache_dir=cache_dir)
    val_bundle = embedder.embed_split(val_records, "val", cache_dir=cache_dir)
    test_bundle = embedder.embed_split(test_records, "test", cache_dir=cache_dir)

    candidates = _regressor_candidates(random_state)
    best_name = None
    best_val_metrics = None
    for name in regressors:
        if name not in candidates:
            raise KeyError(f"Unknown regressor {name!r}; expected one of {sorted(candidates)}")
        model = clone(candidates[name])
        model.fit(train_bundle.features, train_bundle.scores)
        val_pred = model.predict(val_bundle.features)
        val_metrics = compute_metrics(val_bundle.scores, val_pred)
        if best_val_metrics is None or val_metrics.spearman > best_val_metrics.spearman:
            best_name = name
            best_val_metrics = val_metrics

    assert best_name is not None
    assert best_val_metrics is not None

    final_model = clone(candidates[best_name])
    trainval_X = np.concatenate([train_bundle.features, val_bundle.features], axis=0)
    trainval_y = np.concatenate([train_bundle.scores, val_bundle.scores], axis=0)
    final_model.fit(trainval_X, trainval_y)
    test_predictions = final_model.predict(test_bundle.features).astype(np.float32)
    test_metrics = compute_metrics(test_bundle.scores, test_predictions)

    return ClipResult(
        clip_model=clip_model,
        clip_pretrained=clip_pretrained,
        regressor=best_name,
        val_metrics=best_val_metrics,
        test_metrics=test_metrics,
        test_predictions=test_predictions,
    )
