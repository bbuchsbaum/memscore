from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode

from membench.clip_regression import ClipEmbedder, normalize_tta_mode, tta_num_views


def _dummy_preprocess(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1)


def test_tta_mode_helpers_handle_aliases() -> None:
    assert normalize_tta_mode("none") == "none"
    assert normalize_tta_mode("5-crop") == "fivecrop"
    assert normalize_tta_mode("ten_crop") == "tencrop"
    assert tta_num_views("none") == 1
    assert tta_num_views("fivecrop") == 5
    assert tta_num_views("tencrop") == 10


def test_cache_name_includes_tta_only_when_enabled() -> None:
    embedder = ClipEmbedder.__new__(ClipEmbedder)
    embedder.model_name = "ViT-B-32"
    embedder.pretrained = "openai"
    embedder.tta_mode = "none"
    embedder.tta_resize = None
    assert embedder._cache_name("train") == "vit-b-32_openai_train_embeddings.npz"

    embedder.tta_mode = "fivecrop"
    embedder.tta_resize = 256
    assert embedder._cache_name("train") == "vit-b-32_openai_tta-fivecrop-256_train_embeddings.npz"


def test_load_image_views_returns_distinct_five_crops(tmp_path: Path) -> None:
    gradient = np.zeros((20, 20, 3), dtype=np.uint8)
    gradient[..., 0] = np.arange(20, dtype=np.uint8)[None, :]
    gradient[..., 1] = np.arange(20, dtype=np.uint8)[:, None]
    image_path = tmp_path / "gradient.png"
    Image.fromarray(gradient).save(image_path)

    embedder = ClipEmbedder.__new__(ClipEmbedder)
    embedder.preprocess = _dummy_preprocess
    embedder.tta_mode = "fivecrop"
    embedder.tta_resize = 12
    embedder.input_size = 8
    embedder.num_views = 5
    embedder._resize_interpolation = InterpolationMode.BICUBIC

    views = embedder._load_image_views(image_path)

    assert views.shape[0] == 5
    assert views.shape[1:] == (3, 8, 8)
    assert not torch.allclose(views[0], views[1])
