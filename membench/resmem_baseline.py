from __future__ import annotations

import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from .data import Record

PathLike = Union[Path, str]


def _default_cache_dir() -> Path:
    return Path(__file__).resolve().parents[1] / ".cache" / "resmem"


def ensure_resmem_checkpoint(cache_dir: Optional[PathLike] = None) -> Path:
    cache_root = Path(cache_dir) if cache_dir else _default_cache_dir()
    cache_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = cache_root / "model.pt"
    if checkpoint_path.exists():
        return checkpoint_path

    with urllib.request.urlopen("https://pypi.org/pypi/resmem/json") as response:
        payload = json.load(response)

    wheel_url = None
    for artifact in payload.get("urls", []):
        if artifact.get("packagetype") == "bdist_wheel" and artifact.get("filename", "").endswith(".whl"):
            wheel_url = artifact["url"]
            break
    if wheel_url is None:
        raise RuntimeError("Could not find a resmem wheel on PyPI")

    wheel_path = cache_root / Path(wheel_url).name
    if not wheel_path.exists():
        urllib.request.urlretrieve(wheel_url, wheel_path)

    with zipfile.ZipFile(wheel_path) as archive:
        member = "resmem/model.pt"
        with archive.open(member) as source, checkpoint_path.open("wb") as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)

    return checkpoint_path


@dataclass
class ResMemScorer:
    checkpoint_path: Path
    batch_size: int = 32
    device: str = "cpu"

    def __post_init__(self) -> None:
        from resmem_legacy.model import ResMem, transformer

        self._device = torch.device(self.device)
        self._transform = transformer
        self._model = ResMem(pretrained=False)
        state_dict = torch.load(self.checkpoint_path, map_location=self._device)
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        self._model.eval()

    def predict(self, records: list[Record]) -> np.ndarray:
        predictions: list[float] = []
        for start in range(0, len(records), self.batch_size):
            batch = records[start : start + self.batch_size]
            tensor_batch = torch.stack([self._load_image(record.image_path) for record in batch], dim=0)
            with torch.inference_mode():
                batch_pred = self._model(tensor_batch.to(self._device)).squeeze(1).cpu().numpy()
            predictions.extend(float(value) for value in batch_pred)
        return np.asarray(predictions, dtype=np.float32)

    def _load_image(self, image_path: Path) -> torch.Tensor:
        with Image.open(image_path) as image:
            return self._transform(image.convert("RGB"))
