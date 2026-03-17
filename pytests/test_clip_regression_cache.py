from pathlib import Path

import numpy as np

from membench.clip_regression import ClipEmbedder
from membench.data import Record


def test_cache_uses_current_manifest_scores_and_identifiers(tmp_path: Path) -> None:
    cache_path = tmp_path / "embeddings.npz"
    np.savez_compressed(
        cache_path,
        features=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        scores=np.asarray([0.1, 0.2], dtype=np.float32),
        identifiers=np.asarray(["old-a", "old-b"], dtype=object),
        paths=np.asarray(
            [str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")],
            dtype=object,
        ),
    )

    records = [
        Record(image_path=tmp_path / "a.jpg", score=0.7, identifier="new-a"),
        Record(image_path=tmp_path / "b.jpg", score=0.9, identifier="new-b"),
    ]

    embedder = ClipEmbedder.__new__(ClipEmbedder)
    bundle = embedder._load_cache(cache_path, records)

    assert bundle is not None
    assert np.array_equal(bundle.features, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    assert np.array_equal(bundle.scores, np.asarray([0.7, 0.9], dtype=np.float32))
    assert bundle.identifiers == ["new-a", "new-b"]
