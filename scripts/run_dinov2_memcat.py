"""Quick DINOv2 evaluation on MemCat — standalone script."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from membench.data import Record
from membench.figrim import (
    evaluate_resmem_on_splits,
    evaluate_ridge_on_splits,
    generate_repeated_splits,
    summarize_split_metrics,
    compare_against_resmem,
    _load_prediction_cache,
    _write_prediction_cache,
)
from membench.memcat import load_memcat_dataset
from membench.resmem_baseline import ResMemScorer


def embed_dinov2(
    records: list[Record],
    model_name: str = "vit_large_patch14_dinov2.lvd142m",
    batch_size: int = 32,
    device: str = "cpu",
    cache_path: Path | None = None,
) -> np.ndarray:
    if cache_path and cache_path.exists():
        payload = np.load(cache_path, allow_pickle=True)
        cached_paths = payload["paths"].tolist()
        expected_paths = [str(r.image_path) for r in records]
        if cached_paths == expected_paths:
            print(f"  loaded cached DINOv2 embeddings from {cache_path}")
            return payload["features"].astype(np.float32)

    print(f"  extracting DINOv2 features model={model_name}")
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.to(device)
    model.eval()
    data_config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**data_config, is_training=False)

    features: list[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        imgs = []
        for rec in batch:
            with Image.open(rec.image_path) as img:
                imgs.append(transform(img.convert("RGB")))
        tensor_batch = torch.stack(imgs, dim=0).to(device)
        with torch.inference_mode():
            emb = model(tensor_batch)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        features.append(emb.cpu().numpy().astype(np.float32))

    all_features = np.concatenate(features, axis=0)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            features=all_features,
            paths=np.asarray([str(r.image_path) for r in records], dtype=object),
        )
    return all_features


def main() -> None:
    memcat_root = Path("/Users/bbuchsbaum/code/memscore/data/MemCat")
    cache_dir = Path("/Users/bbuchsbaum/code/memscore/.cache/memscore")
    checkpoint = Path("/Users/bbuchsbaum/code/memscore/data/FIGRIM/.cache/memscore/resmem/model.pt")
    results_dir = Path("/Users/bbuchsbaum/code/memscore/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    num_splits = 10
    device = "cpu"

    dataset = load_memcat_dataset(memcat_root)
    print(f"MemCat: {len(dataset.records)} records, {len(set(dataset.categories))} categories")

    splits = generate_repeated_splits(
        dataset.categories, num_splits=num_splits, random_state=0,
        train_size=0.7, val_size=0.1, test_size=0.2,
    )
    score_array = np.asarray([r.score for r in dataset.records], dtype=np.float32)

    # --- ResMem baseline ---
    resmem_cache = cache_dir / "resmem" / "memcat_predictions.npz"
    resmem_preds = _load_prediction_cache(resmem_cache, dataset.records)
    if resmem_preds is None:
        print("Scoring ResMem...")
        scorer = ResMemScorer(checkpoint_path=checkpoint, batch_size=32, device=device)
        resmem_preds = scorer.predict(dataset.records)
        _write_prediction_cache(resmem_cache, dataset.records, resmem_preds)
    else:
        print("Loaded cached ResMem predictions")
    resmem_evals = evaluate_resmem_on_splits(resmem_preds, score_array, splits)
    resmem_summary = summarize_split_metrics(resmem_evals)
    print(f"ResMem: mean_spearman={resmem_summary['mean_spearman']:.4f}")

    # --- DINOv2 ---
    dinov2_cache = cache_dir / "dinov2" / "memcat_all.npz"
    dinov2_features = embed_dinov2(
        dataset.records,
        model_name="vit_large_patch14_dinov2.lvd142m",
        batch_size=32,
        device=device,
        cache_path=dinov2_cache,
    )
    print(f"  DINOv2 features shape: {dinov2_features.shape}")

    dinov2_evals = evaluate_ridge_on_splits(
        features=dinov2_features, scores=score_array, splits=splits, pca_dim=None,
    )
    dinov2_summary = summarize_split_metrics(dinov2_evals)
    dinov2_vs_resmem = compare_against_resmem(resmem_evals, dinov2_evals)
    print(f"DINOv2 (ridge, no PCA): mean_spearman={dinov2_summary['mean_spearman']:.4f} "
          f"delta={dinov2_vs_resmem['mean_delta_spearman']:.4f} "
          f"win_rate={dinov2_vs_resmem['spearman_win_rate']:.2f}")

    # --- CLIP ViT-B-32 (reuse cached embeddings) ---
    from membench.clip_regression import ClipEmbedder
    clip_cache = cache_dir / "clip"
    clip_embedder = ClipEmbedder(
        model_name="ViT-B-32", pretrained="openai",
        batch_size=32, device=device, tta_mode="fivecrop", tta_resize=256,
    )
    clip_bundle = clip_embedder.embed_split(
        dataset.records, split_name="memcat_all", cache_dir=clip_cache,
    )
    clip_features = clip_bundle.features
    print(f"  CLIP ViT-B-32 features shape: {clip_features.shape}")

    clip_evals = evaluate_ridge_on_splits(
        features=clip_features, scores=score_array, splits=splits, pca_dim=None,
    )
    clip_summary = summarize_split_metrics(clip_evals)
    clip_vs_resmem = compare_against_resmem(resmem_evals, clip_evals)
    print(f"CLIP ViT-B-32 (ridge, no PCA): mean_spearman={clip_summary['mean_spearman']:.4f} "
          f"delta={clip_vs_resmem['mean_delta_spearman']:.4f} "
          f"win_rate={clip_vs_resmem['spearman_win_rate']:.2f}")

    # --- Fused CLIP + DINOv2 ---
    fused_features = np.concatenate([clip_features, dinov2_features], axis=1)
    print(f"  Fused features shape: {fused_features.shape}")

    fused_evals = evaluate_ridge_on_splits(
        features=fused_features, scores=score_array, splits=splits, pca_dim=None,
    )
    fused_summary = summarize_split_metrics(fused_evals)
    fused_vs_resmem = compare_against_resmem(resmem_evals, fused_evals)
    print(f"CLIP+DINOv2 fused (ridge, no PCA): mean_spearman={fused_summary['mean_spearman']:.4f} "
          f"delta={fused_vs_resmem['mean_delta_spearman']:.4f} "
          f"win_rate={fused_vs_resmem['spearman_win_rate']:.2f}")

    # --- Summary ---
    print("\n=== MemCat Summary ===")
    print(f"{'Model':<35} {'Spearman':>10} {'Delta':>10} {'Win%':>6}")
    print("-" * 65)
    print(f"{'ResMem':<35} {resmem_summary['mean_spearman']:>10.4f} {'--':>10} {'--':>6}")
    print(f"{'DINOv2 (ridge)':<35} {dinov2_summary['mean_spearman']:>10.4f} "
          f"{dinov2_vs_resmem['mean_delta_spearman']:>+10.4f} "
          f"{dinov2_vs_resmem['spearman_win_rate']*100:>5.0f}%")
    print(f"{'CLIP ViT-B-32 (ridge)':<35} {clip_summary['mean_spearman']:>10.4f} "
          f"{clip_vs_resmem['mean_delta_spearman']:>+10.4f} "
          f"{clip_vs_resmem['spearman_win_rate']*100:>5.0f}%")
    print(f"{'CLIP+DINOv2 fused (ridge)':<35} {fused_summary['mean_spearman']:>10.4f} "
          f"{fused_vs_resmem['mean_delta_spearman']:>+10.4f} "
          f"{fused_vs_resmem['spearman_win_rate']*100:>5.0f}%")

    results = {
        "resmem": resmem_summary,
        "dinov2": {**dinov2_summary, "vs_resmem": dinov2_vs_resmem},
        "clip_vit_b_32": {**clip_summary, "vs_resmem": clip_vs_resmem},
        "clip_dinov2_fused": {**fused_summary, "vs_resmem": fused_vs_resmem},
    }
    out = results_dir / "memcat_dinov2.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
