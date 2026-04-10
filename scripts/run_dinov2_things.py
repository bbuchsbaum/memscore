"""Quick DINOv2 evaluation on a THINGS-style manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from membench.clip_regression import ClipEmbedder
from membench.data import Record
from membench.figrim import (
    _load_prediction_cache,
    compare_against_resmem,
    evaluate_resmem_on_splits,
    generate_repeated_splits,
    summarize_split_metrics,
)
from membench.manifest_study import load_manifest_study_dataset
from membench.metrics import compute_metrics


def build_fast_ridge_pipeline(pca_dim: int | None, alphas: np.ndarray) -> Pipeline:
    steps = [("scale", StandardScaler())]
    if pca_dim is not None:
        steps.append(("pca", PCA(n_components=pca_dim, svd_solver="full")))
    # Use plain Ridge with a fixed solver; RidgeCV stayed dominated by SVD and
    # was too slow for interactive directional runs on THINGS.
    steps.append(("model", Ridge(alpha=float(alphas[0]), solver="cholesky")))
    return Pipeline(steps)


def evaluate_fast_ridge_on_splits(
    *,
    features: np.ndarray,
    scores: np.ndarray,
    splits: list,
    alphas: np.ndarray,
    pca_dim: int | None = None,
) -> list[dict[str, object]]:
    evaluations: list[dict[str, object]] = []
    for split in splits:
        train_X = features[split.train]
        val_X = features[split.val]
        test_X = features[split.test]

        train_y = scores[split.train]
        val_y = scores[split.val]
        test_y = scores[split.test]

        best_alpha = None
        best_val_metrics = None
        best_val_model = None
        for alpha in alphas:
            candidate = build_fast_ridge_pipeline(pca_dim, np.asarray([alpha], dtype=np.float64))
            candidate.fit(train_X, train_y)
            val_pred = candidate.predict(val_X).astype(np.float32)
            val_metrics = compute_metrics(val_y, val_pred)
            key = (val_metrics.spearman, -val_metrics.mse)
            best_key = (
                (-np.inf, -np.inf)
                if best_val_metrics is None
                else (best_val_metrics.spearman, -best_val_metrics.mse)
            )
            if key > best_key:
                best_alpha = float(alpha)
                best_val_metrics = val_metrics
                best_val_model = candidate

        assert best_alpha is not None
        assert best_val_metrics is not None
        assert best_val_model is not None

        final_model = clone(best_val_model)
        trainval_X = np.concatenate([train_X, val_X], axis=0)
        trainval_y = np.concatenate([train_y, val_y], axis=0)
        final_model.fit(trainval_X, trainval_y)
        test_pred = final_model.predict(test_X).astype(np.float32)
        test_metrics = compute_metrics(test_y, test_pred)

        evaluations.append(
            {
                "seed": split.seed,
                "val_metrics": best_val_metrics.as_dict(),
                "test_metrics": test_metrics.as_dict(),
                "ridge_alpha": best_alpha,
            }
        )
    return evaluations


def embed_dinov2(
    records: list[Record],
    *,
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
        if start == 0 or ((start // batch_size) + 1) % 20 == 0:
            print(f"  processed {min(start + batch_size, len(records))}/{len(records)} images")

    all_features = np.concatenate(features, axis=0)
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            features=all_features,
            paths=np.asarray([str(r.image_path) for r in records], dtype=object),
        )
    return all_features


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a DINOv2 probe on a THINGS-style scored manifest.")
    parser.add_argument("--manifest", default="data/THINGS/things_pilot_balanced_500x10_manifest.csv")
    parser.add_argument("--root", default="data/THINGS/extracted")
    parser.add_argument("--dataset-name", default="things_balanced_500x10_fast")
    parser.add_argument("--label-column", default="concept")
    parser.add_argument("--cache-dir", default=".cache/memscore312")
    parser.add_argument("--output-json", default="results/things_dinov2_probe.json")
    parser.add_argument("--clip-model", default="ViT-B-32")
    parser.add_argument("--dino-model", default="vit_large_patch14_dinov2.lvd142m")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-splits", type=int, default=5)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0],
        help="Small ridge alpha grid for faster directional runs.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    alpha_grid = np.asarray(args.alphas, dtype=np.float64)

    dataset = load_manifest_study_dataset(
        args.manifest,
        root=args.root,
        dataset_name=args.dataset_name,
        label_column=args.label_column,
    )
    print(f"THINGS-style dataset: {len(dataset.records)} records, {len(set(dataset.labels))} labels")

    splits = generate_repeated_splits(
        dataset.labels,
        num_splits=args.num_splits,
        random_state=0,
        train_size=0.7,
        val_size=0.1,
        test_size=0.2,
    )
    score_array = np.asarray([r.score for r in dataset.records], dtype=np.float32)

    resmem_cache = cache_dir / "resmem" / f"{args.dataset_name}_predictions.npz"
    resmem_preds = _load_prediction_cache(resmem_cache, dataset.records)
    if resmem_preds is None:
        raise FileNotFoundError(
            f"Missing cached ResMem predictions at {resmem_cache}. Run a THINGS study first to populate them."
        )
    resmem_evals = evaluate_resmem_on_splits(resmem_preds, score_array, splits)
    resmem_summary = summarize_split_metrics(resmem_evals)
    print(f"ResMem: mean_spearman={resmem_summary['mean_spearman']:.4f}")

    clip_embedder = ClipEmbedder(
        model_name=args.clip_model,
        pretrained="openai",
        batch_size=args.batch_size,
        device=args.device,
        tta_mode="fivecrop",
        tta_resize=256,
    )
    clip_bundle = clip_embedder.embed_split(
        dataset.records,
        split_name=f"{args.dataset_name}_all",
        cache_dir=cache_dir / "clip",
    )
    clip_features = clip_bundle.features
    clip_evals = evaluate_fast_ridge_on_splits(
        features=clip_features,
        scores=score_array,
        splits=splits,
        alphas=alpha_grid,
        pca_dim=None,
    )
    clip_summary = summarize_split_metrics(clip_evals)
    clip_vs_resmem = compare_against_resmem(resmem_evals, clip_evals)
    print(
        f"CLIP {args.clip_model} (ridge, no PCA): mean_spearman={clip_summary['mean_spearman']:.4f} "
        f"delta={clip_vs_resmem['mean_delta_spearman']:.4f} "
        f"win_rate={clip_vs_resmem['spearman_win_rate']:.2f}"
    )

    dino_slug = (
        args.dino_model.replace("/", "-")
        .replace(".", "-")
        .replace(":", "-")
    )
    dino_cache = cache_dir / "dinov2" / f"{args.dataset_name}_{dino_slug}_all.npz"
    dino_features = embed_dinov2(
        dataset.records,
        model_name=args.dino_model,
        batch_size=args.batch_size,
        device=args.device,
        cache_path=dino_cache,
    )
    print(f"  DINOv2 features shape: {dino_features.shape}")
    dino_evals = evaluate_fast_ridge_on_splits(
        features=dino_features,
        scores=score_array,
        splits=splits,
        alphas=alpha_grid,
        pca_dim=None,
    )
    dino_summary = summarize_split_metrics(dino_evals)
    dino_vs_resmem = compare_against_resmem(resmem_evals, dino_evals)
    print(
        f"DINOv2 (ridge, no PCA): mean_spearman={dino_summary['mean_spearman']:.4f} "
        f"delta={dino_vs_resmem['mean_delta_spearman']:.4f} "
        f"win_rate={dino_vs_resmem['spearman_win_rate']:.2f}"
    )

    fused_features = np.concatenate([clip_features, dino_features], axis=1)
    print(f"  Fused features shape: {fused_features.shape}")
    fused_evals = evaluate_fast_ridge_on_splits(
        features=fused_features,
        scores=score_array,
        splits=splits,
        alphas=alpha_grid,
        pca_dim=None,
    )
    fused_summary = summarize_split_metrics(fused_evals)
    fused_vs_resmem = compare_against_resmem(resmem_evals, fused_evals)
    print(
        f"CLIP+DINOv2 fused (ridge, no PCA): mean_spearman={fused_summary['mean_spearman']:.4f} "
        f"delta={fused_vs_resmem['mean_delta_spearman']:.4f} "
        f"win_rate={fused_vs_resmem['spearman_win_rate']:.2f}"
    )

    results = {
        "resmem": resmem_summary,
        "alphas": alpha_grid.tolist(),
        "clip": {**clip_summary, "vs_resmem": clip_vs_resmem, "model": args.clip_model},
        "dinov2": {**dino_summary, "vs_resmem": dino_vs_resmem, "model": args.dino_model},
        "clip_dinov2_fused": {**fused_summary, "vs_resmem": fused_vs_resmem},
    }
    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
