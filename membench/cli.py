from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np

from .clip_regression import run_clip_regression
from .data import DatasetSplits, load_lamem_splits, load_manifest
from .figrim import parse_pca_dims, parse_positive_dims, run_figrim_study, write_study_outputs
from .metrics import compute_metrics
from .resmem_baseline import ResMemScorer, ensure_resmem_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark simpler memorability models against ResMem.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    lamem = subparsers.add_parser("lamem", help="Run a LaMem benchmark using official split files.")
    lamem.add_argument("--lamem-root", required=True, help="Path to the LaMem dataset root.")
    lamem.add_argument("--splits-dir", required=True, help="Path to the directory containing LaMem split files.")
    lamem.add_argument("--fold", type=int, default=1, help="LaMem fold number when using train_<fold>.txt splits.")
    lamem.add_argument("--train-file", help="Optional explicit train split file.")
    lamem.add_argument("--val-file", help="Optional explicit validation split file.")
    lamem.add_argument("--test-file", help="Optional explicit test split file.")
    _add_shared_args(lamem)

    manifest = subparsers.add_parser("manifest", help="Run a benchmark from a CSV manifest with path,score,split columns.")
    manifest.add_argument("--manifest", required=True, help="CSV containing path, score, split, and optional id.")
    manifest.add_argument("--root", help="Base directory for relative image paths in the manifest.")
    _add_shared_args(manifest)

    figrim = subparsers.add_parser("figrim-study", help="Run repeated FIGRIM sweeps across CLIP backbones and PCA dimensions.")
    figrim.add_argument("--figrim-root", required=True, help="Path to the FIGRIM download directory.")
    figrim.add_argument(
        "--protocols",
        nargs="+",
        default=["across", "within"],
        help="FIGRIM protocol CSVs to evaluate: across/within or across_category/within_category.",
    )
    figrim.add_argument(
        "--score-types",
        nargs="+",
        default=["corrected_hit_rate"],
        help="FIGRIM target definitions to evaluate: corrected_hit_rate or hit_rate.",
    )
    figrim.add_argument(
        "--pca-dims",
        nargs="+",
        default=["none", "128", "64", "32"],
        help="PCA dimensions to evaluate before ridge regression; use 'none' to keep full CLIP embeddings.",
    )
    figrim.add_argument(
        "--pls-dims",
        nargs="*",
        default=[],
        help="Optional PLS component counts to evaluate as a supervised regression alternative.",
    )
    figrim.add_argument("--num-splits", type=int, default=10, help="Number of repeated stratified splits to run.")
    figrim.add_argument("--train-size", type=float, default=0.7, help="Fraction of data assigned to train.")
    figrim.add_argument("--val-size", type=float, default=0.1, help="Fraction of data assigned to validation.")
    figrim.add_argument("--test-size", type=float, default=0.2, help="Fraction of data assigned to test.")
    figrim.add_argument(
        "--clip-models",
        nargs="+",
        default=["ViT-B-32", "RN50"],
        help="One or more CLIP vision backbones to evaluate.",
    )
    figrim.add_argument(
        "--clip-pretrained",
        default="openai",
        help="open_clip pretrained tag, for example 'openai' or 'laion2b_s32b_b82k'.",
    )
    figrim.add_argument(
        "--clip-tta",
        default="none",
        help="CLIP embedding test-time augmentation: none, fivecrop, or tencrop.",
    )
    figrim.add_argument(
        "--clip-tta-resize",
        type=int,
        help="Optional resize target for CLIP TTA; defaults to the standard 0.875 crop heuristic.",
    )
    figrim.add_argument("--batch-size", type=int, default=32, help="Batch size for ResMem and CLIP inference.")
    figrim.add_argument("--device", default="cpu", help="Torch device; defaults to cpu.")
    figrim.add_argument(
        "--cache-dir",
        default=".cache/membench",
        help="Directory for CLIP embeddings and downloaded checkpoints.",
    )
    figrim.add_argument("--resmem-checkpoint", help="Optional explicit path to a ResMem model.pt checkpoint.")
    figrim.add_argument(
        "--output-json",
        default="results/figrim_study.json",
        help="Where to write the detailed FIGRIM study JSON.",
    )
    figrim.add_argument(
        "--output-csv",
        default="results/figrim_study_summary.csv",
        help="Where to write the compact FIGRIM study summary CSV.",
    )
    figrim.add_argument("--random-state", type=int, default=0, help="Random seed for repeated data splits.")
    figrim.add_argument(
        "--ensemble-weight-step",
        type=float,
        default=0.05,
        help="Simplex step size for validation-weighted ensemble search when multiple backbones are present.",
    )

    return parser


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--clip-models",
        nargs="+",
        default=["ViT-L-14"],
        help="One or more CLIP vision backbones to evaluate.",
    )
    parser.add_argument(
        "--clip-pretrained",
        default="openai",
        help="open_clip pretrained tag, for example 'openai' or 'laion2b_s32b_b82k'.",
    )
    parser.add_argument(
        "--clip-tta",
        default="none",
        help="CLIP embedding test-time augmentation: none, fivecrop, or tencrop.",
    )
    parser.add_argument(
        "--clip-tta-resize",
        type=int,
        help="Optional resize target for CLIP TTA; defaults to the standard 0.875 crop heuristic.",
    )
    parser.add_argument(
        "--regressors",
        nargs="+",
        default=["ridge", "mlp"],
        help="Lightweight regressors to compare on top of frozen embeddings.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for ResMem and CLIP inference.")
    parser.add_argument("--device", default="cpu", help="Torch device; defaults to cpu.")
    parser.add_argument(
        "--cache-dir",
        default=".cache/membench",
        help="Directory for CLIP embeddings and downloaded checkpoints.",
    )
    parser.add_argument("--resmem-checkpoint", help="Optional explicit path to a ResMem model.pt checkpoint.")
    parser.add_argument(
        "--output-json",
        default="results/benchmark_results.json",
        help="Where to write the benchmark summary JSON.",
    )
    parser.add_argument(
        "--output-predictions",
        default="results/test_predictions.csv",
        help="Where to write test-set predictions for the compared models.",
    )
    parser.add_argument("--random-state", type=int, default=0, help="Random seed for the MLP regressor.")
    parser.add_argument("--limit-train", type=int, help="Optional cap on the number of training records.")
    parser.add_argument("--limit-val", type=int, help="Optional cap on the number of validation records.")
    parser.add_argument("--limit-test", type=int, help="Optional cap on the number of test records.")


def _load_splits(args: argparse.Namespace) -> DatasetSplits:
    if args.command == "lamem":
        return load_lamem_splits(
            root=args.lamem_root,
            splits_dir=args.splits_dir,
            fold=args.fold,
            train_file=args.train_file,
            val_file=args.val_file,
            test_file=args.test_file,
        )
    if args.command == "manifest":
        return load_manifest(args.manifest, root=args.root)
    raise ValueError(f"Unhandled command {args.command!r}")


def _subset_records(records, limit: Optional[int], seed: int):
    if limit is None or limit <= 0 or len(records) <= limit:
        return records
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(records), size=limit, replace=False))
    return [records[index] for index in indices.tolist()]


def _write_predictions(path: Path, splits: DatasetSplits, resmem_predictions: np.ndarray, clip_predictions: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "image_path", "ground_truth", "resmem_prediction", "clip_prediction"])
        for record, resmem_pred, clip_pred in zip(splits.test, resmem_predictions, clip_predictions):
            writer.writerow(
                [
                    record.identifier,
                    str(record.image_path),
                    f"{record.score:.8f}",
                    f"{float(resmem_pred):.8f}",
                    f"{float(clip_pred):.8f}",
                ]
            )


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "figrim-study":
        study = run_figrim_study(
            figrim_root=args.figrim_root,
            protocols=args.protocols,
            score_types=args.score_types,
            clip_models=args.clip_models,
            clip_pretrained=args.clip_pretrained,
            clip_tta_mode=args.clip_tta,
            clip_tta_resize=args.clip_tta_resize,
            pca_dims=parse_pca_dims(args.pca_dims),
            pls_dims=parse_positive_dims(args.pls_dims),
            num_splits=args.num_splits,
            batch_size=args.batch_size,
            device=args.device,
            cache_dir=args.cache_dir,
            random_state=args.random_state,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            ensemble_weight_step=args.ensemble_weight_step,
            resmem_checkpoint=args.resmem_checkpoint,
        )
        output_json = Path(args.output_json).expanduser().resolve()
        output_csv = Path(args.output_csv).expanduser().resolve()
        write_study_outputs(study, output_json=output_json, output_csv=output_csv)
        print(json.dumps(study["summary_rows"], indent=2))
        print(f"Wrote study JSON to {output_json}")
        print(f"Wrote study CSV to {output_csv}")
        return 0

    splits = _load_splits(args)
    splits = DatasetSplits(
        train=_subset_records(splits.train, args.limit_train, args.random_state),
        val=_subset_records(splits.val, args.limit_val, args.random_state + 1),
        test=_subset_records(splits.test, args.limit_test, args.random_state + 2),
    )

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.resmem_checkpoint).expanduser().resolve() if args.resmem_checkpoint else ensure_resmem_checkpoint(cache_dir / "resmem")
    resmem_scorer = ResMemScorer(checkpoint_path=checkpoint_path, batch_size=args.batch_size, device=args.device)
    resmem_predictions = resmem_scorer.predict(splits.test)
    test_truth = np.asarray([record.score for record in splits.test], dtype=np.float32)
    resmem_metrics = compute_metrics(test_truth, resmem_predictions)

    clip_runs = []
    best_clip = None
    for clip_model in args.clip_models:
        clip_result = run_clip_regression(
            train_records=splits.train,
            val_records=splits.val,
            test_records=splits.test,
            clip_model=clip_model,
            clip_pretrained=args.clip_pretrained,
            batch_size=args.batch_size,
            cache_dir=cache_dir / "clip",
            device=args.device,
            regressors=args.regressors,
            random_state=args.random_state,
            tta_mode=args.clip_tta,
            tta_resize=args.clip_tta_resize,
        )
        clip_runs.append(clip_result)
        if best_clip is None or clip_result.val_metrics.spearman > best_clip.val_metrics.spearman:
            best_clip = clip_result

    assert best_clip is not None

    summary = {
        "dataset": args.command,
        "sizes": {
            "train": len(splits.train),
            "val": len(splits.val),
            "test": len(splits.test),
        },
        "resmem": {
            "checkpoint_path": str(checkpoint_path),
            "test_metrics": resmem_metrics.as_dict(),
        },
        "clip_search": [result.as_dict() for result in clip_runs],
        "best_clip": best_clip.as_dict(),
        "delta_vs_resmem": {
            "spearman": best_clip.test_metrics.spearman - resmem_metrics.spearman,
            "mse": best_clip.test_metrics.mse - resmem_metrics.mse,
        },
    }

    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    output_predictions = Path(args.output_predictions).expanduser().resolve()
    _write_predictions(output_predictions, splits, resmem_predictions, best_clip.test_predictions)

    print(json.dumps(summary, indent=2))
    print(f"Wrote summary to {output_json}")
    print(f"Wrote test predictions to {output_predictions}")
    return 0
