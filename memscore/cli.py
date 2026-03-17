from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from . import __version__, api


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="memscore",
        description="Score image memorability and benchmark simpler challengers against ResMem.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="Score one or more image files or directories with ResMem.")
    predict.add_argument("paths", nargs="+", help="Image files or directories to score.")
    predict.add_argument("--recursive", action="store_true", help="Recurse into subdirectories when paths are directories.")
    predict.add_argument("--batch-size", type=int, default=32, help="Batch size for ResMem inference.")
    predict.add_argument("--device", default="cpu", help="Torch device for inference.")
    predict.add_argument("--cache-dir", default=".cache/memscore", help="Directory for downloaded checkpoints.")
    predict.add_argument("--resmem-checkpoint", help="Optional explicit path to a ResMem checkpoint.")
    predict.add_argument("--output", help="Optional CSV path for predictions.")

    manifest = subparsers.add_parser(
        "benchmark-manifest",
        help="Benchmark frozen CLIP challengers against ResMem from a custom CSV manifest.",
    )
    manifest.add_argument("--manifest", required=True, help="CSV containing path, score, split, and optional id.")
    manifest.add_argument("--root", help="Base directory for relative image paths in the manifest.")
    _add_shared_benchmark_args(manifest)

    lamem = subparsers.add_parser(
        "benchmark-lamem",
        help="Benchmark frozen CLIP challengers against ResMem on LaMem split files.",
    )
    lamem.add_argument("--lamem-root", required=True, help="Path to the LaMem dataset root.")
    lamem.add_argument("--splits-dir", required=True, help="Directory containing the official LaMem split files.")
    lamem.add_argument("--fold", type=int, default=1, help="Fold number when using train_<fold>.txt splits.")
    lamem.add_argument("--train-file", help="Optional explicit train split file.")
    lamem.add_argument("--val-file", help="Optional explicit validation split file.")
    lamem.add_argument("--test-file", help="Optional explicit test split file.")
    _add_shared_benchmark_args(lamem)

    figrim = subparsers.add_parser(
        "study-figrim",
        help="Run the repeated FIGRIM robustness study used for the ResMem-vs-CLIP comparisons.",
    )
    figrim.add_argument("--figrim-root", required=True, help="Path to the FIGRIM download directory.")
    figrim.add_argument(
        "--protocols",
        nargs="+",
        default=["across", "within"],
        help="FIGRIM protocols to evaluate: across/within or across_category/within_category.",
    )
    figrim.add_argument(
        "--score-types",
        nargs="+",
        default=["corrected_hit_rate"],
        help="FIGRIM targets to evaluate: corrected_hit_rate or hit_rate.",
    )
    figrim.add_argument(
        "--pca-dims",
        nargs="+",
        default=["none", "128", "64", "32"],
        help="PCA dimensions to evaluate before ridge regression; use 'none' for full embeddings.",
    )
    figrim.add_argument(
        "--pls-dims",
        nargs="*",
        default=[],
        help="Optional PLS component counts to evaluate as a supervised alternative.",
    )
    figrim.add_argument("--num-splits", type=int, default=10, help="Number of repeated stratified splits to run.")
    figrim.add_argument("--train-size", type=float, default=0.7, help="Fraction assigned to train.")
    figrim.add_argument("--val-size", type=float, default=0.1, help="Fraction assigned to validation.")
    figrim.add_argument("--test-size", type=float, default=0.2, help="Fraction assigned to test.")
    figrim.add_argument(
        "--clip-models",
        nargs="+",
        default=["ViT-B-32", "RN50"],
        help="One or more frozen CLIP vision backbones to evaluate.",
    )
    figrim.add_argument("--clip-pretrained", default="openai", help="open_clip pretrained tag.")
    figrim.add_argument("--clip-tta", default="none", help="CLIP test-time augmentation: none, fivecrop, or tencrop.")
    figrim.add_argument("--clip-tta-resize", type=int, help="Optional resize target for CLIP test-time augmentation.")
    figrim.add_argument("--batch-size", type=int, default=32, help="Batch size for ResMem and CLIP inference.")
    figrim.add_argument("--device", default="cpu", help="Torch device for inference.")
    figrim.add_argument("--cache-dir", default=".cache/memscore", help="Directory for cached embeddings and checkpoints.")
    figrim.add_argument("--resmem-checkpoint", help="Optional explicit path to a ResMem checkpoint.")
    figrim.add_argument("--output-json", default="results/figrim_study.json", help="Where to write the study JSON.")
    figrim.add_argument("--output-csv", default="results/figrim_study_summary.csv", help="Where to write the summary CSV.")
    figrim.add_argument("--random-state", type=int, default=0, help="Random seed for repeated splits.")
    figrim.add_argument(
        "--ensemble-weight-step",
        type=float,
        default=0.05,
        help="Simplex step size for validation-weighted ensemble search.",
    )

    return parser


def _add_shared_benchmark_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--clip-models",
        nargs="+",
        default=["ViT-B-32"],
        help="One or more frozen CLIP vision backbones to evaluate.",
    )
    parser.add_argument("--clip-pretrained", default="openai", help="open_clip pretrained tag.")
    parser.add_argument("--clip-tta", default="none", help="CLIP test-time augmentation: none, fivecrop, or tencrop.")
    parser.add_argument("--clip-tta-resize", type=int, help="Optional resize target for CLIP test-time augmentation.")
    parser.add_argument(
        "--regressors",
        nargs="+",
        default=["ridge", "mlp"],
        help="Lightweight regressors to evaluate on top of frozen embeddings.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for ResMem and CLIP inference.")
    parser.add_argument("--device", default="cpu", help="Torch device for inference.")
    parser.add_argument("--cache-dir", default=".cache/memscore", help="Directory for cached embeddings and checkpoints.")
    parser.add_argument("--resmem-checkpoint", help="Optional explicit path to a ResMem checkpoint.")
    parser.add_argument("--output-json", default="results/benchmark_results.json", help="Where to write the benchmark summary JSON.")
    parser.add_argument(
        "--output-predictions",
        default="results/test_predictions.csv",
        help="Where to write per-image test predictions.",
    )
    parser.add_argument("--random-state", type=int, default=0, help="Random seed for the challenger selection path.")
    parser.add_argument("--limit-train", type=int, help="Optional cap on the number of training records.")
    parser.add_argument("--limit-val", type=int, help="Optional cap on the number of validation records.")
    parser.add_argument("--limit-test", type=int, help="Optional cap on the number of test records.")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run_predict(args: argparse.Namespace) -> int:
    predictions = api.predict_paths(
        args.paths,
        recursive=args.recursive,
        checkpoint_path=args.resmem_checkpoint,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.cache_dir,
    )
    if args.output:
        output_path = api.write_prediction_csv(predictions, args.output)
        print(f"Wrote predictions to {output_path}")
    print(
        json.dumps(
            [
                {"id": item.identifier, "image_path": str(item.image_path), "memorability": item.score}
                for item in predictions
            ],
            indent=2,
        )
    )
    return 0


def _run_benchmark_manifest(args: argparse.Namespace) -> int:
    run = api.benchmark_manifest(
        args.manifest,
        root=args.root,
        clip_models=args.clip_models,
        clip_pretrained=args.clip_pretrained,
        regressors=args.regressors,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.cache_dir,
        checkpoint_path=args.resmem_checkpoint,
        random_state=args.random_state,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        limit_test=args.limit_test,
        tta_mode=args.clip_tta,
        tta_resize=args.clip_tta_resize,
    )
    output_json = Path(args.output_json).expanduser().resolve()
    output_predictions = api.write_benchmark_predictions(run, args.output_predictions)
    _write_json(output_json, run.summary)
    print(json.dumps(run.summary, indent=2))
    print(f"Wrote summary to {output_json}")
    print(f"Wrote test predictions to {output_predictions}")
    return 0


def _run_benchmark_lamem(args: argparse.Namespace) -> int:
    run = api.benchmark_lamem(
        args.lamem_root,
        splits_dir=args.splits_dir,
        fold=args.fold,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
        clip_models=args.clip_models,
        clip_pretrained=args.clip_pretrained,
        regressors=args.regressors,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.cache_dir,
        checkpoint_path=args.resmem_checkpoint,
        random_state=args.random_state,
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        limit_test=args.limit_test,
        tta_mode=args.clip_tta,
        tta_resize=args.clip_tta_resize,
    )
    output_json = Path(args.output_json).expanduser().resolve()
    output_predictions = api.write_benchmark_predictions(run, args.output_predictions)
    _write_json(output_json, run.summary)
    print(json.dumps(run.summary, indent=2))
    print(f"Wrote summary to {output_json}")
    print(f"Wrote test predictions to {output_predictions}")
    return 0


def _run_figrim(args: argparse.Namespace) -> int:
    study = api.study_figrim(
        args.figrim_root,
        protocols=args.protocols,
        score_types=args.score_types,
        clip_models=args.clip_models,
        clip_pretrained=args.clip_pretrained,
        clip_tta_mode=args.clip_tta,
        clip_tta_resize=args.clip_tta_resize,
        pca_dims=args.pca_dims,
        pls_dims=args.pls_dims,
        num_splits=args.num_splits,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.cache_dir,
        random_state=args.random_state,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        ensemble_weight_step=args.ensemble_weight_step,
        checkpoint_path=args.resmem_checkpoint,
        output_json=args.output_json,
        output_csv=args.output_csv,
    )
    print(json.dumps(study["summary_rows"], indent=2))
    print(f"Wrote study JSON to {Path(args.output_json).expanduser().resolve()}")
    print(f"Wrote study CSV to {Path(args.output_csv).expanduser().resolve()}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "predict":
        return _run_predict(args)
    if args.command == "benchmark-manifest":
        return _run_benchmark_manifest(args)
    if args.command == "benchmark-lamem":
        return _run_benchmark_lamem(args)
    if args.command == "study-figrim":
        return _run_figrim(args)
    raise ValueError(f"Unhandled command {args.command!r}")
