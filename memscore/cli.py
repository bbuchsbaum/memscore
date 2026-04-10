from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from . import __version__, api
from .defaults import (
    STANDARD_CLIP_MODELS,
    STANDARD_CLIP_PRETRAINED,
    STANDARD_CLIP_TTA_MODE,
    STANDARD_CLIP_TTA_RESIZE,
)


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

    standard = subparsers.add_parser(
        "benchmark-standard",
        help="Run the standard frozen 3-CLIP ridge mean-ensemble benchmark on a locked manifest.",
    )
    standard.add_argument("--manifest", required=True, help="CSV containing path, score, split, and optional id.")
    standard.add_argument("--root", help="Base directory for relative image paths in the manifest.")
    standard.add_argument("--dataset-name", help="Optional label used for caches and output metadata.")
    standard.add_argument(
        "--clip-models",
        nargs="+",
        default=list(STANDARD_CLIP_MODELS),
        help="Frozen CLIP backbones to ensemble. Defaults to the standard 3-CLIP recipe.",
    )
    standard.add_argument("--clip-pretrained", default=STANDARD_CLIP_PRETRAINED, help="open_clip pretrained tag.")
    standard.add_argument("--clip-tta", default=STANDARD_CLIP_TTA_MODE, help="CLIP augmentation mode.")
    standard.add_argument("--clip-tta-resize", type=int, default=STANDARD_CLIP_TTA_RESIZE, help="Resize target for CLIP TTA.")
    standard.add_argument("--batch-size", type=int, default=32, help="Batch size for ResMem and CLIP inference.")
    standard.add_argument("--device", default="cpu", help="Torch device for inference.")
    standard.add_argument("--cache-dir", default=".cache/memscore", help="Directory for cached embeddings and checkpoints.")
    standard.add_argument("--resmem-checkpoint", help="Optional explicit path to a ResMem checkpoint.")
    standard.add_argument(
        "--ensemble-weight-step",
        type=float,
        default=0.05,
        help="Simplex step size for the optional validation-weighted ensemble report.",
    )
    standard.add_argument(
        "--output-json",
        default="results/standard_benchmark.json",
        help="Where to write the benchmark summary JSON.",
    )

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

    manifest_study = subparsers.add_parser(
        "study-manifest",
        help="Run a repeated-splits study from a generic scored-image manifest.",
    )
    manifest_study.add_argument("--manifest", required=True, help="CSV containing path, score, and optional label columns.")
    manifest_study.add_argument("--root", help="Base directory for relative image paths in the manifest.")
    manifest_study.add_argument("--dataset-name", help="Optional label for outputs; defaults to the manifest stem.")
    manifest_study.add_argument("--path-column", default="path", help="Column name containing image paths.")
    manifest_study.add_argument("--score-column", default="score", help="Column name containing memorability scores.")
    manifest_study.add_argument("--id-column", default="id", help="Optional identifier column; ignored if missing.")
    manifest_study.add_argument(
        "--label-column",
        help="Optional stratification column, for example category or concept. Defaults to unstratified splits.",
    )
    manifest_study.add_argument(
        "--pca-dims",
        nargs="+",
        default=["none", "128", "64", "32"],
        help="PCA dimensions to evaluate before ridge regression; use 'none' for full embeddings.",
    )
    manifest_study.add_argument(
        "--pls-dims",
        nargs="*",
        default=[],
        help="Optional PLS component counts to evaluate as a supervised alternative.",
    )
    manifest_study.add_argument("--num-splits", type=int, default=10, help="Number of repeated stratified splits to run.")
    manifest_study.add_argument("--train-size", type=float, default=0.7, help="Fraction assigned to train.")
    manifest_study.add_argument("--val-size", type=float, default=0.1, help="Fraction assigned to validation.")
    manifest_study.add_argument("--test-size", type=float, default=0.2, help="Fraction assigned to test.")
    manifest_study.add_argument(
        "--clip-models",
        nargs="+",
        default=["ViT-B-32", "RN50"],
        help="One or more frozen CLIP vision backbones to evaluate.",
    )
    manifest_study.add_argument("--clip-pretrained", default="openai", help="open_clip pretrained tag.")
    manifest_study.add_argument("--clip-tta", default="none", help="CLIP test-time augmentation: none, fivecrop, or tencrop.")
    manifest_study.add_argument("--clip-tta-resize", type=int, help="Optional resize target for CLIP test-time augmentation.")
    manifest_study.add_argument("--batch-size", type=int, default=32, help="Batch size for ResMem and CLIP inference.")
    manifest_study.add_argument("--device", default="cpu", help="Torch device for inference.")
    manifest_study.add_argument("--cache-dir", default=".cache/memscore", help="Directory for cached embeddings and checkpoints.")
    manifest_study.add_argument("--resmem-checkpoint", help="Optional explicit path to a ResMem checkpoint.")
    manifest_study.add_argument("--output-json", default="results/manifest_study.json", help="Where to write the study JSON.")
    manifest_study.add_argument("--output-csv", default="results/manifest_study_summary.csv", help="Where to write the summary CSV.")
    manifest_study.add_argument("--random-state", type=int, default=0, help="Random seed for repeated splits.")
    manifest_study.add_argument(
        "--ensemble-weight-step",
        type=float,
        default=0.05,
        help="Simplex step size for validation-weighted ensemble search.",
    )
    manifest_study.add_argument("--limit-records", type=int, help="Optional cap on total records before repeated splitting.")

    lamem_study = subparsers.add_parser(
        "study-lamem",
        help="Run a reduced-cost LaMem study with repeated official-split subsamples.",
    )
    lamem_study.add_argument("--lamem-root", required=True, help="Path to the LaMem dataset root.")
    lamem_study.add_argument("--splits-dir", required=True, help="Directory containing the official LaMem split files.")
    lamem_study.add_argument("--fold", type=int, default=1, help="Fold number when using train_<fold>.txt splits.")
    lamem_study.add_argument("--train-file", help="Optional explicit train split file.")
    lamem_study.add_argument("--val-file", help="Optional explicit validation split file.")
    lamem_study.add_argument("--test-file", help="Optional explicit test split file.")
    lamem_study.add_argument(
        "--pca-dims",
        nargs="+",
        default=["none", "128", "64", "32"],
        help="PCA dimensions to evaluate before ridge regression; use 'none' for full embeddings.",
    )
    lamem_study.add_argument(
        "--pls-dims",
        nargs="*",
        default=[],
        help="Optional PLS component counts to evaluate as a supervised alternative.",
    )
    lamem_study.add_argument("--num-splits", type=int, default=3, help="Number of repeated subset draws to run.")
    lamem_study.add_argument(
        "--clip-models",
        nargs="+",
        default=["ViT-B-32", "RN50"],
        help="One or more frozen CLIP vision backbones to evaluate.",
    )
    lamem_study.add_argument("--clip-pretrained", default="openai", help="open_clip pretrained tag.")
    lamem_study.add_argument("--clip-tta", default="none", help="CLIP test-time augmentation: none, fivecrop, or tencrop.")
    lamem_study.add_argument("--clip-tta-resize", type=int, help="Optional resize target for CLIP test-time augmentation.")
    lamem_study.add_argument("--batch-size", type=int, default=32, help="Batch size for ResMem and CLIP inference.")
    lamem_study.add_argument("--device", default="cpu", help="Torch device for inference.")
    lamem_study.add_argument("--cache-dir", default=".cache/memscore", help="Directory for cached embeddings and checkpoints.")
    lamem_study.add_argument("--resmem-checkpoint", help="Optional explicit path to a ResMem checkpoint.")
    lamem_study.add_argument("--output-json", default="results/lamem_study.json", help="Where to write the study JSON.")
    lamem_study.add_argument("--output-csv", default="results/lamem_study_summary.csv", help="Where to write the summary CSV.")
    lamem_study.add_argument("--random-state", type=int, default=0, help="Random seed for repeated subset draws.")
    lamem_study.add_argument("--limit-train", type=int, help="Optional cap on train images per repeated subset.")
    lamem_study.add_argument("--limit-val", type=int, help="Optional cap on validation images per repeated subset.")
    lamem_study.add_argument("--limit-test", type=int, help="Optional cap on test images per repeated subset.")
    lamem_study.add_argument(
        "--ensemble-weight-step",
        type=float,
        default=0.05,
        help="Simplex step size for validation-weighted ensemble search.",
    )
    lamem_study.add_argument(
        "--no-resmem-ensembles",
        action="store_true",
        help="Disable ResMem+CLIP hybrid ensemble evaluation.",
    )

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

    memcat = subparsers.add_parser(
        "study-memcat",
        help="Run the repeated MemCat robustness study used for the ResMem-vs-CLIP comparisons.",
    )
    memcat.add_argument("--memcat-root", required=True, help="Path to the MemCat dataset root directory.")
    memcat.add_argument("--csv-path", help="Optional explicit path to the MemCat CSV metadata file.")
    memcat.add_argument(
        "--score-column",
        default="memorability_w_fa_correction",
        help="Column name for memorability scores in the CSV.",
    )
    memcat.add_argument(
        "--pca-dims",
        nargs="+",
        default=["none", "128", "64", "32"],
        help="PCA dimensions to evaluate before ridge regression; use 'none' for full embeddings.",
    )
    memcat.add_argument(
        "--pls-dims",
        nargs="*",
        default=[],
        help="Optional PLS component counts to evaluate as a supervised alternative.",
    )
    memcat.add_argument("--num-splits", type=int, default=10, help="Number of repeated stratified splits to run.")
    memcat.add_argument("--train-size", type=float, default=0.7, help="Fraction assigned to train.")
    memcat.add_argument("--val-size", type=float, default=0.1, help="Fraction assigned to validation.")
    memcat.add_argument("--test-size", type=float, default=0.2, help="Fraction assigned to test.")
    memcat.add_argument(
        "--clip-models",
        nargs="+",
        default=["ViT-B-32", "RN50"],
        help="One or more frozen CLIP vision backbones to evaluate.",
    )
    memcat.add_argument("--clip-pretrained", default="openai", help="open_clip pretrained tag.")
    memcat.add_argument("--clip-tta", default="none", help="CLIP test-time augmentation: none, fivecrop, or tencrop.")
    memcat.add_argument("--clip-tta-resize", type=int, help="Optional resize target for CLIP test-time augmentation.")
    memcat.add_argument("--batch-size", type=int, default=32, help="Batch size for ResMem and CLIP inference.")
    memcat.add_argument("--device", default="cpu", help="Torch device for inference.")
    memcat.add_argument("--cache-dir", default=".cache/memscore", help="Directory for cached embeddings and checkpoints.")
    memcat.add_argument("--resmem-checkpoint", help="Optional explicit path to a ResMem checkpoint.")
    memcat.add_argument("--output-json", default="results/memcat_study.json", help="Where to write the study JSON.")
    memcat.add_argument("--output-csv", default="results/memcat_study_summary.csv", help="Where to write the summary CSV.")
    memcat.add_argument("--random-state", type=int, default=0, help="Random seed for repeated splits.")
    memcat.add_argument(
        "--ensemble-weight-step",
        type=float,
        default=0.05,
        help="Simplex step size for validation-weighted ensemble search.",
    )

    pooled_head = subparsers.add_parser(
        "train-pooled-head",
        help="Train a pooled MLP head on frozen CLIP embeddings using locked train/val/test manifests.",
    )
    pooled_head.add_argument(
        "--dataset-config",
        required=True,
        help="CSV with columns dataset,manifest and optional root. Each manifest must contain locked train/val/test splits.",
    )
    pooled_head.add_argument("--clip-model", default="ViT-B-32", help="Frozen CLIP vision backbone.")
    pooled_head.add_argument("--clip-pretrained", default="openai", help="open_clip pretrained tag.")
    pooled_head.add_argument("--clip-tta", default="none", help="CLIP augmentation mode used during feature extraction.")
    pooled_head.add_argument("--clip-tta-resize", type=int, help="Optional resize target for CLIP augmentation.")
    pooled_head.add_argument("--batch-size", type=int, default=32, help="Batch size for CLIP embedding extraction.")
    pooled_head.add_argument("--device", default="cpu", help="Torch device for embedding extraction and head training.")
    pooled_head.add_argument("--cache-dir", default=".cache/memscore", help="Directory for cached CLIP embeddings.")
    pooled_head.add_argument("--hidden-dim", type=int, default=256, help="Hidden size for the pooled MLP head.")
    pooled_head.add_argument("--dropout", type=float, default=0.1, help="Dropout used in the pooled MLP head.")
    pooled_head.add_argument("--epochs", type=int, default=25, help="Maximum head-training epochs.")
    pooled_head.add_argument("--patience", type=int, default=6, help="Early-stopping patience on pooled validation metrics.")
    pooled_head.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate for the MLP head.")
    pooled_head.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay for the MLP head.")
    pooled_head.add_argument("--head-batch-size", type=int, default=256, help="Batch size for pooled head training.")
    pooled_head.add_argument(
        "--rank-loss-weight",
        type=float,
        default=0.2,
        help="Weight on the pairwise rank loss term in addition to MSE.",
    )
    pooled_head.add_argument("--seed", type=int, default=0, help="Random seed for pooled head training.")
    pooled_head.add_argument(
        "--output-json",
        default="results/pooled_head.json",
        help="Where to write the pooled-head training report.",
    )

    lodo_head = subparsers.add_parser(
        "train-lodo-head",
        help="Train leave-one-dataset-out MLP heads on frozen CLIP embeddings using locked manifests.",
    )
    lodo_head.add_argument(
        "--dataset-config",
        required=True,
        help="CSV with columns dataset,manifest and optional root. Each manifest must contain locked train/val/test splits.",
    )
    lodo_head.add_argument("--clip-model", default="ViT-B-32", help="Frozen CLIP vision backbone.")
    lodo_head.add_argument("--clip-pretrained", default="openai", help="open_clip pretrained tag.")
    lodo_head.add_argument("--clip-tta", default="none", help="CLIP augmentation mode used during feature extraction.")
    lodo_head.add_argument("--clip-tta-resize", type=int, help="Optional resize target for CLIP augmentation.")
    lodo_head.add_argument("--batch-size", type=int, default=32, help="Batch size for CLIP embedding extraction.")
    lodo_head.add_argument("--device", default="cpu", help="Torch device for embedding extraction and head training.")
    lodo_head.add_argument("--cache-dir", default=".cache/memscore", help="Directory for cached CLIP embeddings.")
    lodo_head.add_argument("--hidden-dim", type=int, default=256, help="Hidden size for the leave-one-out MLP head.")
    lodo_head.add_argument("--dropout", type=float, default=0.1, help="Dropout used in the leave-one-out MLP head.")
    lodo_head.add_argument("--epochs", type=int, default=25, help="Maximum head-training epochs.")
    lodo_head.add_argument("--patience", type=int, default=6, help="Early-stopping patience on source validation metrics.")
    lodo_head.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate for the MLP head.")
    lodo_head.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay for the MLP head.")
    lodo_head.add_argument("--head-batch-size", type=int, default=256, help="Batch size for leave-one-out head training.")
    lodo_head.add_argument(
        "--rank-loss-weight",
        type=float,
        default=0.2,
        help="Weight on the pairwise rank loss term in addition to MSE.",
    )
    lodo_head.add_argument("--seed", type=int, default=0, help="Random seed for leave-one-out head training.")
    lodo_head.add_argument(
        "--output-json",
        default="results/lodo_head.json",
        help="Where to write the leave-one-dataset-out report.",
    )

    lora_head = subparsers.add_parser(
        "train-lora-head",
        help="Train a small pooled LoRA adapter plus regression head on locked image manifests.",
    )
    lora_head.add_argument(
        "--dataset-config",
        required=True,
        help="CSV with columns dataset,manifest and optional root. Each manifest must contain locked train/val/test splits.",
    )
    lora_head.add_argument("--clip-model", default="ViT-B-32", help="CLIP vision backbone to adapt.")
    lora_head.add_argument("--clip-pretrained", default="openai", help="open_clip pretrained tag.")
    lora_head.add_argument("--device", default="cpu", help="Torch device for LoRA training.")
    lora_head.add_argument("--batch-size", type=int, default=16, help="Batch size for LoRA training.")
    lora_head.add_argument("--eval-batch-size", type=int, default=32, help="Batch size for validation and test evaluation.")
    lora_head.add_argument("--epochs", type=int, default=6, help="Maximum LoRA training epochs.")
    lora_head.add_argument("--patience", type=int, default=2, help="Early-stopping patience on pooled validation metrics.")
    lora_head.add_argument("--lora-rank", type=int, default=4, help="LoRA rank applied to the final visual MLP blocks.")
    lora_head.add_argument("--lora-alpha", type=float, default=16.0, help="LoRA scaling alpha.")
    lora_head.add_argument("--lora-dropout", type=float, default=0.05, help="Dropout inside the LoRA adapter.")
    lora_head.add_argument("--num-lora-blocks", type=int, default=2, help="Number of final visual transformer blocks to patch.")
    lora_head.add_argument("--head-dropout", type=float, default=0.1, help="Dropout applied in the regression head.")
    lora_head.add_argument("--lora-learning-rate", type=float, default=1e-4, help="AdamW learning rate for LoRA parameters.")
    lora_head.add_argument("--head-learning-rate", type=float, default=5e-4, help="AdamW learning rate for the regression head.")
    lora_head.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    lora_head.add_argument("--rank-loss-weight", type=float, default=0.1, help="Weight on the pairwise rank loss term.")
    lora_head.add_argument(
        "--train-samples-per-dataset",
        type=int,
        default=2048,
        help="Optional cap on per-dataset training samples drawn each epoch.",
    )
    lora_head.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm; set <=0 to disable.")
    lora_head.add_argument("--seed", type=int, default=0, help="Random seed for LoRA training.")
    lora_head.add_argument(
        "--output-json",
        default="results/lora_head.json",
        help="Where to write the LoRA training report.",
    )

    multihead_head = subparsers.add_parser(
        "train-multihead-head",
        help="Train a shared frozen-embedding trunk with dataset-specific output heads.",
    )
    multihead_head.add_argument(
        "--dataset-config",
        required=True,
        help="CSV with columns dataset,manifest and optional root. Each manifest must contain locked train/val/test splits.",
    )
    multihead_head.add_argument("--clip-model", default="ViT-B-32", help="Frozen CLIP vision backbone.")
    multihead_head.add_argument("--clip-pretrained", default="openai", help="open_clip pretrained tag.")
    multihead_head.add_argument("--clip-tta", default="none", help="CLIP augmentation mode used during feature extraction.")
    multihead_head.add_argument("--clip-tta-resize", type=int, help="Optional resize target for CLIP augmentation.")
    multihead_head.add_argument("--batch-size", type=int, default=32, help="Batch size for CLIP embedding extraction.")
    multihead_head.add_argument("--device", default="cpu", help="Torch device for embedding extraction and head training.")
    multihead_head.add_argument("--cache-dir", default=".cache/memscore", help="Directory for cached CLIP embeddings.")
    multihead_head.add_argument("--hidden-dim", type=int, default=256, help="Hidden size for the shared trunk.")
    multihead_head.add_argument("--dropout", type=float, default=0.1, help="Dropout used in the shared trunk.")
    multihead_head.add_argument("--epochs", type=int, default=25, help="Maximum training epochs.")
    multihead_head.add_argument("--patience", type=int, default=6, help="Early-stopping patience on pooled validation metrics.")
    multihead_head.add_argument("--learning-rate", type=float, default=3e-4, help="AdamW learning rate.")
    multihead_head.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    multihead_head.add_argument("--head-batch-size", type=int, default=256, help="Batch size for multi-head training.")
    multihead_head.add_argument(
        "--rank-loss-weight",
        type=float,
        default=0.2,
        help="Weight on the pairwise rank loss term in addition to MSE.",
    )
    multihead_head.add_argument("--seed", type=int, default=0, help="Random seed for multi-head training.")
    multihead_head.add_argument(
        "--output-json",
        default="results/multihead_head.json",
        help="Where to write the multi-head training report.",
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


def _run_benchmark_standard(args: argparse.Namespace) -> int:
    result = api.benchmark_standard_manifest(
        args.manifest,
        root=args.root,
        dataset_name=args.dataset_name,
        clip_models=args.clip_models,
        clip_pretrained=args.clip_pretrained,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.cache_dir,
        checkpoint_path=args.resmem_checkpoint,
        tta_mode=args.clip_tta,
        tta_resize=args.clip_tta_resize,
        ensemble_weight_step=args.ensemble_weight_step,
    )
    output_json = api.write_standard_benchmark_json(result, args.output_json)
    print(json.dumps(result["standard_model"], indent=2))
    print(f"Wrote standard benchmark summary to {output_json}")
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


def _run_manifest_study(args: argparse.Namespace) -> int:
    study = api.study_manifest(
        args.manifest,
        root=args.root,
        dataset_name=args.dataset_name,
        path_column=args.path_column,
        score_column=args.score_column,
        id_column=args.id_column,
        label_column=args.label_column,
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
        limit_records=args.limit_records,
    )
    print(json.dumps(study["summary_rows"], indent=2))
    print(f"Wrote study JSON to {Path(args.output_json).expanduser().resolve()}")
    print(f"Wrote study CSV to {Path(args.output_csv).expanduser().resolve()}")
    return 0


def _run_lamem_study(args: argparse.Namespace) -> int:
    study = api.study_lamem(
        args.lamem_root,
        splits_dir=args.splits_dir,
        fold=args.fold,
        train_file=args.train_file,
        val_file=args.val_file,
        test_file=args.test_file,
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
        limit_train=args.limit_train,
        limit_val=args.limit_val,
        limit_test=args.limit_test,
        ensemble_weight_step=args.ensemble_weight_step,
        checkpoint_path=args.resmem_checkpoint,
        output_json=args.output_json,
        output_csv=args.output_csv,
        include_resmem_ensembles=not args.no_resmem_ensembles,
    )
    print(json.dumps(study["summary_rows"], indent=2))
    print(f"Wrote study JSON to {Path(args.output_json).expanduser().resolve()}")
    print(f"Wrote study CSV to {Path(args.output_csv).expanduser().resolve()}")
    return 0


def _run_memcat(args: argparse.Namespace) -> int:
    study = api.study_memcat(
        args.memcat_root,
        csv_path=getattr(args, "csv_path", None),
        score_column=args.score_column,
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


def _run_pooled_head(args: argparse.Namespace) -> int:
    result = api.train_pooled_head(
        args.dataset_config,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clip_tta_mode=args.clip_tta,
        clip_tta_resize=args.clip_tta_resize,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.cache_dir,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        head_batch_size=args.head_batch_size,
        rank_loss_weight=args.rank_loss_weight,
        seed=args.seed,
        output_json=args.output_json,
    )
    print(json.dumps(result["test_summary"], indent=2))
    print(f"Wrote pooled-head report to {Path(args.output_json).expanduser().resolve()}")
    return 0


def _run_lodo_head(args: argparse.Namespace) -> int:
    result = api.train_lodo_head(
        args.dataset_config,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clip_tta_mode=args.clip_tta,
        clip_tta_resize=args.clip_tta_resize,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.cache_dir,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        head_batch_size=args.head_batch_size,
        rank_loss_weight=args.rank_loss_weight,
        seed=args.seed,
        output_json=args.output_json,
    )
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote leave-one-out report to {Path(args.output_json).expanduser().resolve()}")
    return 0


def _run_lora_head(args: argparse.Namespace) -> int:
    result = api.train_lora_head(
        args.dataset_config,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        device=args.device,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_lora_blocks=args.num_lora_blocks,
        head_dropout=args.head_dropout,
        lora_learning_rate=args.lora_learning_rate,
        head_learning_rate=args.head_learning_rate,
        weight_decay=args.weight_decay,
        rank_loss_weight=args.rank_loss_weight,
        train_samples_per_dataset=args.train_samples_per_dataset,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        output_json=args.output_json,
    )
    print(json.dumps(result["test_summary"], indent=2))
    print(f"Wrote LoRA report to {Path(args.output_json).expanduser().resolve()}")
    return 0


def _run_multihead_head(args: argparse.Namespace) -> int:
    result = api.train_multihead_head(
        args.dataset_config,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        clip_tta_mode=args.clip_tta,
        clip_tta_resize=args.clip_tta_resize,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.cache_dir,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        head_batch_size=args.head_batch_size,
        rank_loss_weight=args.rank_loss_weight,
        seed=args.seed,
        output_json=args.output_json,
    )
    print(json.dumps(result["test_summary"], indent=2))
    print(f"Wrote multi-head report to {Path(args.output_json).expanduser().resolve()}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "predict":
        return _run_predict(args)
    if args.command == "benchmark-manifest":
        return _run_benchmark_manifest(args)
    if args.command == "benchmark-standard":
        return _run_benchmark_standard(args)
    if args.command == "benchmark-lamem":
        return _run_benchmark_lamem(args)
    if args.command == "study-manifest":
        return _run_manifest_study(args)
    if args.command == "study-lamem":
        return _run_lamem_study(args)
    if args.command == "study-figrim":
        return _run_figrim(args)
    if args.command == "study-memcat":
        return _run_memcat(args)
    if args.command == "train-pooled-head":
        return _run_pooled_head(args)
    if args.command == "train-lodo-head":
        return _run_lodo_head(args)
    if args.command == "train-lora-head":
        return _run_lora_head(args)
    if args.command == "train-multihead-head":
        return _run_multihead_head(args)
    raise ValueError(f"Unhandled command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
