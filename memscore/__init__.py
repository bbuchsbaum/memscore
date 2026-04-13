"""Public API for turnkey memorability scoring and benchmarking."""

from .api import (
    BenchmarkRun,
    Prediction,
    benchmark_lamem,
    benchmark_manifest,
    benchmark_standard_manifest,
    default_standard_scorer_path,
    predict_paths,
    predict_standard_paths,
    resolve_standard_scorer_path,
    study_figrim,
    study_lamem,
    study_manifest,
    study_memcat,
    train_standard_scorer,
    write_benchmark_predictions,
    write_prediction_csv,
    write_standard_benchmark_json,
)
from .defaults import (
    STANDARD_CLIP_MODELS,
    STANDARD_CLIP_PRETRAINED,
    STANDARD_CLIP_TTA_MODE,
    STANDARD_CLIP_TTA_RESIZE,
    STANDARD_ENSEMBLE,
    STANDARD_REGRESSOR,
)

__all__ = [
    "BenchmarkRun",
    "Prediction",
    "STANDARD_CLIP_MODELS",
    "STANDARD_CLIP_PRETRAINED",
    "STANDARD_CLIP_TTA_MODE",
    "STANDARD_CLIP_TTA_RESIZE",
    "STANDARD_ENSEMBLE",
    "STANDARD_REGRESSOR",
    "benchmark_lamem",
    "benchmark_manifest",
    "benchmark_standard_manifest",
    "default_standard_scorer_path",
    "predict_paths",
    "predict_standard_paths",
    "resolve_standard_scorer_path",
    "study_figrim",
    "study_lamem",
    "study_manifest",
    "study_memcat",
    "train_standard_scorer",
    "write_benchmark_predictions",
    "write_prediction_csv",
    "write_standard_benchmark_json",
]

__version__ = "0.1.0"
