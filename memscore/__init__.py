"""Public API for turnkey memorability scoring and benchmarking."""

from .api import (
    BenchmarkRun,
    Prediction,
    benchmark_lamem,
    benchmark_manifest,
    benchmark_standard_manifest,
    predict_paths,
    study_figrim,
    study_lamem,
    study_manifest,
    study_memcat,
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
    "predict_paths",
    "study_figrim",
    "study_lamem",
    "study_manifest",
    "study_memcat",
    "write_benchmark_predictions",
    "write_prediction_csv",
    "write_standard_benchmark_json",
]

__version__ = "0.1.0"
