"""Default model recipe used for out-of-sample memorability benchmarks."""

STANDARD_CLIP_MODELS = ("ViT-B-32", "RN50", "ViT-B-16")
STANDARD_CLIP_PRETRAINED = "openai"
STANDARD_CLIP_TTA_MODE = "fivecrop"
STANDARD_CLIP_TTA_RESIZE = 256
STANDARD_REGRESSOR = "ridge"
STANDARD_ENSEMBLE = "mean"

