from __future__ import annotations

import argparse
import json
from pathlib import Path

from membench.locked_ensemble import run_locked_ensemble_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate frozen single models and ensembles on one locked manifest split.")
    parser.add_argument("--manifest", required=True, help="Locked manifest CSV with train/val/test split column.")
    parser.add_argument("--root", help="Base directory for relative image paths in the manifest.")
    parser.add_argument("--dataset-name", help="Optional dataset label used for caches and output.")
    parser.add_argument("--clip-models", nargs="+", required=True, help="Frozen CLIP backbones to include.")
    parser.add_argument("--clip-pretrained", default="openai", help="open_clip pretrained tag.")
    parser.add_argument("--clip-tta", default="none", help="CLIP augmentation mode.")
    parser.add_argument("--clip-tta-resize", type=int, help="Optional resize target for CLIP augmentation.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for CLIP and ResMem inference.")
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument("--cache-dir", default=".cache/memscore", help="Cache directory.")
    parser.add_argument("--ensemble-weight-step", type=float, default=0.05, help="Simplex step for weighted ensembles.")
    parser.add_argument("--output-json", required=True, help="Where to write the result JSON.")
    parser.add_argument("--resmem-checkpoint", help="Optional explicit ResMem checkpoint.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run_locked_ensemble_benchmark(
        manifest_path=args.manifest,
        root=args.root,
        dataset_name=args.dataset_name,
        clip_models=list(args.clip_models),
        clip_pretrained=args.clip_pretrained,
        clip_tta_mode=args.clip_tta,
        clip_tta_resize=args.clip_tta_resize,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.cache_dir,
        ensemble_weight_step=args.ensemble_weight_step,
        resmem_checkpoint=args.resmem_checkpoint,
    )
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["ensembles"], indent=2))
    print(f"Wrote locked ensemble report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
