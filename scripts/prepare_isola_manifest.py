from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat


def _score(record, score_type: str) -> float:
    hits = float(record.hits)
    misses = float(record.misses)
    false_alarms = float(record.false_alarms)
    correct_rejections = float(record.correct_rejections)
    hit_rate = hits / (hits + misses)
    if score_type == "hit_rate":
        return hit_rate
    false_alarm_rate = false_alarms / (false_alarms + correct_rejections)
    corrected = 0.5 + 0.5 * (hit_rate - false_alarm_rate)
    return float(np.clip(corrected, 0.0, 1.0))


def _original_filename(cropped_name: str) -> str:
    if cropped_name.endswith("_crop.jpg"):
        return cropped_name[: -len("_crop.jpg")] + ".jpg"
    if cropped_name.endswith("_crop.png"):
        return cropped_name[: -len("_crop.png")] + ".png"
    return cropped_name


def _category_lookup(sun_mat_path: Path) -> dict[str, str]:
    mat = loadmat(sun_mat_path, squeeze_me=True, struct_as_record=False)
    lookup: dict[str, str] = {}
    for group in np.atleast_1d(mat["SUN"]):
        category = str(group.category)
        for image_url in np.atleast_1d(group.images):
            lookup[Path(str(image_url)).name] = category
    return lookup


def build_manifest(
    isola_root: Path,
    output_dir: Path,
    *,
    num_images: int = 2222,
    score_type: str = "corrected_hit_rate",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "isola_manifest.csv"

    target_images = loadmat(isola_root / "Data" / "Image data" / "target_images.mat")["img"]
    target_data = loadmat(
        isola_root / "Data" / "Experiment data" / "sorted_target_data.mat",
        squeeze_me=True,
        struct_as_record=False,
    )["sorted_target_data"]
    category_by_filename = _category_lookup(isola_root / "Data" / "Image data" / "SUN_urls.mat")

    total = min(int(num_images), int(target_images.shape[3]), int(np.atleast_1d(target_data).shape[0]))

    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "score",
                "id",
                "category",
                "source_filename",
                "score_type",
            ],
        )
        writer.writeheader()

        for idx, record in enumerate(np.atleast_1d(target_data)[:total]):
            cropped_name = Path(str(record.filepath)).name
            original_name = _original_filename(cropped_name)
            category = category_by_filename.get(original_name, "unknown")
            image_array = target_images[:, :, :, idx].astype(np.uint8)
            image_name = f"{idx:04d}_{Path(cropped_name).stem}.png"
            rel_path = Path("images") / image_name
            Image.fromarray(image_array).save(images_dir / image_name)
            writer.writerow(
                {
                    "path": str(rel_path),
                    "score": f"{_score(record, score_type):.8f}",
                    "id": Path(cropped_name).stem,
                    "category": category,
                    "source_filename": original_name,
                    "score_type": score_type,
                }
            )

    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract the original Isola memorability release into images + manifest.")
    parser.add_argument("--isola-root", required=True, help="Path to the extracted Isola dataset root.")
    parser.add_argument(
        "--output-dir",
        default="data/IsolaPrepared",
        help="Directory where extracted PNGs and the manifest CSV will be written.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=2222,
        help="Number of target images to export. The CVPR paper used the first 2222 targets.",
    )
    parser.add_argument(
        "--score-type",
        choices=["corrected_hit_rate", "hit_rate"],
        default="corrected_hit_rate",
        help="Memorability score to compute from the released counts.",
    )
    args = parser.parse_args()

    manifest_path = build_manifest(
        Path(args.isola_root).expanduser().resolve(),
        Path(args.output_dir).expanduser().resolve(),
        num_images=args.num_images,
        score_type=args.score_type,
    )
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
