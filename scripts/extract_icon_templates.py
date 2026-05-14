"""
Extract icon template crops from a YOLOv8 dataset.

Usage:
    python scripts/extract_icon_templates.py datasets/floorplan-icon-detector-v4 app/assets/icon_templates
"""

import argparse
import ast
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract labeled icon crops from a YOLOv8 dataset.",
    )
    parser.add_argument("dataset_dir", help="Roboflow YOLOv8 dataset directory")
    parser.add_argument("output_dir", help="Directory to save extracted templates")
    parser.add_argument(
        "--padding",
        type=int,
        default=2,
        help="Padding pixels to include around each bounding box",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    class_names = read_class_names(dataset_dir / "data.yaml")

    saved_count = 0
    for split in ("train", "valid", "test"):
        image_dir = dataset_dir / split / "images"
        label_dir = dataset_dir / split / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            continue

        for image_path in iter_images(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue

            saved_count += extract_image_crops(
                image_path=image_path,
                label_path=label_path,
                class_names=class_names,
                output_dir=output_dir,
                split=split,
                padding=args.padding,
            )

    print(f"Saved {saved_count} icon templates to {output_dir}")


def read_class_names(data_yaml: Path) -> list[str]:
    for line in data_yaml.read_text(encoding="utf-8").splitlines():
        if line.startswith("names:"):
            return list(ast.literal_eval(line.split(":", 1)[1].strip()))
    raise ValueError(f"Could not find names in {data_yaml}")


def iter_images(image_dir: Path) -> Iterable[Path]:
    return (
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def extract_image_crops(
    image_path: Path,
    label_path: Path,
    class_names: list[str],
    output_dir: Path,
    split: str,
    padding: int,
) -> int:
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size
    count = 0

    for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        parts = line.split()
        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        class_name = class_names[class_id]
        center_x, center_y, width, height = [float(value) for value in parts[1:]]

        left = int(round((center_x - width / 2) * image_width)) - padding
        top = int(round((center_y - height / 2) * image_height)) - padding
        right = int(round((center_x + width / 2) * image_width)) + padding
        bottom = int(round((center_y + height / 2) * image_height)) + padding

        left = max(0, left)
        top = max(0, top)
        right = min(image_width, right)
        bottom = min(image_height, bottom)

        if right <= left or bottom <= top:
            continue

        crop = image.crop((left, top, right, bottom))
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        crop.save(class_dir / f"{split}_{image_path.stem}_{index:03d}.png")
        count += 1

    return count


if __name__ == "__main__":
    main()
