"""Compare EasyOCR and PaddleOCR results on the same floorplan image.

Usage:
    python scripts/compare_ocr_engines.py input.png runs/ocr_compare
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.text_detector import TextDetector
from app.services.object_detector import ObjectDetector


FONT_PATHS = (
    Path("C:/Windows/Fonts/malgun.ttf"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare local OCR engines.")
    parser.add_argument("image_path", help="Input floorplan image path")
    parser.add_argument("output_dir", help="Directory for previews and JSON summary")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    object_detections = ObjectDetector().detect(str(image_path))

    summary = {}
    for engine in ("easyocr", "paddleocr"):
        started_at = time.perf_counter()
        detections = TextDetector(engine=engine).detect(
            str(image_path),
            object_detections=object_detections,
        )
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        summary[engine] = {
            "processing_time_ms": elapsed_ms,
            "text_count": len(detections),
            "texts": [
                {
                    "text": detection.ocr_text,
                    "confidence": detection.confidence,
                    "bbox_px": detection.bbox_px,
                }
                for detection in detections
            ],
        }
        _write_preview(image_path, output_dir / f"{engine}.png", detections)
        print(f"{engine}: {len(detections)} texts in {elapsed_ms} ms")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved comparison results to {output_dir}")


def _write_preview(image_path: Path, output_path: Path, detections: list) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")

    font = _load_font()
    for detection in detections:
        if detection.bbox_px is None:
            continue
        x, y, width, height = [int(round(value)) for value in detection.bbox_px]
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 170, 0), 2)
        _draw_text(
            image,
            detection.ocr_text or "",
            (x, max(0, y - 21)),
            font,
        )
    cv2.imwrite(str(output_path), image)


def _draw_text(
    image: cv2.typing.MatLike,
    text: str,
    position: tuple[int, int],
    font: ImageFont.ImageFont,
) -> None:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)
    x, y = position
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle((bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2), fill="white")
    draw.text((x, y), text, fill=(0, 100, 0), font=font)
    image[:, :] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _load_font() -> ImageFont.ImageFont:
    for path in FONT_PATHS:
        if path.is_file():
            return ImageFont.truetype(str(path), 16)
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
