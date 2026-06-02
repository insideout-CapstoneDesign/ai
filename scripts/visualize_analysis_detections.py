"""
Draw OCR text and ObjectDetector results on a floorplan image.

Usage:
    python scripts/visualize_analysis_detections.py input.png output.png
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas.analyze import Detection
from app.services.object_detector import ObjectDetector
from app.services.text_detector import TextDetector


TYPE_COLORS = {
    "text": (50, 180, 50),
    "elevator": (0, 180, 255),
    "escalator": (255, 120, 0),
    "stair": (80, 200, 120),
    "restroom_sign": (220, 80, 220),
    "poi_candidate": (80, 140, 255),
}
FONT_PATH = Path("C:/Windows/Fonts/malgun.ttf")
FONT_SIZE = 18


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize OCR and icon detections.",
    )
    parser.add_argument("image_path", help="Input floorplan image path")
    parser.add_argument("output_path", help="Output preview image path")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    output_path = Path(args.output_path)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")

    object_detections = ObjectDetector().detect(str(image_path))
    text_detections = TextDetector().detect(
        str(image_path),
        object_detections=object_detections,
    )

    font = load_font()

    for detection in text_detections:
        draw_detection(
            image,
            detection,
            text=detection.ocr_text or "",
            font=font,
        )

    for detection in object_detections:
        label = detection.label or detection.detect_type
        draw_detection(
            image,
            detection,
            text=f"{label} {detection.confidence:.2f}",
            font=font,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(
        f"Saved {len(text_detections)} text detections and "
        f"{len(object_detections)} object detections to {output_path}"
    )


def draw_detection(
    image: cv2.typing.MatLike,
    detection: Detection,
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    if detection.bbox_px is None:
        return

    x, y, width, height = [int(round(value)) for value in detection.bbox_px]
    right = x + width
    bottom = y + height
    color = TYPE_COLORS.get(detection.detect_type, (0, 255, 255))

    cv2.rectangle(image, (x, y), (right, bottom), color, 2)
    if text:
        draw_text(image, text[:30], (x, max(0, y - FONT_SIZE - 6)), color, font)


def draw_text(
    image: cv2.typing.MatLike,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_image)
    rgb_color = (color[2], color[1], color[0])
    x, y = position

    bbox = draw.textbbox((x, y), text, font=font)
    padding = 3
    draw.rectangle(
        (
            bbox[0] - padding,
            bbox[1] - padding,
            bbox[2] + padding,
            bbox[3] + padding,
        ),
        fill=(255, 255, 255),
    )
    draw.text((x, y), text, fill=rgb_color, font=font)

    image[:, :] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def load_font() -> ImageFont.ImageFont:
    if FONT_PATH.is_file():
        return ImageFont.truetype(str(FONT_PATH), FONT_SIZE)
    return ImageFont.load_default()


if __name__ == "__main__":
    main()
