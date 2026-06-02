"""Render analyze API response detections into a simple map preview.

Usage:
    python scripts/render_analyze_response.py response.json runs/response_map.png
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


COLORS = {
    "walkable_area": (170, 235, 170),
    "blocked_area": (190, 190, 230),
    "room_area": (80, 190, 255),
    "wall_outline": (80, 80, 80),
    "text_detection": (40, 40, 40),
    "object_detection": (0, 140, 255),
    "poi_candidate": (190, 70, 190),
    "centerline_walkway": (255, 0, 0),
    "poi_access_link": (255, 0, 255),
    "center_node": (0, 0, 255),
    "center_connector": (255, 255, 0),
    "poi_access_node": (255, 0, 255),
}

DRAW_ORDER = (
    "walkable_area",
    "blocked_area",
    "room_area",
    "wall_outline",
    "text_detection",
    "object_detection",
    "centerline_walkway",
    "poi_access_link",
    "center_node",
    "center_connector",
    "poi_access_node",
    "poi_candidate",
)

OBJECT_DETECT_TYPES = {
    "accessible_restroom",
    "aed",
    "atm",
    "cafe",
    "clothing_alteration",
    "elevator",
    "escalator",
    "family_restroom",
    "infodesk",
    "phone_charging",
    "restroom_female",
    "restroom_male",
    "shoe_repair",
    "stair",
    "storage_locker",
    "subway_station",
    "water_fountain",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render analyze API response detections into a preview image.",
    )
    parser.add_argument("response_path", help="Analyze API response JSON path")
    parser.add_argument("output_path", help="Output image path")
    args = parser.parse_args()

    response_path = Path(args.response_path)
    output_path = Path(args.output_path)

    with response_path.open("r", encoding="utf-8") as file:
        response = json.load(file)

    detections = response.get("detections", [])
    canvas_size = _infer_canvas_size(detections)
    canvas = np.full((canvas_size[1], canvas_size[0], 3), 255, dtype=np.uint8)
    overlay = canvas.copy()

    for label in DRAW_ORDER:
        for detection in detections:
            if not _matches_layer(detection, label):
                continue
            _draw_detection(canvas, overlay, detection, COLORS[label])

    blended = cv2.addWeighted(overlay, 0.32, canvas, 0.68, 0)
    blended = _draw_labels(blended, detections)
    _draw_legend(blended)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    saved = cv2.imwrite(str(output_path), blended)
    if not saved:
        raise IOError(f"Failed to write output image: {output_path}")

    print(f"Saved rendered map to {output_path}")
    _print_summary(detections)


def _infer_canvas_size(detections: list[dict]) -> tuple[int, int]:
    max_x = 1
    max_y = 1
    for detection in detections:
        bbox = detection.get("bbox_px")
        if bbox:
            x, y, width, height = bbox
            max_x = max(max_x, int(x + width))
            max_y = max(max_y, int(y + height))

        geom = detection.get("geom_px", {})
        for point in _iter_points(geom):
            max_x = max(max_x, int(point[0]))
            max_y = max(max_y, int(point[1]))

    padding = 40
    return max_x + padding, max_y + padding


def _draw_detection(
    canvas: cv2.typing.MatLike,
    overlay: cv2.typing.MatLike,
    detection: dict,
    color: tuple[int, int, int],
) -> None:
    geom = detection.get("geom_px", {})
    geom_type = geom.get("type")
    coordinates = geom.get("coordinates")
    if not coordinates:
        return

    if geom_type == "Polygon":
        if not coordinates[0] or len(coordinates[0]) < 3:
            return
        polygon = np.array(coordinates[0], dtype=np.int32)
        if detection.get("label") == "wall_outline":
            cv2.polylines(canvas, [polygon], isClosed=True, color=color, thickness=6)
            return

        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(canvas, [polygon], isClosed=True, color=color, thickness=2)
    elif geom_type == "LineString":
        if len(coordinates) < 2:
            return
        line = np.array(coordinates, dtype=np.int32)
        cv2.polylines(canvas, [line], isClosed=False, color=color, thickness=3)
    elif geom_type == "Point":
        x, y = coordinates
        center = (int(x), int(y))
        cv2.circle(canvas, center, 9, (255, 255, 255), thickness=-1)
        cv2.circle(canvas, center, 7, color, thickness=-1)
        cv2.circle(canvas, center, 9, (40, 40, 40), thickness=1)


def _matches_layer(detection: dict, layer: str) -> bool:
    if layer == "text_detection":
        return detection.get("detect_type") == "text"
    if layer == "object_detection":
        return detection.get("detect_type") in OBJECT_DETECT_TYPES
    if layer == "poi_candidate":
        return detection.get("detect_type") == "poi_candidate"
    if layer == "centerline_walkway":
        return detection.get("label") == "centerline_walkway"
    if layer == "poi_access_link":
        return (detection.get("label") or "").startswith("poi_access_link")
    if layer == "center_node":
        return (
            detection.get("detect_type") == "node_candidate"
            and detection.get("label") in {
                "center_junction",
                "center_endpoint",
                "center_turn",
            }
        )
    if layer == "center_connector":
        return detection.get("label") == "center_connector"
    if layer == "poi_access_node":
        return (detection.get("label") or "").startswith("poi_access_node")
    return detection.get("label") == layer


def _draw_labels(
    image: cv2.typing.MatLike,
    detections: list[dict],
) -> cv2.typing.MatLike:
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = _load_font(15)
    small_font = _load_font(12)

    for detection in detections:
        if detection.get("detect_type") != "text" or not detection.get("ocr_text"):
            continue
        bbox = detection.get("bbox_px")
        if not bbox:
            continue
        x, y, _, _ = bbox
        draw.text(
            (x, max(0, y - 15)),
            str(detection["ocr_text"]),
            font=small_font,
            fill=(20, 20, 20),
        )

    for detection in detections:
        if detection.get("detect_type") not in OBJECT_DETECT_TYPES:
            continue
        bbox = detection.get("bbox_px")
        if not bbox:
            continue
        x, y, _, height = bbox
        draw.text(
            (x, y + height + 2),
            str(detection.get("label") or detection.get("detect_type")),
            font=small_font,
            fill=(180, 80, 0),
        )

    for detection in detections:
        if detection.get("label") == "room_area" and detection.get("ocr_text"):
            _draw_centered_text(draw, detection, detection["ocr_text"], font, (35, 35, 35))

    for detection in detections:
        if detection.get("detect_type") != "poi_candidate":
            continue
        if detection.get("label") == "store.unknown":
            continue
        geom = detection.get("geom_px", {})
        coordinates = geom.get("coordinates")
        if geom.get("type") != "Point" or not coordinates:
            continue
        text = detection.get("ocr_text") or detection.get("label")
        if not text:
            continue
        x, y = int(coordinates[0]), int(coordinates[1])
        draw.text((x + 10, y - 8), str(text), font=small_font, fill=(80, 20, 80))

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    detection: dict,
    text: str,
    font: ImageFont.ImageFont,
    color: tuple[int, int, int],
) -> None:
    bbox = detection.get("bbox_px")
    if not bbox:
        return
    x, y, width, height = bbox
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    draw.text(
        (
            x + (width - text_width) / 2,
            y + (height - text_height) / 2,
        ),
        text,
        font=font,
        fill=color,
    )


def _load_font(size: int) -> ImageFont.ImageFont:
    font_paths = (
        "C:/Windows/Fonts/malgun.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    )
    for font_path in font_paths:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def _draw_legend(image: cv2.typing.MatLike) -> None:
    labels = list(DRAW_ORDER)
    x = 18
    y = 24
    line_height = 28
    width = 350
    height = 18 + line_height * len(labels)
    cv2.rectangle(image, (x - 8, y - 18), (x + width, y + height - 18), (255, 255, 255), -1)
    cv2.rectangle(image, (x - 8, y - 18), (x + width, y + height - 18), (60, 60, 60), 1)

    for index, label in enumerate(labels):
        item_y = y + index * line_height
        color = COLORS[label]
        cv2.line(image, (x, item_y), (x + 32, item_y), color, 5)
        cv2.putText(
            image,
            label,
            (x + 42, item_y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )


def _iter_points(geom: dict) -> list[list[float]]:
    geom_type = geom.get("type")
    coordinates = geom.get("coordinates")
    if not coordinates:
        return []
    if geom_type == "Point":
        return [coordinates]
    if geom_type == "LineString":
        return coordinates
    if geom_type == "Polygon":
        return coordinates[0]
    return []


def _print_summary(detections: list[dict]) -> None:
    for label in DRAW_ORDER:
        count = sum(1 for detection in detections if _matches_layer(detection, label))
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
