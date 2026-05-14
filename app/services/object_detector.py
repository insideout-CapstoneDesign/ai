"""
오브젝트 추출 Detector.

도면에서 고정된 POI 아이콘을 OpenCV 템플릿 매칭으로 감지한다.
아이콘 스타일이 일정한 도면에서는 학습 모델보다 템플릿 매칭이 더 단순하고
재현 가능한 결과를 제공한다.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import cv2
import numpy as np

from app.core.config import settings
from app.schemas.analyze import Detection
from app.services.base import Detector

logger = logging.getLogger(__name__)

DetectType = Literal[
    "elevator",
    "stair",
    "escalator",
    "restroom_sign",
    "poi_candidate",
]


@dataclass(frozen=True)
class IconTemplate:
    label: str
    image: cv2.typing.MatLike
    width: int
    height: int


class ObjectDetector(Detector):
    """OpenCV 템플릿 매칭으로 도면의 POI 아이콘 객체를 감지한다."""

    name = "object"
    version = "v0.2-template-matching"

    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    TEMPLATE_SCALES = (0.7, 1.0)

    DETECT_TYPE_BY_LABEL: dict[str, DetectType] = {
        "elevator": "elevator",
        "escalator": "escalator",
        "stair": "stair",
        "restroom_male": "restroom_sign",
        "restroom_female": "restroom_sign",
        "family_restroom": "restroom_sign",
        "accessible_restroom": "restroom_sign",
    }

    def __init__(self) -> None:
        self._templates: list[IconTemplate] | None = None

    def detect(self, image_path: str) -> List[Detection]:
        """
        이미지에서 POI 아이콘 객체를 감지한다.

        템플릿 매칭 결과 중 threshold 이상인 후보를 클래스별 NMS로 중복 제거한 뒤
        Detection 형태로 반환한다.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image file not found or unreadable: {image_path}")

        templates = self._get_templates()
        if not templates:
            logger.warning("No icon templates loaded from %s", settings.icon_template_dir)
            return []

        candidates_by_label: dict[str, list[tuple[list[int], float]]] = {}
        for template in templates:
            if template.height > image.shape[0] or template.width > image.shape[1]:
                continue

            threshold = self._threshold_for_label(template.label)
            matched = cv2.matchTemplate(
                image,
                template.image,
                cv2.TM_CCOEFF_NORMED,
            )
            _, max_score, _, _ = cv2.minMaxLoc(matched)
            if max_score < threshold:
                continue

            locations = cv2.findNonZero(
                (matched >= threshold).astype("uint8")
            )
            if locations is None:
                continue

            label_candidates = candidates_by_label.setdefault(template.label, [])
            for point in locations.reshape(-1, 2):
                x = int(point[0])
                y = int(point[1])
                score = float(matched[y, x])
                label_candidates.append(
                    ([x, y, template.width, template.height], score)
                )

        detections: List[Detection] = []
        for label, candidates in candidates_by_label.items():
            detections.extend(self._deduplicate(label, candidates))

        detections = self._deduplicate_across_labels(detections)
        logger.info("Detected %d POI icon objects from %s", len(detections), image_path)
        return detections

    def _get_templates(self) -> list[IconTemplate]:
        if self._templates is None:
            self._templates = self._load_templates()
        return self._templates

    def _load_templates(self) -> list[IconTemplate]:
        template_root = Path(settings.icon_template_dir)
        if not template_root.is_absolute():
            project_root = Path(__file__).resolve().parents[2]
            template_root = project_root / template_root

        templates: list[IconTemplate] = []
        if not template_root.is_dir():
            logger.warning("Icon template directory does not exist: %s", template_root)
            return templates

        for class_dir in sorted(path for path in template_root.iterdir() if path.is_dir()):
            for template_path in sorted(class_dir.iterdir()):
                if template_path.suffix.lower() not in self.IMAGE_EXTENSIONS:
                    continue

                raw_template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
                if raw_template is None:
                    logger.warning("Failed to read icon template: %s", template_path)
                    continue

                templates.extend(
                    self._scaled_templates(
                        label=class_dir.name,
                        raw_template=raw_template,
                    )
                )

        logger.info("Loaded %d icon templates from %s", len(templates), template_root)
        return templates

    def _scaled_templates(
        self,
        label: str,
        raw_template: cv2.typing.MatLike,
    ) -> list[IconTemplate]:
        templates: list[IconTemplate] = []
        height, width = raw_template.shape[:2]

        for scale in self.TEMPLATE_SCALES:
            scaled_width = max(3, int(round(width * scale)))
            scaled_height = max(3, int(round(height * scale)))
            if scaled_width == width and scaled_height == height:
                scaled = raw_template
            else:
                scaled = cv2.resize(
                    raw_template,
                    (scaled_width, scaled_height),
                    interpolation=cv2.INTER_AREA,
                )

            templates.append(
                IconTemplate(
                    label=label,
                    image=scaled,
                    width=scaled_width,
                    height=scaled_height,
                )
            )

        return templates

    def _deduplicate(
        self,
        label: str,
        candidates: list[tuple[list[int], float]],
    ) -> list[Detection]:
        threshold = self._threshold_for_label(label)
        boxes = [candidate[0] for candidate in candidates]
        scores = [candidate[1] for candidate in candidates]
        selected_indexes = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=scores,
            score_threshold=threshold,
            nms_threshold=settings.icon_template_nms_threshold,
        )

        detections: List[Detection] = []
        for index in self._flatten_indexes(selected_indexes):
            bbox = [float(value) for value in boxes[index]]
            detections.append(
                Detection(
                    detect_type=self.DETECT_TYPE_BY_LABEL.get(label, "poi_candidate"),
                    confidence=max(0.0, min(1.0, float(scores[index]))),
                    geom_px=self._bbox_to_polygon(bbox),
                    bbox_px=bbox,
                    label=label,
                )
            )

        return detections

    def _deduplicate_across_labels(self, detections: list[Detection]) -> list[Detection]:
        if not detections:
            return []

        candidates = [
            (
                detection,
                [int(round(value)) for value in detection.bbox_px],
                float(detection.confidence),
            )
            for detection in detections
            if detection.bbox_px is not None
        ]
        boxes = [candidate[1] for candidate in candidates]
        scores = [candidate[2] for candidate in candidates]
        selected_indexes = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=scores,
            score_threshold=0.0,
            nms_threshold=settings.icon_template_nms_threshold,
        )
        return [candidates[index][0] for index in self._flatten_indexes(selected_indexes)]

    def _threshold_for_label(self, label: str) -> float:
        return settings.icon_template_match_threshold

    @staticmethod
    def _flatten_indexes(indexes: cv2.typing.MatLike) -> list[int]:
        if indexes is None or len(indexes) == 0:
            return []
        return [int(index) for index in np.array(indexes).reshape(-1)]

    @staticmethod
    def _bbox_to_polygon(bbox: list[float]) -> dict[str, object]:
        x, y, width, height = bbox
        right = x + width
        bottom = y + height
        return {
            "type": "Polygon",
            "coordinates": [[
                [x, y],
                [right, y],
                [right, bottom],
                [x, bottom],
                [x, y],
            ]],
        }
