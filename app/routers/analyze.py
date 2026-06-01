"""Floorplan analysis API router."""

import logging
import time

from fastapi import APIRouter, HTTPException

from app.core.image_loader import cleanup_image, download_image
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.services.graph_detector import GraphDetector
from app.services.object_detector import ObjectDetector
from app.services.poi_detector import PoiDetector
from app.services.structure_detector import StructureDetector
from app.services.text_detector import TextDetector

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1",
    tags=["analyze"],
)

text_detector = TextDetector()
object_detector = ObjectDetector()
poi_detector = PoiDetector()
structure_detector = StructureDetector()
graph_detector = GraphDetector()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze floorplan image",
    description="Download and analyze a floorplan image from S3, MinIO, or HTTP.",
)
def analyze_floorplan(request: AnalyzeRequest) -> AnalyzeResponse:
    """Run all detectors and return their combined results."""
    start_time = time.time()
    local_path = None

    try:
        try:
            local_path = download_image(request.image_url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except IOError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Image fetch failed: {exc}",
            )

        objects = object_detector.detect(local_path)
        texts = text_detector.detect(
            local_path,
            object_detections=objects,
        )
        pois = poi_detector.detect(
            local_path,
            text_detections=texts,
            object_detections=objects,
        )
        structures = structure_detector.detect(
            local_path,
            text_detections=texts,
            object_detections=objects,
        )
        graph = graph_detector.detect(
            local_path,
            object_detections=objects,
            structure_detections=structures,
            poi_detections=pois,
        )

        elapsed_ms = int((time.time() - start_time) * 1000)
        return AnalyzeResponse(
            floorplan_id=request.floorplan_id,
            model_version=(
                request.options.model_version
                if request.options
                else "v1.0"
            ),
            processing_time_ms=elapsed_ms,
            detections=texts + objects + pois + structures + graph,
        )
    finally:
        if local_path:
            cleanup_image(local_path)
