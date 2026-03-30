"""Detection router — POST /api/detect"""

import base64
import io
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from yolo import detect as yolo_detect

router = APIRouter(tags=["Detection"])


class Base64Request(BaseModel):
    image: str  # base64-encoded image (with or without data URI prefix)
    confidence: Optional[float] = 0.25


class DetectionResult(BaseModel):
    count: int
    detections: list[dict]


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image. Ensure it is a valid JPEG/PNG.")
    return frame


@router.post("/yolo", response_model=DetectionResult, summary="Detect objects (multipart)")
async def detect_multipart(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, etc.)"),
    confidence: float = Form(0.25, ge=0.0, le=1.0, description="Minimum confidence threshold"),
):
    """Detect objects in an uploaded image file."""
    raw = await file.read()
    frame = _decode_image(raw)
    detections = _run_detect(frame, confidence)
    return DetectionResult(count=len(detections), detections=detections)


@router.post("/yolo/base64", response_model=DetectionResult, summary="Detect objects (Base64)")
async def detect_base64(body: Base64Request):
    """Detect objects in a Base64-encoded image."""
    image_str = body.image
    # Strip data URI prefix if present (e.g. "data:image/jpeg;base64,...")
    if "," in image_str:
        image_str = image_str.split(",", 1)[1]
    try:
        raw = base64.b64decode(image_str)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 string.")
    frame = _decode_image(raw)
    detections = _run_detect(frame, body.confidence)
    return DetectionResult(count=len(detections), detections=detections)


def _run_detect(frame: np.ndarray, confidence_threshold: float) -> list[dict]:
    results = yolo_detect(frame)
    return [d for d in results if d["confidence"] >= confidence_threshold]
