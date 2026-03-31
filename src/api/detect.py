"""Detection router — POST /api/yolo"""

import asyncio
import base64 as b64lib
from functools import partial
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from yolo import detect as yolo_detect

router = APIRouter(tags=["Detection"])


class DetectionResult(BaseModel):
    count: int
    detections: list[dict]
    preview_image: Optional[str] = None


def _detect_sync(frame: np.ndarray, confidence: float, preview: bool = False) -> tuple[list[dict], Optional[str]]:
    """Sync YOLO detection (runs in thread)."""
    results = yolo_detect(frame)
    valid_detections = [d for d in results if d["confidence"] >= confidence]
    
    preview_b64 = None
    if preview:
        # MARK: Draw bounding boxes for preview response
        annotated = frame.copy()
        for d in valid_detections:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            label = f"{d['class']} {d['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, max(y1 - 10, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        _, buffer = cv2.imencode(".jpg", annotated)
        preview_b64 = "data:image/jpeg;base64," + b64lib.b64encode(buffer).decode("utf-8")
        
    return valid_detections, preview_b64


@router.post("/yolo", response_model=DetectionResult, summary="Detect objects")
async def detect(
    file: Optional[UploadFile] = File(None),
    base64: Optional[str] = Form(None),
    confidence: float = Form(0.25, ge=0.25, le=1.0),
    preview: bool = Form(False),
):
    """
    Detect objects. Provide either `file` (image upload) or `base64` (encoded image).
    """
    if file is not None:
        raw = await file.read()
    elif base64 is not None:
        img = base64
        if "," in img:
            img = img.split(",", 1)[1]
        try:
            raw = b64lib.b64decode(img)
        except Exception:
            raise HTTPException(400, "Invalid Base64.")
    else:
        raise HTTPException(400, "Provide file or base64.")
    
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Invalid image.")
    
    # Run YOLO in thread to not block event loop
    loop = asyncio.get_running_loop()
    detections, preview_img = await loop.run_in_executor(
        None, partial(_detect_sync, frame, confidence, preview)
    )
    
    return DetectionResult(
        count=len(detections), 
        detections=detections, 
        preview_image=preview_img
    )
