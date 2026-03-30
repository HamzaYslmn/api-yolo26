"""Detection router — POST /api/yolo"""

import base64 as b64lib
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


@router.post("/yolo", response_model=DetectionResult, summary="Detect objects")
async def detect(
    file: Optional[UploadFile] = File(None),
    base64: Optional[str] = Form(None),
    confidence: float = Form(0.25, ge=0.25, le=1.0),
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
    
    results = yolo_detect(frame)
    detections = [d for d in results if d["confidence"] >= confidence]
    return DetectionResult(count=len(detections), detections=detections)
