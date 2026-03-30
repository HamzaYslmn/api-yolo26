"""Detection router — POST /api/yolo"""

import base64
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from yolo import detect as yolo_detect

router = APIRouter(tags=["Detection"])


class DetectionResult(BaseModel):
    count: int
    detections: list[dict]


@router.post("/yolo", response_model=DetectionResult, summary="Detect objects in image")
async def detect(
    request: Request,
    file: Optional[UploadFile] = File(None, description="Image file (JPEG, PNG, etc.)"),
    confidence: float = Form(0.25, ge=0.0, le=1.0, description="Minimum confidence threshold"),
):
    """
    Detect objects in an image.
    
    Accepts either:
    - **Multipart form**: `file` (image) + optional `confidence`
    - **JSON body**: `{"image": "<base64>", "confidence": 0.25}`
    """
    content_type = request.headers.get("content-type", "")
    
    # Parse input
    if "application/json" in content_type:
        body = await request.json()
        image_str = body.get("image", "")
        confidence = body.get("confidence", 0.25)
        if not image_str:
            raise HTTPException(400, "Missing 'image' field.")
        if "," in image_str:
            image_str = image_str.split(",", 1)[1]
        try:
            raw = base64.b64decode(image_str)
        except Exception:
            raise HTTPException(400, "Invalid Base64.")
    else:
        if file is None:
            raise HTTPException(400, "No file provided.")
        raw = await file.read()
    
    # Decode image
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Invalid image.")
    
    # YOLO inference (blocking)
    results = yolo_detect(frame)
    detections = [d for d in results if d["confidence"] >= confidence]
    
    return DetectionResult(count=len(detections), detections=detections)
