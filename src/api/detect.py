"""Detection router — POST /api/yolo"""

import base64 as b64lib
from typing import Optional, Union

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from yolo import detect_async, is_url, download_image_safely

router = APIRouter(tags=["Detection"])


class DetectionResult(BaseModel):
    count: int
    detections: list[dict]
    preview_image: Optional[str] = None


@router.post("/yolo", response_model=DetectionResult, summary="Detect objects")
async def detect(
    file: Optional[UploadFile] = File(None, description="Image file upload"),
    data: Optional[str] = Form(None, description="Image as base64 string or URL (http/https)"),
    confidence: float = Form(0.25, ge=0.10, le=1.0),
    preview: bool = Form(False),
):
    """
    Detect objects in an image.
    
    Input options (provide one):
    - `file`: Multipart file upload
    - `data`: Base64 string (with/without data URI) or HTTP/HTTPS URL (max 5 MB)
    """
    raw: bytes
    
    if file is not None:
        raw = await file.read()
    elif data is not None:
        data = data.strip()
        if is_url(data):
            raw = await download_image_safely(data)
        else:
            img = data.split(",", 1)[-1] if "," in data else data
            try:
                raw = b64lib.b64decode(img)
            except Exception:
                raise HTTPException(400, "Invalid input. Provide a valid base64 string or URL.")
    else:
        raise HTTPException(400, "Provide file upload or data (base64/URL).")
    
    # Decode image
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Invalid image.")
    
    # Run detection (non-blocking)
    detections, preview_img = await detect_async(frame, confidence, preview)
    
    return DetectionResult(
        count=len(detections), 
        detections=detections, 
        preview_image=preview_img
    )
