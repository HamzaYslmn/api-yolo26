"""Detection router — POST /yolo"""

import base64 as b64lib
from typing import Literal, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from yolo import detect_async, is_url, download_image_safely

router = APIRouter(tags=["Detection"])


@router.post("/yolo", summary="Detect objects")
async def detect(
    file: Optional[UploadFile] = File(None),
    data: Optional[str] = Form(None),
    confidence: float = Form(0.25, ge=0.10, le=1.0),
    format: Literal["json", "image"] = Form("json"),
):
    """
    Detect objects. Send `file` (upload) or `data` (base64/URL).
    Set `format=image` to get annotated JPEG.
    """
    # Get raw bytes
    if file:
        raw = await file.read()
    elif data:
        data = data.strip()
        if is_url(data):
            raw = await download_image_safely(data)
        else:
            try:
                raw = b64lib.b64decode(data.split(",")[-1])
            except Exception:
                raise HTTPException(400, "Invalid base64.")
    else:
        raise HTTPException(400, "Send file or data.")
    
    # Decode
    frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Invalid image.")
    
    # Detect
    detections, preview = await detect_async(frame, confidence, format == "image")
    
    if format == "image" and preview:
        return Response(content=preview, media_type="image/jpeg")
    
    return DetectionResult(count=len(detections), detections=detections)
