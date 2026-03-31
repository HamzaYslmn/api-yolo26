"""YOLO26n object detection module."""

import asyncio
import base64 as b64lib
import os
from typing import Optional

import cv2
import numpy as np

cv2.setUseOptimized(True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolo26n.pt")
_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        try:
            import torch
            torch.set_num_threads(2)
            torch.set_num_interop_threads(1)
        except Exception:
            pass
        _model = YOLO(MODEL_PATH)
        _model.to("cpu")
    return _model


def detect(frame: np.ndarray) -> list[dict]:
    """Run inference and return all detections with class, confidence, and bbox (xyxy)."""
    model = _get_model()
    results = model(frame, verbose=False)
    names = getattr(model, "names", {})
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class": names.get(int(box.cls[0]), "unknown"),
            "confidence": round(float(box.conf[0]), 4),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })
    return detections


def detect_with_preview(
    frame: np.ndarray, 
    confidence: float, 
    preview: bool = False
) -> tuple[list[dict], Optional[str]]:
    """
    Run YOLO detection with optional annotated preview.
    
    This is a blocking/sync function — call via asyncio.to_thread().
    """
    results = detect(frame)
    valid = [d for d in results if d["confidence"] >= confidence]
    
    preview_b64 = None
    if preview:
        annotated = frame.copy()
        for d in valid:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            label = f"{d['class']} {d['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, max(y1 - 10, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        _, buffer = cv2.imencode(".jpg", annotated)
        preview_b64 = "data:image/jpeg;base64," + b64lib.b64encode(buffer).decode()
        
    return valid, preview_b64


async def detect_async(
    frame: np.ndarray, 
    confidence: float, 
    preview: bool = False
) -> tuple[list[dict], Optional[str]]:
    """Async wrapper — runs detection in thread pool, non-blocking."""
    return await asyncio.to_thread(detect_with_preview, frame, confidence, preview)