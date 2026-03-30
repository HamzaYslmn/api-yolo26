"""YOLO26s object detection module."""

import os
import cv2

cv2.setUseOptimized(True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolo26s.pt")
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


def detect(frame) -> list[dict]:
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