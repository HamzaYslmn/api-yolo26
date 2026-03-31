"""YOLO26s object detection module (ONNX Runtime)."""

import asyncio
import os
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

cv2.setUseOptimized(True)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "yolo26s.onnx")
_session: Optional[ort.InferenceSession] = None

# COCO class names
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]


def _get_session() -> ort.InferenceSession:
    """Load ONNX model (lazy, singleton)."""
    global _session
    if _session is None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(MODEL_PATH, opts, providers=["CPUExecutionProvider"])
    return _session


def _preprocess(frame: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[float, float]]:
    """Resize and normalize image for YOLO input (640x640)."""
    h, w = frame.shape[:2]
    scale = min(640 / w, 640 / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(frame, (new_w, new_h))
    padded = np.full((640, 640, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = resized
    
    # BCHW, normalized
    blob = padded.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
    
    return blob, (new_w, new_h), (scale, scale)


def detect(frame: np.ndarray) -> list[dict]:
    """Run ONNX inference and return detections."""
    session = _get_session()
    blob, (new_w, new_h), (sx, sy) = _preprocess(frame)
    
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    preds = outputs[0][0]  # Shape: (300, 6) -> [x1, y1, x2, y2, conf, cls]
    
    detections = []
    for pred in preds:
        x1, y1, x2, y2, conf, cls_id = pred
        if conf < 0.01:  # Skip very low confidence
            continue
            
        # Scale back to original
        x1, x2 = float(x1 / sx), float(x2 / sx)
        y1, y2 = float(y1 / sy), float(y2 / sy)
        
        cls_idx = int(cls_id)
        cls_name = COCO_NAMES[cls_idx] if cls_idx < len(COCO_NAMES) else "unknown"
        
        detections.append({
            "class": cls_name,
            "confidence": round(float(conf), 4),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })
    
    return detections


def detect_with_preview(
    frame: np.ndarray, 
    confidence: float, 
    preview: bool = False
) -> tuple[list[dict], Optional[bytes]]:
    """Run detection with optional annotated preview."""
    results = detect(frame)
    valid = [d for d in results if d["confidence"] >= confidence]
    
    preview_bytes = None
    if preview:
        annotated = frame.copy()
        for d in valid:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            label = f"{d['class']} {d['confidence']:.2f}"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, max(y1 - 10, 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        ok, buffer = cv2.imencode(".jpg", annotated)
        if ok:
            preview_bytes = bytes(buffer)
        
    return valid, preview_bytes


async def detect_async(
    frame: np.ndarray, 
    confidence: float, 
    preview: bool = False
) -> tuple[list[dict], Optional[bytes]]:
    """Async wrapper — runs detection in thread pool."""
    return await asyncio.to_thread(detect_with_preview, frame, confidence, preview)