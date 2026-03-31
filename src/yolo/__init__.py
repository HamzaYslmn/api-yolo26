"""YOLO module — YOLO26n object detection."""

from .main import detect


def init():
    """Pre-load the YOLO model on startup (warm up)."""
    from .main import _get_model
    _get_model()
