"""YOLO module — YOLO26n object detection."""

from .main import detect, detect_with_preview, detect_async
from .helpers import is_url, download_image_safely


def init():
    """Pre-load the YOLO model on startup (warm up)."""
    from .main import _get_model
    _get_model()
