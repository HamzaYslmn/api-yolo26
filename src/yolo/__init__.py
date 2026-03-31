"""YOLO module — YOLO26n object detection (ONNX)."""

from .main import detect, detect_with_preview, detect_async
from .helpers import is_url, download_image_safely


def init():
    """Pre-load the ONNX model on startup."""
    from .main import _get_session
    _get_session()
