"""Helper utilities for YOLO detection API."""

from urllib.parse import urlparse

import httpx
from fastapi import HTTPException

MAX_URL_DOWNLOAD_SIZE = 5 * 1024 * 1024  # 5 MB
DOWNLOAD_TIMEOUT = 10.0  # seconds
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp"}


def is_url(value: str) -> bool:
    """Check if value looks like a URL."""
    try:
        parsed = urlparse(value.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


async def download_image_safely(url: str) -> bytes:
    """
    Download image from URL with safety measures:
    - Strict size limit (5 MB)
    - Streaming download to prevent memory bombs
    - Content-Type validation
    - Timeout protection
    """
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(DOWNLOAD_TIMEOUT),
            follow_redirects=True,
            max_redirects=3,
        ) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()

                # Validate content type
                content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
                if content_type and content_type not in ALLOWED_CONTENT_TYPES:
                    raise HTTPException(400, f"Invalid content type: {content_type}. Expected an image.")

                # Check Content-Length header if available
                content_length = response.headers.get("content-length")
                if content_length and int(content_length) > MAX_URL_DOWNLOAD_SIZE:
                    raise HTTPException(413, f"Image too large. Max size: {MAX_URL_DOWNLOAD_SIZE // (1024*1024)} MB.")

                # Stream download with size check to prevent ZIP bombs / buffer overflow
                chunks = []
                total_size = 0
                async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
                    total_size += len(chunk)
                    if total_size > MAX_URL_DOWNLOAD_SIZE:
                        raise HTTPException(413, f"Image too large. Max size: {MAX_URL_DOWNLOAD_SIZE // (1024*1024)} MB.")
                    chunks.append(chunk)

                return b"".join(chunks)

    except httpx.TimeoutException:
        raise HTTPException(504, "Timeout downloading image from URL.")
    except httpx.HTTPStatusError as e:
        raise HTTPException(400, f"Failed to download image: HTTP {e.response.status_code}")
    except httpx.RequestError as e:
        raise HTTPException(400, f"Failed to download image: {type(e).__name__}")
