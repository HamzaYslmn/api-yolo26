# YOLO Detection API

FastAPI + YOLO26s object detection service.

## Run

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Swagger UI: http://localhost:8000/docs

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/detect` | Multipart file upload |
| POST | `/api/detect/base64` | Base64-encoded image (JSON body) |

## Request Examples

**Multipart:**
```bash
curl -X POST http://localhost:8000/api/detect \
  -F "file=@image.jpg" \
  -F "confidence=0.25"
```

**Base64:**
```bash
curl -X POST http://localhost:8000/api/detect/base64 \
  -H "Content-Type: application/json" \
  -d '{"image": "<base64string>", "confidence": 0.25}'
```

## Response

```json
{
  "count": 2,
  "detections": [
    {
      "class": "person",
      "confidence": 0.9123,
      "bbox": { "x1": 10.0, "y1": 20.0, "x2": 200.0, "y2": 400.0 }
    }
  ]
}
```
