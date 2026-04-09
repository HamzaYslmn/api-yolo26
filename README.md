# 🚀 api-yolo26s (YOLO26s + ONNX Runtime)

### 🔗 Live API Documentation
**[https://api-yolo26-5n7z.onrender.com/api/docs](https://api-yolo26-5n7z.onrender.com/api/docs)**

[![API Docs](https://img.shields.io/badge/API-Docs-blue?style=for-the-badge&logo=fastapi)](https://api-yolo26-5n7z.onrender.com/api/docs)

---

FastAPI + YOLO26s object detection API using ONNX Runtime (CPU-optimized, ~150-200 MB memory footprint).

**Features:**
- 🧠 YOLO26s model via ONNX Runtime (~36 MB)
- 📥 Multi-format input: file upload, base64, or URL
- 🖼️ JSON or annotated JPEG output
- 🔒 Secure URL downloads (5 MB limit, streaming, timeout protection)
- 💾 Lightweight: ~150-200 MB RAM (vs ~550 MB with PyTorch)

---

## 🛠️ Model Conversion

Use `convert_model.py` to download and convert any YOLO model:

```bash
# Convert YOLO26s (default)
uv run python convert_model.py yolo26s

# Convert other models
uv run python convert_model.py yolo11n
uv run python convert_model.py yolo8m --imgsz 640
```

The script will:
1. Download the `.pt` model from Ultralytics
2. Export to ONNX format (optimized, simplified)
3. Save both files to `src/yolo/`

---

## 📦 Local Development

```bash
cd src
uv sync
uv run uvicorn main:app --host 127.0.0.1 --port 8080 --reload
```

Visit: `http://127.0.0.1:8080/api/docs`

---

## ☁️ Render.com Deployment

### Configuration

| Setting              | Value                                                      |
| -------------------- | ---------------------------------------------------------- |
| **Root Directory**   | `src`                                                      |
| **Build Command**    | `pip install uv && uv sync --active`                       |
| **Start Command**    | `uv run --active uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type**    | Free (512 MB) or Starter (1 GB recommended)                |
| **Python Version**   | 3.10+                                                      |

### Environment Variables

| Variable | Description                 |
| -------- | --------------------------- |
| `PORT`   | Automatically set by Render |

---

## 🌐 API Endpoints

| Method | Endpoint      | Description                              |
| ------ | ------------- | ---------------------------------------- |
| POST   | `/api/yolo`   | Detect objects (file/base64/URL)         |
| GET    | `/api/status` | System RAM & CPU status                  |

### POST `/api/yolo`

**Parameters:**
- `file`: (optional) Multipart file upload
- `data`: (optional) Base64 string or HTTP/HTTPS URL
- `confidence`: (default: 0.25) Min confidence threshold (0.1-1.0)
- `format`: (default: "json") Output format: `json` or `image`

**Response (format=json):**
```json
{
  "count": 3,
  "detections": [
    {
      "class": "bicycle",
      "confidence": 0.9598,
      "bbox": [83.4, 84.3, 380.4, 281.0]
    },
    ...
  ]
}
```

**Response (format=image):**
- Returns annotated JPEG image (Content-Type: image/jpeg)

---

## 📚 API Documentation

- **Swagger UI**: `/api/docs`
- **ReDoc**: `/api/redoc`
