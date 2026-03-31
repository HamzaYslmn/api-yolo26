# 🚀 api-yolo26 (YOLO26n)

### 🔗 Live API Documentation
**[https://api-yolo26.onrender.com/docs](https://api-yolo26.onrender.com/docs)**

[![API Docs](https://img.shields.io/badge/API-Docs-blue?style=for-the-badge&logo=fastapi)](https://api-yolo26.onrender.com/docs)

---

Free API test for YOLO26n object detection (CPU-only).

---

## Render.com Deployment

### Start Command

```bash
uv run uvicorn main:app --host 0.0.0.0 --port $PORT
```

> Run this command from the `src/` directory.

### Render Configuration

| Setting              | Value                                                      |
| -------------------- | ---------------------------------------------------------- |
| **Root Directory**   | `src`                                                      |
| **Build Command**    | `pip install uv && uv sync --active`                       |
| **Start Command**    | `uv run --active uvicorn main:app --host 0.0.0.0 --port $PORT` |
| **Instance Type**    | Choose based on your needs (Free tier works for testing)   |
| **Python Version**   | 3.10+                                                      |

### Environment Variables (if needed)

| Variable | Description                 |
| -------- | --------------------------- |
| `PORT`   | Automatically set by Render |

---

## Local Development

```bash
cd src
uv sync
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Endpoints

| Method | Endpoint      | Description                        |
| ------ | ------------- | ---------------------------------- |
| POST   | `/api/yolo`   | Detect objects (multipart or JSON) |
| GET    | `/api/status` | System RAM & CPU status            |

---

## API Documentation

Once deployed locally, access the docs at:

- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
