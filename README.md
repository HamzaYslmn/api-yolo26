# api-yolo26

Free API test for YOLO26s object detection.

---

## Render.com Deployment

### Start Command

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

> Run this command from the `src/` directory.

### Render Configuration

| Setting              | Value                                                                                       |
| -------------------- | ------------------------------------------------------------------------------------------- |
| **Root Directory**   | `src`                                                                                       |
| **Build Command**    | `pip install uv && uv sync`                                                                 |
| **Start Command**    | `uvicorn main:app --host 0.0.0.0 --port $PORT`                                              |
| **Instance Type**    | Choose based on your needs (Free tier works for testing)                                   |
| **Python Version**   | 3.10+                                                                                       |

### Environment Variables (if needed)

| Variable | Description             |
| -------- | ----------------------- |
| `PORT`   | Automatically set by Render |

---

## Local Development

```bash
cd src
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

---

## API Documentation

Once deployed, access the docs at:

- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
