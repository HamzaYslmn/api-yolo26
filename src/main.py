"""FastAPI entry point — YOLO detection API."""

from contextlib import asynccontextmanager
from pathlib import Path
import importlib
import logging

from fastapi import FastAPI

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_SRC = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    from yolo import init
    log.info("Loading YOLO model...")
    init()
    log.info("Model ready — server is up")
    yield
    log.info("Shutdown complete")


app = FastAPI(
    title="YOLO API",
    description="by HamzaYslmn",
    version="1.0.0",
    lifespan=lifespan,
    root_path="/api",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/", include_in_schema=False)
async def root():
    """Health check for Render."""
    return {"status": "ok"}


def _include_routers(directory: str, prefix: str):
    """Auto-discover and register routers from a directory."""
    for py in sorted((_SRC / directory).rglob("*.py")):
        if py.name.startswith("_"):
            continue
        mod_name = ".".join(py.relative_to(_SRC).with_suffix("").parts)
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "router"):
                app.include_router(mod.router, prefix=prefix)
                log.info("Registered router: %s → %s", mod_name, prefix)
        except Exception as e:
            log.error("Router load error %s: %s", mod_name, e)


_include_routers("api", "")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)