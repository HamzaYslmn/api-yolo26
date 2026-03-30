"""Status router — GET /api/status"""

import os
import psutil
from pathlib import Path
from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter(tags=["Status"])


def _read_cgroup(v2_path: str, v1_path: str) -> int | None:
    """Read value from cgroups (v2 first, then v1)."""
    for path in [v2_path, v1_path]:
        p = Path(path)
        if p.exists():
            val = p.read_text().strip()
            if val not in ("max", ""):
                v = int(val)
                if v < 9223372036854771712:  # Not "unlimited"
                    return v
    return None


class Memory(BaseModel):
    limit_mb: float
    used_mb: float
    free_mb: float
    process_mb: float
    percent: float


class Status(BaseModel):
    ok: bool
    memory: Memory
    cpu_count: int


@router.get("/status", response_model=Status, summary="System status")
async def status():
    """Container/system memory and health status."""
    proc = psutil.Process(os.getpid())
    rss = proc.memory_info().rss
    
    # Try container limits (cgroups)
    limit = _read_cgroup("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes")
    used = _read_cgroup("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes")
    
    if limit and used:
        limit_mb = limit / (1024 * 1024)
        used_mb = used / (1024 * 1024)
    else:
        mem = psutil.virtual_memory()
        limit_mb = mem.total / (1024 * 1024)
        used_mb = mem.used / (1024 * 1024)
    
    free_mb = limit_mb - used_mb
    process_mb = rss / (1024 * 1024)
    percent = (used_mb / limit_mb * 100) if limit_mb > 0 else 0
    
    return Status(
        ok=percent < 90,
        memory=Memory(
            limit_mb=round(limit_mb, 1),
            used_mb=round(used_mb, 1),
            free_mb=round(free_mb, 1),
            process_mb=round(process_mb, 1),
            percent=round(percent, 1),
        ),
        cpu_count=psutil.cpu_count() or 1,
    )
