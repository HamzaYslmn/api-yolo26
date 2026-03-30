"""Status router — GET /api/status (system info & RAM monitoring)"""

import os
import psutil
from pathlib import Path
from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter(tags=["Status"])


def _get_container_memory_limit() -> int | None:
    """Read container memory limit from cgroups (Linux containers)."""
    # cgroups v2
    cg2_limit = Path("/sys/fs/cgroup/memory.max")
    if cg2_limit.exists():
        val = cg2_limit.read_text().strip()
        if val != "max":
            return int(val)
    
    # cgroups v1
    cg1_limit = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if cg1_limit.exists():
        val = int(cg1_limit.read_text().strip())
        # Check if it's not set to "unlimited" (huge value)
        if val < 9223372036854771712:
            return val
    
    return None


def _get_container_memory_usage() -> int | None:
    """Read container memory usage from cgroups (Linux containers)."""
    # cgroups v2
    cg2_current = Path("/sys/fs/cgroup/memory.current")
    if cg2_current.exists():
        return int(cg2_current.read_text().strip())
    
    # cgroups v1
    cg1_usage = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
    if cg1_usage.exists():
        return int(cg1_usage.read_text().strip())
    
    return None


class MemoryInfo(BaseModel):
    limit_mb: float
    used_mb: float
    available_mb: float
    percent_used: float


class ProcessMemory(BaseModel):
    rss_mb: float  # Resident Set Size (actual RAM used)


class SystemStatus(BaseModel):
    memory: MemoryInfo
    process: ProcessMemory
    cpu_count: int
    cpu_percent: float


@router.get("/status", response_model=SystemStatus, summary="System RAM & CPU status")
async def get_status():
    """Get live container/system memory usage and process-specific RAM consumption."""
    proc = psutil.Process(os.getpid())
    proc_mem = proc.memory_info()
    
    # Try to get container limits first
    container_limit = _get_container_memory_limit()
    container_usage = _get_container_memory_usage()
    
    if container_limit and container_usage:
        # Running in a container with memory limits
        limit_mb = container_limit / (1024 * 1024)
        used_mb = container_usage / (1024 * 1024)
        available_mb = limit_mb - used_mb
        percent_used = (used_mb / limit_mb) * 100 if limit_mb > 0 else 0
    else:
        # Fallback to system memory
        mem = psutil.virtual_memory()
        limit_mb = mem.total / (1024 * 1024)
        used_mb = mem.used / (1024 * 1024)
        available_mb = mem.available / (1024 * 1024)
        percent_used = mem.percent

    return SystemStatus(
        memory=MemoryInfo(
            limit_mb=round(limit_mb, 1),
            used_mb=round(used_mb, 1),
            available_mb=round(available_mb, 1),
            percent_used=round(percent_used, 1),
        ),
        process=ProcessMemory(
            rss_mb=round(proc_mem.rss / (1024 * 1024), 1),
        ),
        cpu_count=psutil.cpu_count() or 1,
        cpu_percent=psutil.cpu_percent(interval=0.1),
    )
