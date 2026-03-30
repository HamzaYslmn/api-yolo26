"""Status router — GET /api/status (system info & RAM monitoring)"""

import os
import psutil
from pydantic import BaseModel
from fastapi import APIRouter

router = APIRouter(tags=["Status"])


class MemoryInfo(BaseModel):
    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float


class ProcessMemory(BaseModel):
    rss_mb: float  # Resident Set Size (actual RAM used)
    vms_mb: float  # Virtual Memory Size


class SystemStatus(BaseModel):
    memory: MemoryInfo
    process: ProcessMemory
    cpu_count: int
    cpu_percent: float


@router.get("/status", response_model=SystemStatus, summary="System RAM & CPU status")
async def get_status():
    """Get live system memory usage and process-specific RAM consumption."""
    mem = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    proc_mem = proc.memory_info()

    return SystemStatus(
        memory=MemoryInfo(
            total_mb=round(mem.total / (1024 * 1024), 1),
            available_mb=round(mem.available / (1024 * 1024), 1),
            used_mb=round(mem.used / (1024 * 1024), 1),
            percent_used=round(mem.percent, 1),
        ),
        process=ProcessMemory(
            rss_mb=round(proc_mem.rss / (1024 * 1024), 1),
            vms_mb=round(proc_mem.vms / (1024 * 1024), 1),
        ),
        cpu_count=psutil.cpu_count() or 1,
        cpu_percent=psutil.cpu_percent(interval=0.1),
    )
