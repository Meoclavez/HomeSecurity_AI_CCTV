"""Health and telemetry endpoint for Docker healthchecks and monitoring."""

import os
import shutil
from fastapi import APIRouter
from sqlalchemy import select, func
from app.config import settings
from app.database import async_session_factory
from app.models.db_models import CameraModel, SecurityEventModel
from app.services.hailo_inference_service import hailo_inference_service

router = APIRouter(prefix="/api/v1/health", tags=["Health & Telemetry"])


@router.get("")
async def health_check():
    """Returns detailed hardware, storage, and service telemetry."""
    # Check disk usage
    total, used, free = shutil.disk_usage(str(settings.STORAGE_DIR))
    used_pct = round((used / total) * 100, 1)

    # Check hardware nodes
    vaapi_ok = os.path.exists(settings.VAAPI_DEVICE)
    hailo_ok = os.path.exists(settings.HAILO_DEVICE)

    # Query counts from DB
    async with async_session_factory() as session:
        cam_count = await session.scalar(select(func.count(CameraModel.id)))
        event_count = await session.scalar(select(func.count(SecurityEventModel.id)))

    active_tracks = len(hailo_inference_service.kinematic_engine.tracks)

    return {
        "status": "healthy",
        "version": settings.VERSION,
        "hardware": {
            "vaapi_device": settings.VAAPI_DEVICE,
            "vaapi_available": vaapi_ok,
            "hailo_device": settings.HAILO_DEVICE,
            "hailo_available": hailo_ok or hailo_inference_service.device_available,
        },
        "storage": {
            "used_percent": used_pct,
            "free_gb": round(free / (1024**3), 2),
            "retention_days": settings.STORAGE_RETENTION_DAYS,
        },
        "telemetry": {
            "total_cameras": cam_count,
            "total_events": event_count,
            "active_person_tracks": active_tracks,
        }
    }
