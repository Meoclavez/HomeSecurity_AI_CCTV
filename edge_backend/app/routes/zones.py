"""Zone and privacy mask configuration API routes."""

from typing import List
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models.db_models import CameraModel
from app.models.schemas import ZoneConfig, MuteCameraRequest
from app.services.ai_zone_service import ai_zone_service
from datetime import datetime, timedelta
from app.services.auth_service import auth_service, general_rate_limiter

router = APIRouter(
    prefix="/api/v1/cameras",
    tags=["Camera Zones & Privacy"],
    dependencies=[Depends(auth_service.verify_api_access), Depends(general_rate_limiter)]
)


@router.get("/{camera_id}/zones", response_model=List[ZoneConfig])
async def get_camera_zones(camera_id: str, db: AsyncSession = Depends(get_db)):
    """Fetch configured privacy masks, intrusion polygons, and tripwires for a camera."""
    stmt = select(CameraModel).where(CameraModel.id == camera_id)
    res = await db.execute(stmt)
    cam = res.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    tracker = ai_zone_service.zone_trackers.get(camera_id)
    return list(tracker.zones.values()) if tracker else []


@router.post("/{camera_id}/zones", response_model=ZoneConfig)
async def create_or_update_camera_zone(
    camera_id: str,
    zone: ZoneConfig,
    db: AsyncSession = Depends(get_db)
):
    """Create or update a zone (Privacy mask, Tripwire line, or Intrusion polygon)."""
    stmt = select(CameraModel).where(CameraModel.id == camera_id)
    res = await db.execute(stmt)
    cam = res.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    zone.camera_id = camera_id
    tracker = ai_zone_service.zone_trackers.get(camera_id)
    existing_zones = list(tracker.zones.values()) if tracker else []

    updated = [z for z in existing_zones if z.id != zone.id] + [zone]
    ai_zone_service.set_camera_zones(camera_id, updated)
    return zone


@router.delete("/{camera_id}/zones/{zone_id}")
async def delete_camera_zone(
    camera_id: str,
    zone_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a zone configuration."""
    tracker = ai_zone_service.zone_trackers.get(camera_id)
    if not tracker or zone_id not in tracker.zones:
        raise HTTPException(status_code=404, detail="Zone not found")

    updated = [z for z in tracker.zones.values() if z.id != zone_id]
    ai_zone_service.set_camera_zones(camera_id, updated)
    return {"status": "success", "message": f"Deleted zone {zone_id}"}


@router.post("/{camera_id}/mute")
async def mute_camera_alerts(
    camera_id: str,
    req: MuteCameraRequest,
    db: AsyncSession = Depends(get_db)
):
    """Mute alerts for a camera feed for X minutes (invoked from mobile lockscreen action)."""
    stmt = select(CameraModel).where(CameraModel.id == camera_id)
    res = await db.execute(stmt)
    cam = res.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    cam.muted_until = datetime.utcnow() + timedelta(minutes=req.duration_minutes)
    await db.commit()
    return {"status": "success", "muted_until": cam.muted_until.isoformat()}
