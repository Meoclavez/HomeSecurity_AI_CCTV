"""Camera management and snapshot routes backed by SQLite persistence."""

import cv2
import numpy as np
from typing import List
from fastapi import APIRouter, HTTPException, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.config import settings
from app.database import get_db
from app.models.db_models import CameraModel
from app.models.schemas import CameraFeed, CameraListResponse, CameraStatus, DeviceTokenRegistration
from app.services.video_ingest_service import video_ingest_service
from app.services.notification_service import notification_service

router = APIRouter(prefix="/api/v1/cameras", tags=["Cameras"])


@router.get("", response_model=CameraListResponse)
async def list_cameras(db: AsyncSession = Depends(get_db)):
    """Retrieve all configured camera feeds with status and stream URLs from database."""
    stmt = select(CameraModel)
    res = await db.execute(stmt)
    db_cameras = res.scalars().all()

    feeds = [
        CameraFeed(
            id=c.id,
            name=c.name,
            location=c.location,
            rtsp_url="",  # Mask RTSP credentials from mobile API responses for security
            webrtc_url=c.webrtc_url,
            status=CameraStatus(c.status),
            fps=c.fps,
            resolution=c.resolution,
            is_ai_enabled=c.is_ai_enabled,
            ai_models=c.ai_models or [],
            last_seen=c.last_seen
        )
        for c in db_cameras
    ]
    return CameraListResponse(cameras=feeds, total=len(feeds))


@router.get("/{camera_id}", response_model=CameraFeed)
async def get_camera(camera_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieve details for a specific camera."""
    stmt = select(CameraModel).where(CameraModel.id == camera_id)
    res = await db.execute(stmt)
    cam = res.scalar_one_or_none()
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    return CameraFeed(
        id=cam.id,
        name=cam.name,
        location=cam.location,
        rtsp_url="",
        webrtc_url=cam.webrtc_url,
        status=CameraStatus(cam.status),
        fps=cam.fps,
        resolution=cam.resolution,
        is_ai_enabled=cam.is_ai_enabled,
        ai_models=cam.ai_models or [],
        last_seen=cam.last_seen
    )


@router.get("/{camera_id}/snapshot")
async def get_camera_snapshot(camera_id: str):
    """Retrieve the latest real-time frame snapshot as a JPEG."""
    frame = video_ingest_service.get_latest_frame(camera_id)
    if frame is None:
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(frame, f"LIVE: {camera_id.upper()}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 128), 2)

    ret, encoded_img = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return Response(content=encoded_img.tobytes(), media_type="image/jpeg")


@router.post("/register-device")
async def register_device_token(device_reg: DeviceTokenRegistration):
    """Register mobile device token (FCM / APNs) with SQLite persistence for emergency push alerts."""
    await notification_service.register_device(device_reg)
    return {"status": "success", "message": f"Persisted {device_reg.platform} device token."}
