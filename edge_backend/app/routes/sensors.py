"""IoT Sensor Hub and ESP32 Sentry Node routes."""

import uuid
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
import httpx
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.config import settings
from app.database import get_db, async_session_factory
from app.models.db_models import SensorNodeModel, CameraModel, SecurityEventModel, SystemSetupModel
from app.models.schemas import (
    SensorConfigUpdate,
    SensorTelemetry,
    SensorNodeCreate,
    SensorNodeResponse,
    EventType,
    EventSeverity,
)
from app.services.clip_recorder import clip_recorder_service
from app.services.auth_service import auth_service, general_rate_limiter
from app.routes.events import ws_manager
from app.routes import ResilientRoute

logger = logging.getLogger("SensorRoutes")

router = APIRouter(
    dependencies=[Depends(auth_service.verify_api_access), Depends(general_rate_limiter)],
    route_class=ResilientRoute
)


async def is_system_armed(db: AsyncSession) -> bool:
    """Check if the security system is currently armed."""
    stmt = select(SystemSetupModel).where(SystemSetupModel.key == "system_armed")
    res = await db.execute(stmt)
    entry = res.scalar_one_or_none()
    if entry:
        return entry.value.lower() in ("true", "1", "armed", "yes")
    return True  # Armed by default in home security


async def record_sensor_incident_clip(event_id: str, camera_id: str):
    """Background task to record 15s incident clip and update event in DB."""
    try:
        import os
        is_testing = bool(os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("TESTING"))
        post_roll = 0 if is_testing else 15
        clip_url = await clip_recorder_service.record_event_clip(
            event_id=event_id,
            camera_id=camera_id,
            post_roll_seconds=post_roll
        )
        async with async_session_factory() as session:
            stmt = select(SecurityEventModel).where(SecurityEventModel.id == event_id)
            res = await session.execute(stmt)
            event = res.scalar_one_or_none()
            if event:
                event.clip_url = clip_url
                await session.commit()
                logger.info(f"Recorded 15s incident clip for sensor event {event_id}: {clip_url}")
    except Exception as e:
        logger.error(f"Failed to record 15s incident clip for sensor event {event_id}: {e}")


@router.get("", response_model=List[SensorNodeResponse])
@router.get("/", response_model=List[SensorNodeResponse])
async def list_sensors(db: AsyncSession = Depends(get_db)):
    """List all registered ESP32 sensor sentry nodes with recent telemetry states."""
    stmt = select(SensorNodeModel).order_by(desc(SensorNodeModel.updated_at))
    res = await db.execute(stmt)
    nodes = res.scalars().all()
    return [SensorNodeResponse.model_validate(n) for n in nodes]


@router.post("/register", response_model=SensorNodeResponse)
async def register_sensor_node(node_in: SensorNodeCreate, db: AsyncSession = Depends(get_db)):
    """Register or update an ESP32 sentry node in the SQLite database."""
    node: Optional[SensorNodeModel] = None

    if node_in.id:
        stmt = select(SensorNodeModel).where(SensorNodeModel.id == node_in.id)
        res = await db.execute(stmt)
        node = res.scalar_one_or_none()

    if not node and node_in.mac_address:
        stmt = select(SensorNodeModel).where(SensorNodeModel.mac_address == node_in.mac_address)
        res = await db.execute(stmt)
        node = res.scalar_one_or_none()

    now = datetime.utcnow()

    if node:
        node.name = node_in.name
        node.node_type = node_in.node_type
        node.ip_address = node_in.ip_address
        if node_in.mac_address:
            node.mac_address = node_in.mac_address
        if node_in.associated_camera_id:
            node.associated_camera_id = node_in.associated_camera_id
        if node_in.enabled_sensors is not None:
            node.enabled_sensors = node_in.enabled_sensors
        if node_in.sensor_states is not None:
            node.sensor_states = node_in.sensor_states
        node.last_heartbeat = now
        node.updated_at = now
    else:
        node_id = node_in.id or f"sentry_{uuid.uuid4().hex[:8]}"
        default_sensors = {
            "pir_enabled": True,
            "ultrasonic_enabled": True,
            "door1_enabled": True,
            "door2_enabled": True,
            "distance_threshold_cm": 50.0,
        }
        node = SensorNodeModel(
            id=node_id,
            name=node_in.name,
            node_type=node_in.node_type,
            ip_address=node_in.ip_address,
            mac_address=node_in.mac_address,
            associated_camera_id=node_in.associated_camera_id,
            enabled_sensors=node_in.enabled_sensors if node_in.enabled_sensors is not None else default_sensors,
            sensor_states=node_in.sensor_states if node_in.sensor_states is not None else {},
            last_heartbeat=now,
            created_at=now,
            updated_at=now,
        )
        db.add(node)

    await db.commit()
    await db.refresh(node)
    logger.info(f"Registered ESP32 sensor sentry node: {node.id} ({node.name}) at {node.ip_address}")
    return SensorNodeResponse.model_validate(node)


@router.post("/{node_id}/telemetry")
async def ingest_sensor_telemetry(
    node_id: str,
    telemetry: SensorTelemetry,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Ingest sensor readings, update SensorNodeModel state, broadcast to WebSocket, and trigger automated 15s incident clip if door opened or PIR triggered while armed."""
    stmt = select(SensorNodeModel).where(SensorNodeModel.id == node_id)
    res = await db.execute(stmt)
    node = res.scalar_one_or_none()
    if not node:
        raise HTTPException(status_code=404, detail=f"Sensor node {node_id} not found")

    now = datetime.utcnow()
    telemetry_data = telemetry.model_dump() if hasattr(telemetry, "model_dump") else telemetry.dict()
    # Convert datetimes to isoformat strings for JSON storage
    if isinstance(telemetry_data.get("timestamp"), datetime):
        telemetry_data["timestamp"] = telemetry_data["timestamp"].isoformat()

    node.sensor_states = telemetry_data
    node.last_heartbeat = now
    node.updated_at = now
    if telemetry.camera_id and not node.associated_camera_id:
        node.associated_camera_id = telemetry.camera_id

    await db.commit()
    await db.refresh(node)

    # Broadcast to WebSocket (/api/v1/events/ws)
    try:
        ws_payload = {
            "type": "SENSOR_TELEMETRY",
            "node_id": node_id,
            "camera_id": node.associated_camera_id or telemetry.camera_id,
            "telemetry": telemetry_data,
            "timestamp": now.isoformat()
        }
        background_tasks.add_task(ws_manager.broadcast_event, ws_payload)
    except Exception as e:
        logger.warning(f"Could not broadcast telemetry to WebSocket: {e}")

    # Check automated 15s incident clip condition: door opened or PIR triggered while armed
    armed = await is_system_armed(db)
    door_opened = bool(telemetry.door1_open or telemetry.door2_open)
    pir_triggered = bool(telemetry.pir_motion)

    incident_triggered = False
    event_id = None

    if armed and (door_opened or pir_triggered):
        cam_id = node.associated_camera_id or telemetry.camera_id or "cam_living_room"
        event_id = f"evt_sensor_{uuid.uuid4().hex[:8]}"
        event_type = EventType.DOOR_LEFT_OPEN.value if door_opened else EventType.INTRUSION_DETECTED.value

        snapshot_url = clip_recorder_service.save_snapshot(cam_id, event_id)

        db_event = SecurityEventModel(
            id=event_id,
            camera_id=cam_id,
            camera_name=f"Sentry ({node.name})",
            location="Monitored Sensor Zone",
            event_type=event_type,
            severity=EventSeverity.CRITICAL.value,
            confidence=1.0,
            timestamp=now,
            snapshot_url=snapshot_url,
            clip_url=None,
            bounding_box=None,
            keypoints=None,
            kinematics=None,
            metadata_json=telemetry_data,
            acknowledged=False,
        )
        db.add(db_event)
        await db.commit()

        # Broadcast Security Event via WebSocket
        try:
            alert_ws_payload = {
                "type": "SECURITY_ALERT",
                "event_id": event_id,
                "event_type": event_type,
                "camera_id": cam_id,
                "severity": EventSeverity.CRITICAL.value,
                "snapshot_url": snapshot_url,
                "timestamp": now.isoformat()
            }
            background_tasks.add_task(ws_manager.broadcast_event, alert_ws_payload)
        except Exception as e:
            logger.warning(f"Could not broadcast sensor alert to WebSocket: {e}")

        # Trigger automated 15s incident clip recording
        background_tasks.add_task(record_sensor_incident_clip, event_id, cam_id)
        incident_triggered = True
        logger.warning(
            f"[SENSOR INCIDENT TRIGGERED] Node {node_id} (Door={door_opened}, PIR={pir_triggered}) while ARMED. "
            f"Recording 15s incident clip for camera {cam_id} (Event {event_id})."
        )

    return {
        "status": "success",
        "node_id": node_id,
        "incident_triggered": incident_triggered,
        "event_id": event_id,
        "armed": armed,
        "telemetry": telemetry_data,
    }


@router.put("/{node_id}/toggles", response_model=SensorNodeResponse)
async def update_sensor_toggles(
    node_id: str,
    toggles: SensorConfigUpdate,
    db: AsyncSession = Depends(get_db)
):
    """Update sensor toggles and forward config update to the ESP32 node via HTTP POST."""
    stmt = select(SensorNodeModel).where(SensorNodeModel.id == node_id)
    res = await db.execute(stmt)
    node = res.scalar_one_or_none()
    if not node:
        raise HTTPException(status_code=404, detail=f"Sensor node {node_id} not found")

    toggles_dict = toggles.model_dump(exclude_unset=True) if hasattr(toggles, "model_dump") else toggles.dict(exclude_unset=True)

    current_config = dict(node.enabled_sensors or {})
    current_config.update(toggles_dict)

    node.enabled_sensors = current_config
    node.updated_at = datetime.utcnow()
    await db.commit()
    await db.refresh(node)

    # Forward config update to the ESP32 node via HTTP POST
    import os
    is_testing = bool(os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("TESTING"))
    if not is_testing:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.post(
                    f"http://{node.ip_address}/config",
                    json=current_config
                )
                logger.info(f"Forwarded config update to ESP32 node {node_id} at {node.ip_address}: status={resp.status_code}")
        except Exception as e:
            logger.warning(f"Could not forward config to ESP32 node {node_id} at {node.ip_address}: {e}")

    return SensorNodeResponse.model_validate(node)


@router.post("/arm")
async def arm_system(db: AsyncSession = Depends(get_db)):
    """Arm the security system for sensor tripwire/PIR incident clip triggers."""
    stmt = select(SystemSetupModel).where(SystemSetupModel.key == "system_armed")
    res = await db.execute(stmt)
    entry = res.scalar_one_or_none()
    if entry:
        entry.value = "true"
    else:
        db.add(SystemSetupModel(key="system_armed", value="true"))
    await db.commit()
    return {"status": "success", "armed": True}


@router.post("/disarm")
async def disarm_system(db: AsyncSession = Depends(get_db)):
    """Disarm the security system."""
    stmt = select(SystemSetupModel).where(SystemSetupModel.key == "system_armed")
    res = await db.execute(stmt)
    entry = res.scalar_one_or_none()
    if entry:
        entry.value = "false"
    else:
        db.add(SystemSetupModel(key="system_armed", value="false"))
    await db.commit()
    return {"status": "success", "armed": False}
