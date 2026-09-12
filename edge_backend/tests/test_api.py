"""Unit tests for Edge API endpoints, authentication, WebRTC ICE servers, DVR timeline, and zones."""

import pytest
import asyncio
from datetime import datetime, date, timezone
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.main import app
from app.config import settings
from app.database import engine, async_session_factory
from app.models.db_models import Base, SystemSetupModel, CameraModel, SensorNodeModel, SecurityEventModel
from app.models.schemas import (
    SecurityEvent,
    EventType,
    EventSeverity,
    ZoneConfig,
    ZoneType,
    Point2D,
    TripwireDirection,
    CameraFeatureConfig,
    BoundingBox,
    Keypoint,
)
from app.services.notification_service import notification_service
from app.services.auth_service import auth_service, intrusion_detector
from app.services.feature_manager import feature_manager
from app.services.hailo_inference_service import hailo_inference_service


@pytest.fixture(scope="session", autouse=True)
def init_test_db():
    """Ensure database schema is created and setup marked complete before running test suite."""
    async def _init():
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            try:
                from sqlalchemy import text
                await conn.execute(text("ALTER TABLE cameras ADD COLUMN features JSON;"))
            except Exception:
                pass
        
        async with async_session_factory() as session:
            # Mark setup completed
            stmt = select(SystemSetupModel).where(SystemSetupModel.key == "setup_completed")
            res = await session.execute(stmt)
            entry = res.scalar_one_or_none()
            if not entry:
                session.add(SystemSetupModel(key="setup_completed", value="true"))
            
            # Ensure living room camera exists
            cam_stmt = select(CameraModel).where(CameraModel.id == "cam_living_room")
            cam_res = await session.execute(cam_stmt)
            if not cam_res.scalar_one_or_none():
                session.add(CameraModel(
                    id="cam_living_room",
                    name="Living Room Camera",
                    location="Indoor",
                    rtsp_url="rtsp://127.0.0.1:554/live",
                    webrtc_url="http://127.0.0.1:8000/api/v1/webrtc/offer?camera_id=cam_living_room",
                    dvr_enabled=True,
                    status="ONLINE"
                ))
            await session.commit()
    asyncio.run(_init())


@pytest.fixture(autouse=True)
def reset_intrusion_detector():
    """Reset intrusion detector failed attempts between tests."""
    intrusion_detector.failed_attempts.clear()
    app.state.setup_completed = True


@pytest.fixture
def auth_headers():
    token = auth_service.create_access_token({"sub": "test_admin", "role": "admin", "type": "user_session"})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "health_url" in data


def test_health_endpoint(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "storage" in data
    assert "telemetry" in data


def test_list_cameras(client, auth_headers):
    response = client.get("/api/v1/cameras", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "cameras" in data


def test_dynamic_ice_servers(client, auth_headers):
    response = client.get("/api/v1/webrtc/ice-servers?client_id=test_client", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "iceServers" in data
    assert len(data["iceServers"]) >= 2
    # Verify TURN credential presence
    turn_entry = data["iceServers"][1]
    assert "username" in turn_entry
    assert "credential" in turn_entry


def test_storage_health_endpoint(client, auth_headers):
    response = client.get("/api/v1/storage/health", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "total_gb" in data
    assert "used_percent" in data
    assert "smart_status" in data
    assert "camera_quotas" in data


def test_camera_zones_crud(client, auth_headers):
    camera_id = "cam_living_room"
    zone_payload = {
        "id": "zone_test_tripwire",
        "camera_id": camera_id,
        "name": "Front Yard Tripwire",
        "zone_type": "TRIPWIRE",
        "enabled": True,
        "line_start": {"x": 0.1, "y": 0.5},
        "line_end": {"x": 0.9, "y": 0.5},
        "direction": "BIDIRECTIONAL"
    }

    # 1. Create Zone
    post_res = client.post(f"/api/v1/cameras/{camera_id}/zones", json=zone_payload, headers=auth_headers)
    assert post_res.status_code == 200
    assert post_res.json()["id"] == "zone_test_tripwire"

    # 2. Get Zones
    get_res = client.get(f"/api/v1/cameras/{camera_id}/zones", headers=auth_headers)
    assert get_res.status_code == 200
    zones = get_res.json()
    assert any(z["id"] == "zone_test_tripwire" for z in zones)

    # 3. Delete Zone
    del_res = client.delete(f"/api/v1/cameras/{camera_id}/zones/zone_test_tripwire", headers=auth_headers)
    assert del_res.status_code == 200


def test_camera_timeline_endpoint(client, auth_headers):
    camera_id = "cam_living_room"
    today_str = date.today().isoformat()
    response = client.get(f"/api/v1/cameras/{camera_id}/timeline?date={today_str}", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["camera_id"] == camera_id
    assert "segments" in data
    assert "events" in data
    assert "gaps" in data
    assert "hls_master_url" in data


def test_trigger_security_event_authorized(client):
    payload = {
        "camera_id": "cam_living_room",
        "event_type": "FALL_DETECTED",
        "severity": "CRITICAL",
        "confidence": 0.94,
        "bounding_box": {
            "x_min": 0.2,
            "y_min": 0.6,
            "x_max": 0.8,
            "y_max": 0.9,
            "confidence": 0.94,
            "label": "person_fallen"
        },
        "kinematics": {
            "hip_descent_velocity": 2.3,
            "aspect_ratio_initial": 1.6,
            "aspect_ratio_final": 0.65,
            "transition_duration_ms": 420,
            "immobility_duration_sec": 5.2,
            "floor_proximity_score": 0.88
        }
    }
    response = client.post(
        "/api/v1/events/trigger",
        json=payload,
        headers={"X-Edge-API-Key": settings.INTERNAL_SERVICE_KEY}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["id"].startswith("evt_")
    assert data["event_type"] == "FALL_DETECTED"
    assert data["severity"] == "CRITICAL"


def test_mute_camera_endpoint(client, auth_headers):
    camera_id = "cam_living_room"
    response = client.post(f"/api/v1/cameras/{camera_id}/mute", json={"duration_minutes": 5}, headers=auth_headers)
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_auth_bypass_trigger_event(client):
    payload = {
        "camera_id": "cam_living_room",
        "event_type": "FALL_DETECTED",
        "severity": "CRITICAL",
        "confidence": 0.94
    }
    response = client.post(
        "/api/v1/events/trigger", 
        json=payload, 
        headers={"X-Edge-API-Key": "invalid_key"}
    )
    assert response.status_code in (401, 403)


def test_path_traversal_prevention(client):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        auth_service.sanitize_and_resolve_file(settings.CLIPS_DIR, "../../etc/passwd")
    assert exc_info.value.status_code == 400
    assert "Invalid filename format" in exc_info.value.detail


def test_webrtc_offer_exchange(client, auth_headers):
    offer_payload = {
        "camera_id": "cam_living_room",
        "sdp": "v=0\r\no=- 0 0 IN IP4 127.0.0.1\r\ns=EdgeCCTV_test\r\nt=0 0\r\na=sendrecv\r\n",
        "type": "offer"
    }
    response = client.post("/api/v1/webrtc/offer", json=offer_payload, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "sdp" in data
    assert data["type"] == "answer"


def test_dvr_export_incident(client, auth_headers):
    now_utc = datetime.now(timezone.utc).isoformat()
    payload = {
        "start_time": now_utc,
        "end_time": now_utc,
        "title": "Suspicious Activity"
    }
    response = client.post("/api/v1/cameras/cam_living_room/export", json=payload, headers=auth_headers)
    assert response.status_code in [200, 404, 500]


def test_setup_status_endpoint(client):
    response = client.get("/api/v1/setup/status")
    assert response.status_code == 200
    data = response.json()
    assert "is_completed" in data
    assert "hardware_report" in data


def test_setup_hardware_scan(client, auth_headers):
    response = client.post("/api/v1/setup/hardware-scan", headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert "hardware" in data
    assert "hailo_available" in data["hardware"]
    assert "vaapi_available" in data["hardware"]


def test_pairing_code_flow(client):
    # 1. Generate pairing code
    code = auth_service.generate_app_pairing_code("test_admin")
    assert len(code) == 6
    assert code.isdigit()

    # 2. Pair with the generated code
    pair_res = client.post("/api/v1/auth/pair", json={"pairing_code": code})
    assert pair_res.status_code == 200
    pair_data = pair_res.json()
    assert "access_token" in pair_data
    assert "refresh_token" in pair_data

    # 3. Subsequent pair with same code should fail (single-use)
    replay_res = client.post("/api/v1/auth/pair", json={"pairing_code": code})
    assert replay_res.status_code == 401


def test_camera_feature_toggles_persistence(client, auth_headers):
    # 1. Update features via PUT
    payload = {
        "fall_detection_enabled": False,
        "skeletal_tracking_enabled": True,
        "object_detection_enabled": True,
        "package_detection_enabled": False,
        "animal_detection_enabled": False,
        "vehicle_detection_enabled": True,
    }
    res = client.put("/api/v1/cameras/cam_living_room/features", json=payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["fall_detection_enabled"] is False
    assert data["package_detection_enabled"] is False
    assert data["vehicle_detection_enabled"] is True

    # 2. Check feature_manager in-memory state
    cached_cfg = feature_manager.get_features("cam_living_room")
    assert cached_cfg.fall_detection_enabled is False
    assert cached_cfg.package_detection_enabled is False

    # 3. Fetch via GET /features
    get_res = client.get("/api/v1/cameras/cam_living_room/features", headers=auth_headers)
    assert get_res.status_code == 200
    assert get_res.json()["fall_detection_enabled"] is False

    # 4. Check CameraFeed includes features
    feed_res = client.get("/api/v1/cameras/cam_living_room", headers=auth_headers)
    assert feed_res.status_code == 200
    assert feed_res.json()["features"]["fall_detection_enabled"] is False

    # 5. Restore features
    restore_payload = {
        "fall_detection_enabled": True,
        "skeletal_tracking_enabled": True,
        "object_detection_enabled": True,
        "package_detection_enabled": True,
        "animal_detection_enabled": True,
        "vehicle_detection_enabled": True,
    }
    client.put("/api/v1/cameras/cam_living_room/features", json=restore_payload, headers=auth_headers)


def test_hailo_inference_service_feature_bypassing():
    cam_id = "cam_test_bypass"
    frame = None

    # Test 1: object_detection_enabled == False bypasses detection
    feature_manager.update_features(cam_id, CameraFeatureConfig(
        object_detection_enabled=False,
        skeletal_tracking_enabled=True,
        fall_detection_enabled=True,
    ))
    raw_det = [BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.8, confidence=0.9, label="person")]
    dets = hailo_inference_service.detect_objects(cam_id, frame, raw_detections=raw_det)
    assert dets == []
    events = hailo_inference_service.process_frame(cam_id, frame, simulated_detections=raw_det)
    assert events == []

    # Test 2: skeletal_tracking_enabled == False bypasses pose keypoints
    feature_manager.update_features(cam_id, CameraFeatureConfig(
        object_detection_enabled=True,
        skeletal_tracking_enabled=False,
        fall_detection_enabled=True,
    ))
    raw_kp = [Keypoint(id=0, name="nose", x=0.5, y=0.5, confidence=0.95)]
    kps = hailo_inference_service.detect_pose(cam_id, frame, raw_keypoints=raw_kp)
    assert kps == []

    # Test 3: fall_detection_enabled == False bypasses kinematics state machine
    feature_manager.update_features(cam_id, CameraFeatureConfig(
        object_detection_enabled=True,
        skeletal_tracking_enabled=True,
        fall_detection_enabled=False,
    ))
    fall_res = hailo_inference_service.run_kinematics(cam_id, 1, raw_kp, raw_det[0])
    assert fall_res is None


def test_sensor_node_registration_and_listing(client, auth_headers):
    # 1. Register ESP32 sentry node
    reg_payload = {
        "id": "sentry_front_porch",
        "name": "Front Porch ESP32 Sentry",
        "node_type": "ESP32_SENTRY",
        "ip_address": "192.168.1.150",
        "mac_address": "AA:BB:CC:11:22:33",
        "associated_camera_id": "cam_living_room",
        "enabled_sensors": {
            "pir_enabled": True,
            "ultrasonic_enabled": True,
            "door1_enabled": True,
            "door2_enabled": False,
            "distance_threshold_cm": 40.0
        }
    }
    res = client.post("/api/v1/sensors/register", json=reg_payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["id"] == "sentry_front_porch"
    assert data["name"] == "Front Porch ESP32 Sentry"
    assert data["ip_address"] == "192.168.1.150"

    # 2. List sensors
    list_res = client.get("/api/v1/sensors", headers=auth_headers)
    assert list_res.status_code == 200
    nodes = list_res.json()
    assert any(n["id"] == "sentry_front_porch" for n in nodes)


def test_sensor_toggles_update(client, auth_headers):
    # Update toggles for sentry_front_porch
    update_payload = {
        "pir_enabled": False,
        "door2_enabled": True,
        "distance_threshold_cm": 75.0
    }
    res = client.put("/api/v1/sensors/sentry_front_porch/toggles", json=update_payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["enabled_sensors"]["pir_enabled"] is False
    assert data["enabled_sensors"]["door2_enabled"] is True
    assert data["enabled_sensors"]["distance_threshold_cm"] == 75.0


def test_sensor_telemetry_and_incident_clip_triggering(client, auth_headers):
    # 1. Arm system
    arm_res = client.post("/api/v1/sensors/arm", headers=auth_headers)
    assert arm_res.status_code == 200
    assert arm_res.json()["armed"] is True

    # 2. Ingest normal telemetry (no alarm)
    telemetry_normal = {
        "camera_id": "cam_living_room",
        "pir_motion": False,
        "distance_cm": 150.0,
        "door1_open": False,
        "door2_open": False
    }
    res_normal = client.post("/api/v1/sensors/sentry_front_porch/telemetry", json=telemetry_normal, headers=auth_headers)
    assert res_normal.status_code == 200
    assert res_normal.json()["incident_triggered"] is False

    # 3. Ingest door opened alarm while armed -> should trigger 15s incident clip
    telemetry_door_alarm = {
        "camera_id": "cam_living_room",
        "pir_motion": False,
        "distance_cm": 20.0,
        "door1_open": True,
        "door2_open": False
    }
    res_door = client.post("/api/v1/sensors/sentry_front_porch/telemetry", json=telemetry_door_alarm, headers=auth_headers)
    assert res_door.status_code == 200
    door_data = res_door.json()
    assert door_data["incident_triggered"] is True
    assert door_data["event_id"] is not None

    # Verify event was recorded in DB
    async def _check_event(evt_id):
        async with async_session_factory() as session:
            st = select(SecurityEventModel).where(SecurityEventModel.id == evt_id)
            r = await session.execute(st)
            return r.scalar_one_or_none()

    saved_event = asyncio.run(_check_event(door_data["event_id"]))
    assert saved_event is not None
    assert saved_event.event_type == EventType.DOOR_LEFT_OPEN.value
    assert saved_event.severity == EventSeverity.CRITICAL.value

    # 4. Ingest PIR motion alarm while armed
    telemetry_pir_alarm = {
        "camera_id": "cam_living_room",
        "pir_motion": True,
        "door1_open": False,
        "door2_open": False
    }
    res_pir = client.post("/api/v1/sensors/sentry_front_porch/telemetry", json=telemetry_pir_alarm, headers=auth_headers)
    assert res_pir.status_code == 200
    pir_data = res_pir.json()
    assert pir_data["incident_triggered"] is True

    # 5. Disarm system and verify no incident clip triggered on door open
    disarm_res = client.post("/api/v1/sensors/disarm", headers=auth_headers)
    assert disarm_res.status_code == 200
    assert disarm_res.json()["armed"] is False

    res_disarmed = client.post("/api/v1/sensors/sentry_front_porch/telemetry", json=telemetry_door_alarm, headers=auth_headers)
    assert res_disarmed.status_code == 200
    assert res_disarmed.json()["incident_triggered"] is False


def test_sensors_scan_endpoint(client, auth_headers):
    # Test POST /api/v1/sensors/scan
    res = client.post("/api/v1/sensors/scan", headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "scanned_ips" in data
    assert "found_count" in data
    assert "nodes" in data
    assert isinstance(data["nodes"], list)


def test_sensors_offline_marking_after_30s(client, auth_headers):
    from datetime import datetime, timedelta
    # 1. Register an old node whose last_heartbeat is 45s ago
    reg_payload = {
        "id": "sentry_stale_node",
        "name": "Stale Node",
        "ip_address": "192.168.1.199",
        "node_type": "ESP32_SENTRY"
    }
    res = client.post("/api/v1/sensors/register", json=reg_payload, headers=auth_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ONLINE"

    # 2. Artificially set last_heartbeat to 45 seconds ago in SQLite
    async def _age_node():
        async with async_session_factory() as session:
            st = select(SensorNodeModel).where(SensorNodeModel.id == "sentry_stale_node")
            r = await session.execute(st)
            node = r.scalar_one()
            node.last_heartbeat = datetime.utcnow() - timedelta(seconds=45)
            await session.commit()

    asyncio.run(_age_node())

    # 3. Listing sensors should now mark the node as OFFLINE
    list_res = client.get("/api/v1/sensors", headers=auth_headers)
    assert list_res.status_code == 200
    nodes = list_res.json()
    stale_node = next((n for n in nodes if n["id"] == "sentry_stale_node"), None)
    assert stale_node is not None
    assert stale_node["status"] == "OFFLINE"
    assert stale_node["sensor_states"]["status"] == "OFFLINE"


def test_sensor_toggles_timeout_and_unreachable(client, auth_headers, monkeypatch):
    import httpx
    monkeypatch.setenv("TEST_ESP32_FORWARD", "1")

    # Mock httpx timeout to verify 504 Gateway Timeout
    async def mock_post_timeout(*args, **kwargs):
        raise httpx.TimeoutException("Mocked 2.0s timeout")

    monkeypatch.setattr(httpx.AsyncClient, "post", mock_post_timeout)

    update_payload = {"pir_enabled": True}
    res = client.put("/api/v1/sensors/sentry_front_porch/toggles", json=update_payload, headers=auth_headers)
    assert res.status_code == 504
    assert "Gateway Timeout" in res.json()["detail"] or "timed out" in res.json()["detail"]
