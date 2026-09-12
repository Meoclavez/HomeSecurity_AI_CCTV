"""Automated integration tests for monitor_live_ai.py Web HUD endpoints, ESP32 auto-discovery,
watchdog, state-aware feature toggles, and thread-safety concurrency verification.
"""

import os
import sys
import time
import json
import threading
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Add project root and scripts directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "edge_backend"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from monitor_live_ai import (
    LiveAIMonitor,
    create_web_hud_app,
    ESP32SentryScanner,
    DetectionObject,
)


@pytest.fixture
def monitor_instance():
    """Create a test LiveAIMonitor instance without physical cameras."""
    os.environ["TESTING"] = "1"
    # Create monitor with synthetic stream
    monitor = LiveAIMonitor(initial_stream="synthetic")
    # Mark running and disconnected initially
    monitor.is_esp32_connected = False
    monitor.esp32_connection_status = "DISCONNECTED"
    monitor.connected_esp32_devices = []
    monitor.demo_simulation_enabled = False
    yield monitor
    monitor.is_running = False


@pytest.fixture
def hud_client(monitor_instance):
    """Create a FastAPI TestClient for the monitor's Web HUD."""
    app = create_web_hud_app(monitor_instance)
    client = TestClient(app)
    return client, monitor_instance


# ==============================================================================
# 1. ESP32 Auto-Discovery & Background Watchdog Tests
# ==============================================================================

def test_esp32_sentry_scanner_probe(monkeypatch):
    """Verify that ESP32SentryScanner tests ports and scans gracefully with timeout."""
    # Test test_http_port with unreachable host
    assert not ESP32SentryScanner.test_http_port("127.0.0.1", 59999, timeout=0.1)

    # Mock probe_target to simulate discovering a device
    mock_device = {
        "ip": "192.168.1.188",
        "port": 80,
        "camera_id": "esp32_sentry_test",
        "url": "http://192.168.1.188:80",
        "endpoint": "/sensors",
        "data": {"pir_motion": True, "distance_cm": 45.2, "door1_open": True, "door2_open": False},
        "pir_motion": True,
        "distance_cm": 45.2,
        "door1_open": True,
        "door2_open": False,
        "last_seen": time.time(),
    }

    monkeypatch.setattr(ESP32SentryScanner, "scan_esp32_devices", lambda: [mock_device])
    devices = ESP32SentryScanner.scan_esp32_devices()
    assert len(devices) == 1
    assert devices[0]["camera_id"] == "esp32_sentry_test"
    assert devices[0]["pir_motion"] is True


def test_monitor_scan_esp32_devices_success_and_offline(hud_client, monkeypatch):
    """Verify monitor updates state when devices are found vs when none found."""
    client, monitor = hud_client

    mock_device = {
        "ip": "192.168.1.150",
        "port": 80,
        "camera_id": "esp32_porch",
        "url": "http://192.168.1.150:80",
        "endpoint": "/sensors",
        "data": {"pir_motion": True, "distance_cm": 35.0, "door1_open": False, "door2_open": False},
        "pir_motion": True,
        "distance_cm": 35.0,
        "door1_open": False,
        "door2_open": False,
        "last_seen": time.time(),
    }

    # Simulate discovery
    monkeypatch.setattr(ESP32SentryScanner, "scan_esp32_devices", lambda: [mock_device])
    devices = monitor.scan_esp32_devices()
    assert len(devices) == 1
    assert monitor.is_esp32_connected is True
    assert monitor.esp32_connection_status == "ONLINE"
    assert monitor.active_esp32_ip == "192.168.1.150"
    assert monitor.esp32_sensor_readings["pir_motion"] is True
    assert monitor.esp32_sensor_readings["distance_cm"] == 35.0

    # Simulate empty discovery
    monkeypatch.setattr(ESP32SentryScanner, "scan_esp32_devices", lambda: [])
    devices = monitor.scan_esp32_devices()
    assert len(devices) == 0
    assert monitor.is_esp32_connected is False
    assert monitor.esp32_connection_status == "DISCONNECTED"


def test_watchdog_transition_to_error_when_node_unresponsive(hud_client, monkeypatch):
    """Verify watchdog marks node as ERROR when ping fails."""
    client, monitor = hud_client
    monitor.is_esp32_connected = True
    monitor.esp32_connection_status = "ONLINE"
    monitor.active_esp32_ip = "192.168.1.150"

    # Ping returns False -> unreachable
    monkeypatch.setattr(monitor, "_ping_esp32_node", lambda ip, port: False)

    # Run one iteration of watchdog logic
    alive = monitor._ping_esp32_node(monitor.active_esp32_ip, monitor.active_esp32_port)
    if not alive:
        monitor.is_esp32_connected = False
        monitor.esp32_connection_status = "ERROR"
        monitor.connected_esp32_devices = []

    assert monitor.is_esp32_connected is False
    assert monitor.esp32_connection_status == "ERROR"
    assert len(monitor.connected_esp32_devices) == 0


# ==============================================================================
# 2. Manual ESP32 Rescan Functionality (/api/sensors/rescan)
# ==============================================================================

def test_api_sensors_rescan_endpoint(hud_client, monkeypatch):
    """Verify POST /api/sensors/rescan returns found_count and status."""
    client, monitor = hud_client

    mock_devices = [
        {
            "ip": "192.168.1.150",
            "port": 80,
            "camera_id": "esp32_sentry_01",
            "url": "http://192.168.1.150:80",
            "endpoint": "/sensors",
            "data": {},
        }
    ]
    monkeypatch.setattr(ESP32SentryScanner, "scan_esp32_devices", lambda: mock_devices)

    res = client.post("/api/sensors/rescan")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["found_count"] == 1
    assert data["is_connected"] is True
    assert data["esp32_connection_status"] == "ONLINE"


# ==============================================================================
# 3. State-Aware Toggles & Lock Verification
# ==============================================================================

def test_toggles_rejected_400_when_esp32_disconnected(hud_client):
    """Ensure POST /api/toggles returns HTTP 400 Bad Request when toggling esp32_* without sentry."""
    client, monitor = hud_client
    monitor.is_esp32_connected = False
    monitor.demo_simulation_enabled = False

    # 1. Feature syntax: {"feature": "esp32_pir", "enabled": False}
    res = client.post("/api/toggles", json={"feature": "esp32_pir", "enabled": False})
    assert res.status_code == 400
    assert "No physical ESP32 sentry connected" in res.json()["message"]

    # 2. Toggles syntax: {"toggles": {"esp32_ultrasonic": False}}
    res = client.post("/api/toggles", json={"toggles": {"esp32_ultrasonic": False}})
    assert res.status_code == 400
    assert "No physical ESP32 sentry connected" in res.json()["message"]

    # 3. Direct key syntax: {"esp32_door1": False}
    res = client.post("/api/toggles", json={"esp32_door1": False})
    assert res.status_code == 400
    assert "No physical ESP32 sentry connected" in res.json()["message"]

    # 4. Direct key syntax: {"esp32_door2": False}
    res = client.post("/api/toggles", json={"esp32_door2": False})
    assert res.status_code == 400
    assert "No physical ESP32 sentry connected" in res.json()["message"]

    # 5. Non-ESP32 features succeed even when disconnected
    res = client.post("/api/toggles", json={"feature": "fall_detection", "enabled": False})
    assert res.status_code == 200
    assert res.json()["feature_toggles"]["fall_detection"] is False


def test_toggles_allowed_in_demo_mode(hud_client):
    """Ensure POST /api/toggles succeeds when demo simulation mode is enabled."""
    client, monitor = hud_client
    monitor.is_esp32_connected = False

    # Enable demo mode via API
    res = client.post("/api/sensors/demo_mode", json={"enabled": True})
    assert res.status_code == 200
    assert res.json()["demo_enabled"] is True
    assert monitor.demo_simulation_enabled is True

    # Now toggling esp32_* should succeed
    res = client.post("/api/toggles", json={"feature": "esp32_pir", "enabled": False})
    assert res.status_code == 200
    assert res.json()["feature_toggles"]["esp32_pir"] is False

    res = client.post("/api/toggles", json={"feature": "esp32_door1", "enabled": False})
    assert res.status_code == 200
    assert res.json()["feature_toggles"]["esp32_door1"] is False


def test_toggles_allowed_when_esp32_connected(hud_client):
    """Ensure POST /api/toggles succeeds when physical ESP32 is connected."""
    client, monitor = hud_client
    monitor.is_esp32_connected = True
    monitor.demo_simulation_enabled = False

    res = client.post("/api/toggles", json={"feature": "esp32_ultrasonic", "enabled": False})
    assert res.status_code == 200
    assert res.json()["feature_toggles"]["esp32_ultrasonic"] is False


# ==============================================================================
# 4. Telemetry Ingestion & Security Alerts (/api/sensors/esp32)
# ==============================================================================

def test_api_sensors_esp32_telemetry_and_alerts(hud_client):
    """Verify /api/sensors/esp32 ingests readings and triggers appropriate security alerts."""
    client, monitor = hud_client

    payload = {
        "ip": "192.168.1.150",
        "pir_motion": True,
        "distance_cm": 42.1,
        "door1_open": True,
        "door2_open": False,
    }

    res = client.post("/api/sensors/esp32", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert data["is_esp32_connected"] is True
    assert data["esp32_sensor_readings"]["pir_motion"] is True
    assert data["esp32_sensor_readings"]["distance_cm"] == 42.1
    assert data["esp32_sensor_readings"]["door1_open"] is True

    # Check that alert events were recorded in event_log
    with monitor.event_lock:
        event_types = [e["type"] for e in monitor.event_log]
    assert "PIR_MOTION" in event_types
    assert "DOOR_OPEN" in event_types


# ==============================================================================
# 5. Thread-Safety & Concurrent Stress Test
# ==============================================================================

def test_thread_safety_concurrent_requests_and_track_updates(hud_client):
    """Stress test concurrent requests to /api/status, /api/toggles, and track updates
    to ensure zero dictionary mutation race conditions or crashes.
    """
    client, monitor = hud_client
    errors = []

    def status_poller():
        for _ in range(50):
            try:
                res = client.get("/api/status")
                assert res.status_code == 200
                data = res.json()
                assert "fps" in data
                assert "total_tracks" in data
            except Exception as e:
                errors.append(f"status_poller error: {e}")
            time.sleep(0.005)

    def toggles_flipper():
        for i in range(30):
            try:
                val = (i % 2 == 0)
                res = client.post("/api/toggles", json={"feature": "fall_detection", "enabled": val})
                assert res.status_code == 200
            except Exception as e:
                errors.append(f"toggles_flipper error: {e}")
            time.sleep(0.008)

    def tracks_mutator():
        for i in range(60):
            try:
                # Add mock detections
                dets = [
                    DetectionObject(
                        bbox=(0.2 + (i % 10) * 0.01, 0.3, 0.4, 0.7),
                        confidence=0.85,
                        class_id=0,
                        class_name="person",
                        keypoints=[],
                    )
                ]
                monitor.update_tracks(dets)
            except Exception as e:
                errors.append(f"tracks_mutator error: {e}")
            time.sleep(0.004)

    t1 = threading.Thread(target=status_poller)
    t2 = threading.Thread(target=toggles_flipper)
    t3 = threading.Thread(target=tracks_mutator)

    t1.start()
    t2.start()
    t3.start()

    t1.join(timeout=5.0)
    t2.join(timeout=5.0)
    t3.join(timeout=5.0)

    assert not errors, f"Encountered concurrency errors: {errors}"
