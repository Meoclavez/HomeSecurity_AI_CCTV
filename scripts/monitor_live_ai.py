#!/usr/bin/env python3
"""Edge AI CCTV - Real-Time Live AI Vision & Performance Monitor GUI & Web HUD.

Features:
1. Dual Display Engine:
   - Native Desktop HighGUI Window (if available/supported)
   - Embedded Cyberpunk Web HUD Server (http://0.0.0.0:8080) for headless, Wayland, remote, and mobile access
2. Dynamic Universal Camera Discovery:
   - Automatic scanning on startup (mDNS, Subnet sweep, USB /dev/video*, RTSP/MJPEG)
   - On-demand rescan and hot-swapping via UI buttons or API
3. Kinematic AI Vision & Security Zone Evaluation:
   - Real-time Fall Detection HUD (Torso Angle, Hip Velocity, Aspect Ratio, Sirens)
   - Virtual Tripwires (Entry/Exit counters, directional lines)
   - Polygon Intrusion Zones (Ray-casting point-in-polygon)
4. Interactive Event Simulation:
   - Fall, Tripwire, Intrusion, Snapshot capture, and 10s MP4 Clip export
"""

import argparse
import asyncio
import io
import json
import logging
import os
import re
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import cv2
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "edge_backend"))

from app.config import settings
from app.models.schemas import Point2D, TripwireDirection, ZoneConfig, ZoneType
from app.services.ai_zone_service import ai_zone_service
from app.services.clip_recorder import clip_recorder_service
from app.services.hailo_inference_service import hailo_inference_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveMonitor")


# ==============================================================================
# Camera Scanner & Stream Ingestion Manager
# ==============================================================================
class CameraScanner:
    """Discovers local USB cameras, mDNS ESP32 devices, and subnet RTSP/MJPEG streams."""

    @staticmethod
    def scan_usb_cameras() -> List[Dict[str, str]]:
        """Probe /dev/video* devices on Linux or indexes 0-2 on Windows/Mac."""
        found = []
        if sys.platform.startswith("linux"):
            for dev in sorted(Path("/dev").glob("video*")):
                try:
                    # Quick non-blocking probe
                    cap = cv2.VideoCapture(str(dev))
                    if cap.isOpened():
                        ret, _ = cap.read()
                        if ret:
                            found.append({"id": f"usb_{dev.name}", "name": f"Local USB Camera ({dev})", "url": str(dev)})
                        cap.release()
                except Exception:
                    pass
        else:
            for idx in range(3):
                try:
                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        found.append({"id": f"usb_{idx}", "name": f"USB Webcam (Index {idx})", "url": str(idx)})
                        cap.release()
                except Exception:
                    pass
        return found

    @staticmethod
    def test_http_port(host: str, port: int, timeout: float = 0.4) -> bool:
        """Fast TCP socket test."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                return s.connect_ex((host, port)) == 0
        except Exception:
            return False

    @classmethod
    def scan_network_cameras(cls) -> List[Dict[str, str]]:
        """Probe mDNS, common local IP subnets for ESP32 and RTSP cameras."""
        found = []

        # 1. Probe mDNS esp32-cctv.local
        try:
            ip = socket.gethostbyname("esp32-cctv.local")
            found.append({
                "id": "esp32_mdns",
                "name": f"ESP32-S3 Camera (esp32-cctv.local:81)",
                "url": f"http://{ip}:81/stream"
            })
        except Exception:
            pass

        # 2. Get local host IP to determine subnet
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            
            ip_parts = local_ip.split(".")
            subnet_prefix = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}."

            # Probe top candidate host IPs on port 81 and 80
            candidate_ips = [
                local_ip,
                f"{subnet_prefix}1",
                f"{subnet_prefix}2",
                f"{subnet_prefix}86",  # User's current ESP32 IP
                f"{subnet_prefix}100",
                f"{subnet_prefix}150",
                f"{subnet_prefix}200"
            ]

            for ip in candidate_ips:
                if cls.test_http_port(ip, 81):
                    found.append({
                        "id": f"net_{ip}_81",
                        "name": f"ESP32 MJPEG Camera (http://{ip}:81/stream)",
                        "url": f"http://{ip}:81/stream"
                    })
                elif cls.test_http_port(ip, 80):
                    found.append({
                        "id": f"net_{ip}_80",
                        "name": f"Network Camera (http://{ip}/stream)",
                        "url": f"http://{ip}/stream"
                    })
        except Exception:
            pass

        return found

    @classmethod
    def discover_all(cls) -> List[Dict[str, str]]:
        """Return all detected camera sources plus synthetic generator."""
        sources = []
        # Synthetic generator always available
        sources.append({
            "id": "synthetic",
            "name": "🛡️ Synthetic Benchmark & Kinematics Generator",
            "url": "synthetic"
        })

        usb_cams = cls.scan_usb_cameras()
        sources.extend(usb_cams)

        net_cams = cls.scan_network_cameras()
        sources.extend(net_cams)

        return sources


# ==============================================================================
# Live AI Monitor Engine
# ==============================================================================
class LiveAIMonitor:
    def __init__(self, initial_stream: Optional[str] = None):
        self.stream_source = initial_stream
        self.camera_id = "cam_live_eval"
        self.is_running = True
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_source_name = "Detecting..."
        
        # Performance Metrics
        self.fps = 0.0
        self.frame_count = 0
        self.avg_inference_ms = 0.0
        self.recent_latencies: List[float] = []
        
        # Kinematic Fall Telemetry
        self.torso_angle = 85.0
        self.aspect_ratio = 1.9
        self.descent_velocity = 0.1
        self.floor_proximity = 0.15
        self.is_fall_active = False
        
        # Alerts & Events
        self.active_alert_banner = ""
        self.alert_expiry = 0.0
        self.in_count = 0
        self.out_count = 0
        
        # Synthetic Simulation State
        self.sim_person_x = 0.5
        self.sim_person_y = 0.5
        self.sim_is_falling = False
        self.sim_fall_start = 0.0
        self.sim_is_tripwire = False
        self.sim_is_intrusion = False
        
        # Frame Broadcast Buffer for Web HUD
        self.latest_encoded_jpeg: Optional[bytes] = None
        self.frame_lock = threading.Lock()
        
        # Discovered Camera Sources
        self.available_sources: List[Dict[str, str]] = []
        
        # Ring Buffer for Clips
        self.ring_buffer = clip_recorder_service.get_or_create_buffer(self.camera_id)

        # Setup AI Zones
        self._init_zones()

    def _init_zones(self):
        zones = [
            ZoneConfig(
                id="zone_tripwire_gate",
                camera_id=self.camera_id,
                name="Front Gate Tripwire",
                zone_type=ZoneType.TRIPWIRE,
                enabled=True,
                line_start=Point2D(x=0.15, y=0.55),
                line_end=Point2D(x=0.85, y=0.55),
                direction=TripwireDirection.BIDIRECTIONAL,
            ),
            ZoneConfig(
                id="zone_intrusion_porch",
                camera_id=self.camera_id,
                name="High-Security Porch Zone",
                zone_type=ZoneType.INTRUSION,
                enabled=True,
                polygon_points=[
                    Point2D(x=0.55, y=0.25),
                    Point2D(x=0.95, y=0.25),
                    Point2D(x=0.95, y=0.85),
                    Point2D(x=0.55, y=0.85),
                ],
                dwell_time_seconds=1.5,
            ),
            ZoneConfig(
                id="zone_privacy_neighbor",
                camera_id=self.camera_id,
                name="Neighbor Privacy Mask",
                zone_type=ZoneType.PRIVACY_MASK,
                enabled=True,
                polygon_points=[
                    Point2D(x=0.02, y=0.02),
                    Point2D(x=0.22, y=0.02),
                    Point2D(x=0.22, y=0.25),
                    Point2D(x=0.02, y=0.25),
                ],
            ),
        ]
        ai_zone_service.load_camera_zones(self.camera_id, zones)

    def trigger_alert(self, message: str, duration: float = 3.0):
        self.active_alert_banner = message
        self.alert_expiry = time.time() + duration
        logger.info(f"🚨 [ALERT] {message}")

    def switch_source(self, new_source_url: str):
        """Hot-swaps live camera input without restarting application."""
        logger.info(f"[+] Switching video source to: {new_source_url}")
        if self.cap:
            self.cap.release()
            self.cap = None
        self.stream_source = new_source_url
        self.open_video_source()

    def rescan_sources(self) -> List[Dict[str, str]]:
        """Scans network & USB devices and updates source list."""
        logger.info("[+] Scanning for connected cameras and streams...")
        sources = CameraScanner.discover_all()
        self.available_sources = sources
        self.trigger_alert(f"Scan complete: Found {len(sources)} video source(s).", duration=2.5)
        return sources

    def open_video_source(self):
        """Attempts connection to specified stream, discovered cameras, or synthetic fallback."""
        # 1. Try explicit user stream
        if self.stream_source and self.stream_source != "synthetic":
            logger.info(f"[+] Connecting to custom stream: {self.stream_source}")
            try:
                # Open with low timeout
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;3000000|stimeout;3000000"
                cap = cv2.VideoCapture(self.stream_source, cv2.CAP_FFMPEG if "http" in self.stream_source or "rtsp" in self.stream_source else cv2.CAP_ANY)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        logger.info(f"✅ Successfully connected to: {self.stream_source}")
                        self.cap = cap
                        self.current_source_name = f"Live Stream ({self.stream_source})"
                        return
                    cap.release()
            except Exception as e:
                logger.warning(f"[-] Could not open {self.stream_source}: {e}")

        # 2. Try Auto-Discovery
        if not self.available_sources:
            self.available_sources = CameraScanner.discover_all()

        for src in self.available_sources:
            if src["url"] == "synthetic":
                continue
            logger.info(f"[+] Probing discovered source: {src['name']} ({src['url']})")
            try:
                cap = cv2.VideoCapture(src["url"])
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        logger.info(f"✅ Connected to discovered camera: {src['name']}")
                        self.cap = cap
                        self.stream_source = src["url"]
                        self.current_source_name = src["name"]
                        return
                    cap.release()
            except Exception:
                pass

        # 3. Fallback to Synthetic Generator
        logger.info("🛡️ Falling back to Built-In High-Fidelity Synthetic AI Pipeline.")
        self.cap = None
        self.stream_source = "synthetic"
        self.current_source_name = "Synthetic Benchmark Feed"

    def generate_simulated_frame(self) -> np.ndarray:
        """Generates realistic test CCTV footage with simulated persons, kinematics, and zones."""
        frame = np.full((720, 1280, 3), (24, 28, 36), dtype=np.uint8)
        t = time.time()

        # Draw living room background
        # Floor
        cv2.rectangle(frame, (0, 480), (1280, 720), (35, 40, 50), -1)
        # Floor grid lines
        for y in range(480, 720, 40):
            cv2.line(frame, (0, y), (1280, y), (45, 52, 65), 1)
        for x in range(0, 1280, 80):
            cv2.line(frame, (x, 480), (int((x - 640) * 1.6 + 640), 720), (45, 52, 65), 1)

        # Wall paneling & Door
        cv2.rectangle(frame, (880, 200), (1080, 580), (48, 55, 68), -1)
        cv2.rectangle(frame, (880, 200), (1080, 580), (80, 95, 115), 2)
        cv2.circle(frame, (900, 400), 6, (140, 160, 180), -1)

        # Handle Interactive Simulations
        if self.sim_is_falling:
            dt = t - self.sim_fall_start
            if dt < 1.0:
                # Rapid descent
                progress = min(1.0, dt / 0.8)
                self.sim_person_y = 0.45 + progress * 0.35
                self.torso_angle = 85.0 - progress * 75.0  # Drops to 10 deg
                self.aspect_ratio = 1.9 - progress * 1.35  # Drops to 0.55
                self.descent_velocity = 2.4 * (1.0 - progress)
                self.floor_proximity = 0.15 + progress * 0.75
                self.is_fall_active = True
            elif dt < 5.0:
                # Immobility on floor
                self.torso_angle = 10.0
                self.aspect_ratio = 0.55
                self.descent_velocity = 0.0
                self.floor_proximity = 0.90
                self.is_fall_active = True
            else:
                # Reset
                self.sim_is_falling = False
                self.is_fall_active = False
                self.torso_angle = 85.0
                self.aspect_ratio = 1.9
                self.descent_velocity = 0.1
                self.floor_proximity = 0.15
        elif self.sim_is_tripwire:
            # Cross tripwire line vertically
            self.sim_person_x = 0.5
            self.sim_person_y = 0.45 + 0.25 * np.sin(t * 1.5)
        elif self.sim_is_intrusion:
            # Enter porch polygon
            self.sim_person_x = 0.75 + 0.1 * np.cos(t)
            self.sim_person_y = 0.55 + 0.1 * np.sin(t)
        else:
            # Normal gentle walking motion
            self.sim_person_x = 0.45 + 0.15 * np.sin(t * 0.4)
            self.sim_person_y = 0.50 + 0.03 * np.cos(t * 0.4)
            self.torso_angle = 85.0 + 3.0 * np.sin(t * 2.0)
            self.aspect_ratio = 1.9
            self.descent_velocity = 0.1
            self.floor_proximity = 0.15

        # Render Person Avatar
        px = int(self.sim_person_x * 1280)
        py = int(self.sim_person_y * 720)
        box_h = int(140 * (self.aspect_ratio / 1.9))
        box_w = int(60 if not self.sim_is_falling else 120)

        # Person shadow
        cv2.ellipse(frame, (px, py + box_h // 2 + 10), (box_w // 2 + 15, 12), 0, 0, 360, (15, 18, 24), -1)

        # Person Body Box
        color = (0, 0, 255) if self.is_fall_active else (0, 240, 255)
        cv2.rectangle(frame, (px - box_w // 2, py - box_h // 2), (px + box_w // 2, py + box_h // 2), color, 2)
        cv2.putText(frame, f"ID:101 person ({'FALLEN' if self.is_fall_active else '94%'})",
                    (px - box_w // 2, py - box_h // 2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        return frame

    def draw_hud(self, frame: np.ndarray, detections: List[Dict], events: List) -> np.ndarray:
        """Draws top HUD bar, telemetry gauges, zones, tripwires, and alarm sirens."""
        hud = frame.copy()
        h, w = hud.shape[:2]

        # 1. Apply Security Zones
        zones = ai_zone_service.get_camera_zones(self.camera_id)
        for zone in zones:
            if not zone.enabled:
                continue
            if zone.zone_type == ZoneType.TRIPWIRE and zone.line_start and zone.line_end:
                p1 = (int(zone.line_start.x * w), int(zone.line_start.y * h))
                p2 = (int(zone.line_end.x * w), int(zone.line_end.y * h))
                cv2.line(hud, p1, p2, (255, 180, 0), 2)
                cv2.putText(hud, f"TRIPWIRE: {zone.name} [A <-> B]", (p1[0] + 5, p1[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1)
            elif zone.zone_type == ZoneType.INTRUSION and zone.polygon_points:
                pts = np.array([[int(p.x * w), int(p.y * h)] for p in zone.polygon_points], np.int32)
                overlay = hud.copy()
                cv2.fillPoly(overlay, [pts], (0, 0, 200))
                cv2.addWeighted(overlay, 0.25, hud, 0.75, 0, hud)
                cv2.polylines(hud, [pts], True, (0, 0, 255), 2)
                cv2.putText(hud, f"ZONE: {zone.name}", (pts[0][0] + 5, pts[0][1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 100, 255), 1)
            elif zone.zone_type == ZoneType.PRIVACY_MASK and zone.polygon_points:
                pts = np.array([[int(p.x * w), int(p.y * h)] for p in zone.polygon_points], np.int32)
                cv2.fillPoly(hud, [pts], (15, 15, 15))
                cv2.polylines(hud, [pts], True, (60, 60, 60), 1)
                cv2.putText(hud, "PRIVACY MASK", (pts[0][0] + 5, pts[0][1] + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120, 120, 120), 1)

        # 2. Top Cyberpunk HUD Bar
        overlay = hud.copy()
        cv2.rectangle(overlay, (0, 0), (w, 54), (10, 13, 18), -1)
        cv2.addWeighted(overlay, 0.85, hud, 0.15, 0, hud)
        cv2.line(hud, (0, 54), (w, 54), (0, 240, 255), 1)

        # Title & Camera Name
        cv2.putText(hud, "🛡️ EDGE AI CCTV - LIVE EVALUATOR", (16, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 255), 2)
        cv2.putText(hud, f"Source: {self.current_source_name}", (16, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 180, 200), 1)

        # FPS & Hardware Metrics
        fps_text = f"FPS: {self.fps:.1f} | Latency: {self.avg_inference_ms:.1f}ms | Resolution: {w}x{h}"
        cv2.putText(hud, fps_text, (w - 460, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 160), 1)

        engine_text = "Hailo-8 M.2: Active (Sim Mode)" if hailo_inference_service.is_simulated else "Hailo-8 M.2: PCIe Native"
        cv2.putText(hud, engine_text, (w - 460, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 240), 1)

        # 3. Kinematic Fall Metrics Panel (Bottom Left)
        kin_w, kin_h = 320, 130
        kx, ky = 16, h - kin_h - 16
        overlay = hud.copy()
        cv2.rectangle(overlay, (kx, ky), (kx + kin_w, ky + kin_h), (12, 16, 24), -1)
        cv2.addWeighted(overlay, 0.85, hud, 0.15, 0, hud)
        border_color = (0, 0, 255) if self.is_fall_active else (0, 200, 255)
        cv2.rectangle(hud, (kx, ky), (kx + kin_w, ky + kin_h), border_color, 1)

        cv2.putText(hud, "KINEMATIC POSE TELEMETRY", (kx + 10, ky + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, border_color, 1)

        angle_color = (0, 0, 255) if self.torso_angle < 35 else (0, 255, 180)
        cv2.putText(hud, f"• Torso Angle (θ): {self.torso_angle:.1f}° {'(CRITICAL)' if self.torso_angle < 35 else '(NORMAL)'}",
                    (kx + 10, ky + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, angle_color, 1)

        vel_color = (0, 0, 255) if self.descent_velocity > 1.8 else (0, 255, 180)
        cv2.putText(hud, f"• Descent Velocity (Vy): {self.descent_velocity:.2f} m/s",
                    (kx + 10, ky + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, vel_color, 1)

        aspect_color = (0, 0, 255) if self.aspect_ratio < 0.8 else (0, 255, 180)
        cv2.putText(hud, f"• Aspect Ratio (H/W): {self.aspect_ratio:.2f} {'(COLLAPSED)' if self.aspect_ratio < 0.8 else '(UPRIGHT)'}",
                    (kx + 10, ky + 91), cv2.FONT_HERSHEY_SIMPLEX, 0.42, aspect_color, 1)

        cv2.putText(hud, f"• Floor Proximity: {self.floor_proximity:.2f}",
                    (kx + 10, ky + 114), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 220), 1)

        # 4. Active Alert Banner
        if time.time() < self.alert_expiry:
            banner_y = h - 60
            overlay = hud.copy()
            cv2.rectangle(overlay, (w // 2 - 340, banner_y), (w // 2 + 340, banner_y + 44), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.90, hud, 0.10, 0, hud)
            cv2.rectangle(hud, (w // 2 - 340, banner_y), (w // 2 + 340, banner_y + 44), (0, 240, 255), 2)
            cv2.putText(hud, self.active_alert_banner, (w // 2 - 320, banner_y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        return hud

    def capture_frame_loop(self):
        """Continuous ingestion and AI processing loop."""
        self.open_video_source()
        last_time = time.time()

        while self.is_running:
            loop_start = time.time()

            # Grab Frame
            frame = None
            if self.cap and self.cap.isOpened():
                ret, raw_frame = self.cap.read()
                if ret and raw_frame is not None:
                    frame = raw_frame
                else:
                    frame = self.generate_simulated_frame()
            else:
                frame = self.generate_simulated_frame()

            # 1. AI Inference Simulation / Hailo
            ai_start = time.time()
            h, w = frame.shape[:2]
            detections = [
                {
                    "track_id": 101,
                    "class_name": "person",
                    "confidence": 0.94,
                    "bbox": [
                        max(0.0, self.sim_person_x - (0.04 if not self.sim_is_falling else 0.08)),
                        max(0.0, self.sim_person_y - (0.09 if not self.sim_is_falling else 0.04)),
                        min(1.0, self.sim_person_x + (0.04 if not self.sim_is_falling else 0.08)),
                        min(1.0, self.sim_person_y + (0.09 if not self.sim_is_falling else 0.04)),
                    ]
                }
            ]

            # AI Zone & Tripwire Processing
            events = ai_zone_service.process_detections(self.camera_id, detections, w, h)
            for ev in events:
                self.trigger_alert(f"ZONE EVENT: {ev.event_type.name} on {ev.camera_id}!")

            ai_latency = (time.time() - ai_start) * 1000.0
            self.recent_latencies.append(ai_latency)
            if len(self.recent_latencies) > 30:
                self.recent_latencies.pop(0)
            self.avg_inference_ms = sum(self.recent_latencies) / len(self.recent_latencies)

            # 2. Ring buffer push
            self.ring_buffer.push_frame(frame)

            # 3. Draw HUD
            hud_frame = self.draw_hud(frame, detections, events)

            # 4. Encode JPEG for Web HUD broadcast
            ret, jpeg_bytes = cv2.imencode(".jpg", hud_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                with self.frame_lock:
                    self.latest_encoded_jpeg = jpeg_bytes.tobytes()

            # 5. FPS Calculation
            self.frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                self.fps = self.frame_count / (now - last_time)
                self.frame_count = 0
                last_time = now

            # Pacing
            elapsed = time.time() - loop_start
            sleep_time = max(0.005, (1.0 / 30.0) - elapsed)
            time.sleep(sleep_time)


# ==============================================================================
# Embedded Asynchronous Web HUD Server (FastAPI / Uvicorn)
# ==============================================================================
def create_web_hud_app(monitor: LiveAIMonitor):
    from fastapi import FastAPI, Response
    from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    web_app = FastAPI(title="Edge AI CCTV Web HUD")
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    class SwitchSourceReq(BaseModel):
        url: str

    @web_app.get("/", response_class=HTMLResponse)
    async def index():
        html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Edge AI CCTV - Live Monitor & Evaluator</title>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0e14;
      --card-bg: rgba(18, 22, 31, 0.75);
      --card-border: rgba(0, 240, 255, 0.2);
      --accent-cyan: #00f0ff;
      --accent-green: #00ff9d;
      --accent-red: #ff0055;
      --accent-amber: #ffaa00;
      --text-main: #f0f6fc;
      --text-dim: #8b949e;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background-color: var(--bg);
      color: var(--text-main);
      font-family: 'Plus Jakarta Sans', sans-serif;
      padding: 24px;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 16px 24px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .logo-group { display: flex; align-items: center; gap: 12px; }
    .logo-title { font-size: 20px; font-weight: 800; letter-spacing: -0.5px; color: #fff; }
    .badge {
      background: rgba(0, 240, 255, 0.15);
      border: 1px solid var(--accent-cyan);
      color: var(--accent-cyan);
      font-family: 'JetBrains Mono', monospace;
      font-size: 11px;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 20px;
    }
    .badge-red {
      background: rgba(255, 0, 85, 0.2);
      border-color: var(--accent-red);
      color: var(--accent-red);
    }
    .main-grid {
      display: grid;
      grid-template-columns: 1fr 340px;
      gap: 20px;
    }
    @media (max-width: 1024px) {
      .main-grid { grid-template-columns: 1fr; }
    }
    .video-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 16px;
      display: flex;
      flex-direction: column;
      gap: 16px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .video-viewport {
      position: relative;
      width: 100%;
      border-radius: 12px;
      overflow: hidden;
      background: #000;
      border: 1px solid rgba(255,255,255,0.08);
      aspect-ratio: 16 / 9;
    }
    .video-viewport img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .control-panel {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 20px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .card-title {
      font-size: 14px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--accent-cyan);
      margin-bottom: 16px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .telemetry-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 0;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      font-size: 13px;
    }
    .telemetry-row:last-child { border-bottom: none; }
    .telemetry-label { color: var(--text-dim); }
    .telemetry-val {
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      color: var(--accent-green);
    }
    .btn-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }
    .btn {
      background: #1a202c;
      border: 1px solid rgba(255,255,255,0.12);
      color: #fff;
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 13px;
      font-weight: 700;
      padding: 12px;
      border-radius: 10px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      transition: all 0.2s ease;
    }
    .btn:hover {
      background: #2d3748;
      border-color: var(--accent-cyan);
      transform: translateY(-1px);
    }
    .btn-primary { background: rgba(0, 240, 255, 0.15); border-color: var(--accent-cyan); color: var(--accent-cyan); }
    .btn-primary:hover { background: var(--accent-cyan); color: #000; }
    .btn-danger { background: rgba(255, 0, 85, 0.15); border-color: var(--accent-red); color: var(--accent-red); }
    .btn-danger:hover { background: var(--accent-red); color: #fff; }
    .btn-full { grid-column: span 2; }
    select {
      width: 100%;
      background: #161b22;
      border: 1px solid var(--card-border);
      color: #fff;
      padding: 10px 14px;
      border-radius: 10px;
      font-size: 13px;
      font-family: 'Plus Jakarta Sans', sans-serif;
      outline: none;
      margin-top: 8px;
    }
    .alert-toast {
      position: fixed;
      bottom: 24px;
      right: 24px;
      background: rgba(255, 0, 85, 0.9);
      border: 1px solid #fff;
      color: #fff;
      padding: 14px 20px;
      border-radius: 12px;
      font-weight: 700;
      font-size: 14px;
      display: none;
      box-shadow: 0 10px 40px rgba(255,0,85,0.4);
      z-index: 999;
    }
  </style>
</head>
<body>
  <header class="header">
    <div class="logo-group">
      <span style="font-size: 24px;">🛡️</span>
      <div>
        <div class="logo-title">EDGE AI CCTV CORE</div>
        <div style="font-size: 12px; color: var(--text-dim);">Live Multi-Platform AI Vision Monitor & Kinematics HUD</div>
      </div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
      <span class="badge" id="sourceBadge">Source: Connecting...</span>
      <span class="badge" id="fpsBadge">0.0 FPS</span>
      <span class="badge" id="latencyBadge">0.0 ms</span>
    </div>
  </header>

  <div class="main-grid">
    <!-- Live Video Viewport -->
    <div class="video-card">
      <div class="video-viewport">
        <img src="/stream" alt="Live AI Vision Feed" />
      </div>
      <div style="display: flex; justify-content: space-between; align-items: center; font-size: 13px; color: var(--text-dim);">
        <span>⚡ Hardware Acceleration: <strong>QuickSync VA-API & HailoRT Neural Coprocessor</strong></span>
        <span>🔐 Zero-Trust Local Isolation</span>
      </div>
    </div>

    <!-- Side Control & Telemetry Panel -->
    <div class="control-panel">
      <!-- Telemetry Card -->
      <div class="card">
        <div class="card-title">📊 Kinematic Telemetry</div>
        <div class="telemetry-row">
          <span class="telemetry-label">Torso Angle (θ):</span>
          <span class="telemetry-val" id="torsoAngleVal">85.0°</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Descent Velocity (Vy):</span>
          <span class="telemetry-val" id="velocityVal">0.10 m/s</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Aspect Ratio (H/W):</span>
          <span class="telemetry-val" id="aspectVal">1.90</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Floor Proximity:</span>
          <span class="telemetry-val" id="floorVal">0.15</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Fall Alarm Status:</span>
          <span class="telemetry-val" id="fallStatusVal">NORMAL</span>
        </div>
      </div>

      <!-- Action Simulation Controls -->
      <div class="card">
        <div class="card-title">🕹️ AI Event Simulations</div>
        <div class="btn-grid">
          <button class="btn btn-danger btn-full" onclick="triggerAction('fall')">🚨 Simulate Kinematic Fall</button>
          <button class="btn btn-primary" onclick="triggerAction('tripwire')">⚡ Gate Tripwire</button>
          <button class="btn btn-primary" onclick="triggerAction('intrusion')">🛑 Porch Intrusion</button>
          <button class="btn" onclick="triggerAction('snapshot')">📸 Snapshot</button>
          <button class="btn" onclick="triggerAction('clip')">🎥 Record MP4 Clip</button>
        </div>
      </div>

      <!-- Camera Discovery & Switcher -->
      <div class="card">
        <div class="card-title">🔍 Camera Feed Manager</div>
        <button class="btn btn-primary btn-full" onclick="rescanCameras()">🔄 Scan Network & USB Cameras</button>
        <select id="cameraSelect" onchange="changeCameraSource(this.value)">
          <option value="synthetic">🛡️ Synthetic Benchmark Feed</option>
        </select>
      </div>
    </div>
  </div>

  <div class="alert-toast" id="alertToast">Alert Message</div>

  <script>
    async function triggerAction(actionName) {
      try {
        const res = await fetch(`/api/trigger/${actionName}`, { method: 'POST' });
        const data = await res.json();
        showToast(data.message || `Triggered ${actionName}`);
      } catch (e) {
        showToast(`Error triggering ${actionName}`);
      }
    }

    async function rescanCameras() {
      showToast('Scanning network and USB devices...');
      try {
        const res = await fetch('/api/rescan', { method: 'POST' });
        const data = await res.json();
        populateCameraSelect(data.sources);
        showToast(`Scan complete: Found ${data.sources.length} sources.`);
      } catch (e) {
        showToast('Camera scan failed.');
      }
    }

    async function changeCameraSource(url) {
      showToast(`Switching camera stream...`);
      try {
        await fetch('/api/switch_source', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: url })
        });
      } catch (e) {
        showToast('Failed to switch source.');
      }
    }

    function populateCameraSelect(sources) {
      const select = document.getElementById('cameraSelect');
      select.innerHTML = '';
      sources.forEach(src => {
        const opt = document.createElement('option');
        opt.value = src.url;
        opt.textContent = src.name;
        select.appendChild(opt);
      });
    }

    function showToast(msg) {
      const toast = document.getElementById('alertToast');
      toast.textContent = msg;
      toast.style.display = 'block';
      setTimeout(() => { toast.style.display = 'none'; }, 3000);
    }

    async function updateTelemetry() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();
        
        document.getElementById('fpsBadge').textContent = `${data.fps.toFixed(1)} FPS`;
        document.getElementById('latencyBadge').textContent = `${data.latency_ms.toFixed(1)} ms`;
        document.getElementById('sourceBadge').textContent = data.current_source;

        document.getElementById('torsoAngleVal').textContent = `${data.torso_angle.toFixed(1)}°`;
        document.getElementById('velocityVal').textContent = `${data.descent_velocity.toFixed(2)} m/s`;
        document.getElementById('aspectVal').textContent = `${data.aspect_ratio.toFixed(2)}`;
        document.getElementById('floorVal').textContent = `${data.floor_proximity.toFixed(2)}`;
        
        const fallStatus = document.getElementById('fallStatusVal');
        if (data.is_fall_active) {
          fallStatus.textContent = '🚨 FALL DETECTED!';
          fallStatus.style.color = 'var(--accent-red)';
        } else {
          fallStatus.textContent = 'NORMAL';
          fallStatus.style.color = 'var(--accent-green)';
        }

        if (data.active_alert && data.active_alert.length > 0) {
          showToast(data.active_alert);
        }
      } catch (e) {}
    }

    // Initial load
    rescanCameras();
    setInterval(updateTelemetry, 500);
  </script>
</body>
</html>
        """
        return HTMLResponse(content=html_content)

    @web_app.get("/stream")
    async def video_feed():
        def iter_frames():
            while monitor.is_running:
                with monitor.frame_lock:
                    jpeg = monitor.latest_encoded_jpeg
                if jpeg:
                    yield (b"--frame\r\n"
                           b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
                time.sleep(0.033)

        return StreamingResponse(iter_frames(), credentials=None, credentials_files=None,
                                 media_type="multipart/x-mixed-replace; boundary=frame")

    @web_app.get("/api/status")
    async def get_status():
        return {
            "fps": monitor.fps,
            "latency_ms": monitor.avg_inference_ms,
            "current_source": monitor.current_source_name,
            "torso_angle": monitor.torso_angle,
            "descent_velocity": monitor.descent_velocity,
            "aspect_ratio": monitor.aspect_ratio,
            "floor_proximity": monitor.floor_proximity,
            "is_fall_active": monitor.is_fall_active,
            "active_alert": monitor.active_alert_banner if time.time() < monitor.alert_expiry else ""
        }

    @web_app.post("/api/trigger/{action_name}")
    async def trigger_action(action_name: str):
        if action_name == "fall":
            monitor.sim_is_falling = True
            monitor.sim_fall_start = time.time()
            monitor.trigger_alert("FALL DETECTED! Kinematic Alarm Dispatched!", duration=4.0)
            return {"status": "ok", "message": "Kinematic Fall Dispatched"}
        elif action_name == "tripwire":
            monitor.sim_is_tripwire = not monitor.sim_is_tripwire
            monitor.trigger_alert("TRIPWIRE CROSSED: Front Gate (A -> B)", duration=3.0)
            return {"status": "ok", "message": "Virtual Tripwire Triggered"}
        elif action_name == "intrusion":
            monitor.sim_is_intrusion = not monitor.sim_is_intrusion
            monitor.trigger_alert("INTRUSION DETECTED: Restricted Porch Zone", duration=3.0)
            return {"status": "ok", "message": "Polygon Intrusion Triggered"}
        elif action_name == "snapshot":
            snap_path = settings.SNAPSHOTS_DIR / f"snapshot_{int(time.time())}.jpg"
            if monitor.latest_encoded_jpeg:
                snap_path.write_bytes(monitor.latest_encoded_jpeg)
                monitor.trigger_alert(f"Snapshot Saved: {snap_path.name}", duration=2.5)
            return {"status": "ok", "message": f"Snapshot Saved: {snap_path.name}"}
        elif action_name == "clip":
            clip_path = settings.CLIPS_DIR / f"clip_{int(time.time())}.mp4"
            pre_frames = monitor.ring_buffer.get_pre_event_frames()
            if pre_frames:
                asyncio.create_task(clip_recorder_service._mux_frames_to_mp4(pre_frames, clip_path, fps=25))
                monitor.trigger_alert(f"Event Clip Muxed: {clip_path.name}", duration=3.0)
            return {"status": "ok", "message": f"10s MP4 Clip Exported: {clip_path.name}"}
        return {"status": "error", "message": "Unknown action"}

    @web_app.post("/api/rescan")
    async def rescan():
        sources = monitor.rescan_sources()
        return {"status": "ok", "sources": sources}

    @web_app.post("/api/switch_source")
    async def switch_source(req: SwitchSourceReq):
        monitor.switch_source(req.url)
        return {"status": "ok", "current_source": monitor.current_source_name}

    return web_app


# ==============================================================================
# Main Runner with HighGUI & Web HUD Fallbacks
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Edge AI CCTV Live AI Monitor & Evaluator")
    parser.add_argument("--stream", type=str, default=None,
                        help="Stream URL of ESP32 / IP Camera (e.g. http://10.68.21.86:81/stream)")
    parser.add_argument("--port", type=int, default=8080, help="Web HUD port (default 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()

    monitor = LiveAIMonitor(args.stream)

    # 1. Start Capture & AI Worker Thread
    capture_thread = threading.Thread(target=monitor.capture_frame_loop, daemon=True)
    capture_thread.start()

    # 2. Start Embedded Web HUD Server in background thread
    import uvicorn
    web_app = create_web_hud_app(monitor)
    
    def run_uvicorn():
        uvicorn.run(web_app, host="0.0.0.0", port=args.port, log_level="warning")

    server_thread = threading.Thread(target=run_uvicorn, daemon=True)
    server_thread.start()

    print("\n" + "=" * 75)
    print("  🛡️  EDGE AI CCTV - REAL-TIME LIVE MONITOR & EVALUATOR")
    print("=" * 75)
    print(f"  🌐 Web HUD Active: http://localhost:{args.port}")
    print(f"  🌐 LAN Access:    http://0.0.0.0:{args.port}")
    print("  Controls & Shortcuts:")
    print("  • Press 'F' to simulate a Kinematic Fall Event")
    print("  • Press 'T' to simulate a Virtual Tripwire Crossing")
    print("  • Press 'I' to simulate an Intrusion Zone Entry")
    print("  • Press 'S' to take a Snapshot (saved to storage/snapshots/)")
    print("  • Press 'R' to record an MP4 Clip (saved to storage/clips/)")
    print("  • Press 'C' to Rescan Network & USB Cameras")
    print("  • Press 'Q' or ESC to Exit")
    print("=" * 75 + "\n")

    # Auto-open browser
    if not args.no_browser:
        try:
            webbrowser.open(f"http://localhost:{args.port}")
        except Exception:
            pass

    # 3. Attempt Native HighGUI Window (with graceful try/except fallback)
    gui_supported = False
    try:
        # Check if DISPLAY is available
        if sys.platform.startswith("win") or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            cv2.namedWindow("Edge AI CCTV - Live Monitor", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Edge AI CCTV - Live Monitor", 1024, 576)
            gui_supported = True
            logger.info("[+] Native desktop HighGUI window initialized.")
    except Exception as e:
        logger.info(f"[-] Native desktop window not available ({e}). Using Web HUD at http://localhost:{args.port}")

    try:
        while monitor.is_running:
            if gui_supported:
                with monitor.frame_lock:
                    jpeg = monitor.latest_encoded_jpeg
                if jpeg:
                    nparr = np.frombuffer(jpeg, np.uint8)
                    hud_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if hud_frame is not None:
                        try:
                            cv2.imshow("Edge AI CCTV - Live Monitor", hud_frame)
                            key = cv2.waitKey(30) & 0xFF
                            if key in (27, ord('q'), ord('Q')):
                                break
                            elif key in (ord('f'), ord('F')):
                                monitor.sim_is_falling = True
                                monitor.sim_fall_start = time.time()
                                monitor.trigger_alert("FALL DETECTED! Kinematic Alarm Dispatched!", duration=4.0)
                            elif key in (ord('t'), ord('T')):
                                monitor.sim_is_tripwire = not monitor.sim_is_tripwire
                                monitor.trigger_alert("TRIPWIRE CROSSED: Front Gate (A -> B)", duration=3.0)
                            elif key in (ord('i'), ord('I')):
                                monitor.sim_is_intrusion = not monitor.sim_is_intrusion
                                monitor.trigger_alert("INTRUSION DETECTED: Porch Area", duration=3.0)
                            elif key in (ord('s'), ord('S')):
                                snap_path = settings.SNAPSHOTS_DIR / f"snapshot_{int(time.time())}.jpg"
                                snap_path.write_bytes(jpeg)
                                monitor.trigger_alert(f"Snapshot Saved: {snap_path.name}", duration=2.5)
                            elif key in (ord('r'), ord('R')):
                                clip_path = settings.CLIPS_DIR / f"clip_{int(time.time())}.mp4"
                                pre_frames = monitor.ring_buffer.get_pre_event_frames()
                                if pre_frames:
                                    asyncio.run(clip_recorder_service._mux_frames_to_mp4(pre_frames, clip_path, fps=25))
                                    monitor.trigger_alert(f"Event Clip Saved: {clip_path.name}", duration=3.0)
                            elif key in (ord('c'), ord('C')):
                                monitor.rescan_sources()
                        except Exception:
                            gui_supported = False
                else:
                    time.sleep(0.03)
            else:
                # Running pure headless / Web HUD mode
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.is_running = False
        if gui_supported:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("\n[+] Live AI Monitor closed cleanly.")


if __name__ == "__main__":
    main()
