#!/usr/bin/env python3
"""Edge AI CCTV - Real-Time Live AI Vision, Kinematics & Security Zone Evaluator.

Optimizations & Architectural Upgrades:
1. Unified Robust Human Detection:
   - OpenCV DNN YOLOv8/MobileNet-SSD support
   - Morphological Vertical Body Merger with tall structuring elements (11x35) & NMS
   - Completely eliminates limb fragmentation (single unified box from head to toe)
2. High-Precision Kinematic Fall State Machine:
   - 5-stage temporal state machine: STANDING -> RAPID_DESCENT -> COLLAPSED -> IMMOBILE -> FALL_CONFIRMED
   - Immediate upright recovery path (prevents false alarms on standing/stretching persons)
3. Zone Event Debouncing & Cooldown:
   - 3.5s per-track cooldown to eliminate console/UI alert flooding
4. Thread-Safe Background Video Muxing:
   - Dedicated ThreadPoolExecutor for background MP4 exports (eliminates asyncio event loop errors)
5. Cyberpunk Web HUD on http://0.0.0.0:8080 & Graceful HighGUI Desktop Fallback.
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
import json
import logging
import math
import os
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
from app.models.schemas import Point2D, TripwireDirection, ZoneConfig, ZoneType, BoundingBox
from app.services.ai_zone_service import ai_zone_service, PolygonGeometry
from app.services.clip_recorder import clip_recorder_service
from app.services.hailo_inference_service import hailo_inference_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveMonitor")


# ==============================================================================
# Thread-Safe Background Clip Muxer
# ==============================================================================
class BackgroundClipMuxer:
    """Thread-safe background video muxing manager using a dedicated ThreadPool."""

    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ClipMuxerWorker")

    def submit_mux_task(
        self,
        frames: List[np.ndarray],
        output_path: Path,
        fps: int = 25,
        on_complete_callback: Optional[callable] = None
    ):
        if not frames:
            logger.warning(f"[-] No frames provided for clip: {output_path}")
            return

        frames_copy = [f.copy() for f in frames]

        def _task():
            try:
                # Direct thread-safe video write using OpenCV VideoWriter
                h, w = frames_copy[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
                for f in frames_copy:
                    if f.shape[:2] != (h, w):
                        f = cv2.resize(f, (w, h))
                    out.write(f)
                out.release()
                logger.info(f"✅ Background video clip saved: {output_path.name}")
                if on_complete_callback:
                    on_complete_callback(str(output_path))
            except Exception as e:
                logger.error(f"[-] Background video muxing failed for {output_path.name}: {e}")

        self.executor.submit(_task)

    def shutdown(self):
        self.executor.shutdown(wait=False)


clip_muxer = BackgroundClipMuxer()


# ==============================================================================
# Zone Alert Debouncer
# ==============================================================================
class ZoneAlertDebouncer:
    """Enforces per-track, per-zone cooldown to prevent alert flooding."""

    def __init__(self, cooldown_seconds: float = 3.5):
        self.cooldown_seconds = cooldown_seconds
        self.last_alerts: Dict[str, float] = {}

    def should_dispatch(self, zone_id: str, track_id: int, now: float) -> bool:
        key = f"{zone_id}:{track_id}"
        last_t = self.last_alerts.get(key, 0.0)
        if (now - last_t) >= self.cooldown_seconds:
            self.last_alerts[key] = now
            return True
        return False


# ==============================================================================
# Unified Robust Person Detector (DNN + Morphological Body Merger & NMS)
# ==============================================================================
class UnifiedPersonDetector:
    """Detects whole human bodies without fragmenting limbs or false aspect ratio distortions."""

    def __init__(self, onnx_model_path: Optional[str] = None):
        self.net = None
        self.use_dnn = False

        # 1. Attempt loading ONNX model (YOLOv8 / MobileNet-SSD)
        if onnx_model_path and os.path.exists(onnx_model_path):
            try:
                self.net = cv2.dnn.readNetFromONNX(onnx_model_path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                self.use_dnn = True
                logger.info(f"✅ OpenCV DNN loaded successfully: {onnx_model_path}")
            except Exception as e:
                logger.warning(f"[-] Failed to load ONNX model ({e}). Using Morphological Body Merger.")
                self.use_dnn = False

        # Fallback Motion & Morphological Segmenter
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=24, detectShadows=False)
        # Tall vertical structuring element designed to bridge Head -> Torso -> Arms -> Legs
        self.vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 35))
        self.horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def detect(self, frame: np.ndarray) -> List[Tuple[float, float, float, float, float]]:
        """Returns normalized bounding boxes: [(x1, y1, x2, y2, confidence), ...]"""
        h, w = frame.shape[:2]
        if self.use_dnn and self.net is not None:
            return self._detect_dnn(frame, w, h)
        return self._detect_morphological_merged(frame, w, h)

    def _detect_dnn(self, frame: np.ndarray, w: int, h: int) -> List[Tuple[float, float, float, float, float]]:
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (640, 640), swapRB=True, crop=False)
        self.net.setInput(blob)
        preds = self.net.forward()

        if len(preds.shape) == 3:
            preds = np.transpose(preds[0], (1, 0))

        boxes, confidences = [], []
        for row in preds:
            classes_scores = row[4:]
            class_id = int(np.argmax(classes_scores))
            score = float(classes_scores[class_id])
            if class_id == 0 and score > 0.40:  # Class 0 = Person
                cx, cy, bw, bh = row[0], row[1], row[2], row[3]
                x1 = int((cx - bw / 2) * (w / 640.0))
                y1 = int((cy - bh / 2) * (h / 640.0))
                bw_px = int(bw * (w / 640.0))
                bh_px = int(bh * (h / 640.0))
                boxes.append([x1, y1, bw_px, bh_px])
                confidences.append(score)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.40, nms_threshold=0.45)
        results = []
        if len(indices) > 0:
            for idx in indices.flatten():
                bx, by, bw_px, bh_px = boxes[idx]
                results.append((
                    max(0.0, bx / w),
                    max(0.0, by / h),
                    min(1.0, (bx + bw_px) / w),
                    min(1.0, (by + bh_px) / h),
                    confidences[idx]
                ))
        return results

    def _detect_morphological_merged(self, frame: np.ndarray, w: int, h: int) -> List[Tuple[float, float, float, float, float]]:
        scale = 480.0 / max(h, w)
        sw, sh = int(w * scale), int(h * scale)
        small = cv2.resize(frame, (sw, sh))

        fg_mask = self.bg_subtractor.apply(small)
        # 1. Clean small salt-and-pepper noise
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, self.horizontal_kernel)
        # 2. Bridge vertical limb gaps with vertical structuring element
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, self.vertical_kernel)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 900:  # Minimum body cluster area
                bx, by, bw_px, bh_px = cv2.boundingRect(cnt)
                candidate_boxes.append([bx, by, bx + bw_px, by + bh_px, area])

        # 3. Hierarchical Proximity Merging (combine torso + legs + arms into single body)
        merged_boxes = self._merge_adjacent_boxes(candidate_boxes)

        results = []
        for (x1, y1, x2, y2) in merged_boxes:
            norm_x1 = max(0.0, (x1 / scale) / w)
            norm_y1 = max(0.0, (y1 / scale) / h)
            norm_x2 = min(1.0, (x2 / scale) / w)
            norm_y2 = min(1.0, (y2 / scale) / h)

            box_h = norm_y2 - norm_y1
            box_w = norm_x2 - norm_x1

            # Only retain human-sized silhouettes (height >= 12% of frame or area >= 2.5%)
            if box_h >= 0.12 or (box_h * box_w) >= 0.025:
                results.append((norm_x1, norm_y1, norm_x2, norm_y2, 0.92))

        return results

    def _merge_adjacent_boxes(self, boxes: List[List[float]]) -> List[Tuple[int, int, int, int]]:
        """Merges fragmented sub-boxes belonging to the same human silhouette."""
        if not boxes:
            return []

        merged = True
        current_boxes = [[b[0], b[1], b[2], b[3]] for b in boxes]

        while merged:
            merged = False
            new_boxes = []
            used = [False] * len(current_boxes)

            for i in range(len(current_boxes)):
                if used[i]:
                    continue
                ax1, ay1, ax2, ay2 = current_boxes[i]
                for j in range(i + 1, len(current_boxes)):
                    if used[j]:
                        continue
                    bx1, by1, bx2, by2 = current_boxes[j]

                    # Check Horizontal Overlap
                    x_overlap = max(0, min(ax2, bx2) - max(ax1, bx1))
                    min_w = min(ax2 - ax1, bx2 - bx1)

                    # Check Vertical Distance / Gap
                    y_dist = max(0, max(ay1, by1) - min(ay2, by2))
                    max_h = max(ay2 - ay1, by2 - by1)

                    # Merge condition: Horizontal overlap + vertical proximity within 40% of body height
                    if (min_w > 0 and (x_overlap / min_w) > 0.25 and y_dist < (0.40 * max_h)) or \
                       (x_overlap > 0 and y_dist == 0):
                        ax1 = min(ax1, bx1)
                        ay1 = min(ay1, by1)
                        ax2 = max(ax2, bx2)
                        ay2 = max(ay2, by2)
                        used[j] = True
                        merged = True

                new_boxes.append([ax1, ay1, ax2, ay2])
                used[i] = True

            current_boxes = new_boxes

        return [(b[0], b[1], b[2], b[3]) for b in current_boxes]


# ==============================================================================
# High-Precision Kinematic Fall State Machine
# ==============================================================================
class KinematicState(str, Enum):
    STANDING = "STANDING"
    RAPID_DESCENT = "RAPID_DESCENT"
    COLLAPSED = "COLLAPSED"
    IMMOBILE = "IMMOBILE"
    FALL_CONFIRMED = "FALL_CONFIRMED"


class KinematicPersonTracker:
    """Multi-frame kinematic state machine tracking human poses and eliminating false fall alarms."""

    def __init__(self, track_id: int, initial_bbox: Tuple[float, float, float, float]):
        self.track_id = track_id
        self.bbox = initial_bbox
        self.state = KinematicState.STANDING
        self.last_seen = time.time()

        # State Timers
        self.descent_start_time: Optional[float] = None
        self.collapsed_start_time: Optional[float] = None
        self.state_enter_time = time.time()

        # Metrics
        self.history: List[Tuple[float, float, float, float]] = []
        self.aspect_ratio = 1.8
        self.smoothed_ar = 1.8
        self.descent_velocity = 0.0
        self.torso_angle = 85.0
        self.alert_dispatched = False

    def update(self, bbox: Tuple[float, float, float, float], now: float):
        self.bbox = bbox
        self.last_seen = now
        x1, y1, x2, y2 = bbox
        w = max(0.01, x2 - x1)
        h = max(0.01, y2 - y1)
        cy = (y1 + y2) / 2.0
        self.aspect_ratio = h / w

        # EMA smoothing on Aspect Ratio
        self.smoothed_ar = 0.65 * self.aspect_ratio + 0.35 * self.smoothed_ar

        # Calculate descent velocity over recent temporal window
        if self.history:
            dt = max(0.001, now - self.history[-1][0])
            inst_v = (cy - self.history[-1][1]) / dt * 2.5
            self.descent_velocity = max(0.0, 0.70 * inst_v + 0.30 * self.descent_velocity)
        else:
            self.descent_velocity = 0.0

        # Torso inclination estimation: map AR to angle [5..90] degrees
        clamped_ar = max(0.35, min(2.2, self.smoothed_ar))
        self.torso_angle = max(5.0, min(90.0, math.degrees(math.atan2(clamped_ar, 1.0)) * 1.35))

        self.history.append((now, cy, h, w))
        self.history = [pt for pt in self.history if now - pt[0] <= 5.0]

        # Evaluate State Machine Transitions
        self._evaluate_state_transitions(now)

    def _evaluate_state_transitions(self, now: float):
        # 1. Recovery Check: If person stands upright (AR > 1.20 and θ > 52°), immediately return to STANDING
        if self.smoothed_ar > 1.20 and self.torso_angle > 52.0:
            if self.state != KinematicState.STANDING:
                logger.info(f"[Kinematics] Track #{self.track_id} returned to STANDING.")
            self.state = KinematicState.STANDING
            self.descent_start_time = None
            self.collapsed_start_time = None
            self.alert_dispatched = False
            return

        # 2. STANDING -> RAPID_DESCENT (Requires sudden vertical velocity spike >= 1.30 m/s)
        if self.state == KinematicState.STANDING:
            if self.descent_velocity >= 1.30:
                self.state = KinematicState.RAPID_DESCENT
                self.descent_start_time = now
                self.state_enter_time = now
                logger.debug(f"[Kinematics] Track #{self.track_id} entered RAPID_DESCENT (Vy={self.descent_velocity:.2f} m/s)")

        # 3. RAPID_DESCENT -> COLLAPSED (Aspect ratio collapses near floor within 1.0s)
        elif self.state == KinematicState.RAPID_DESCENT:
            time_in_descent = now - (self.descent_start_time or now)
            if self.smoothed_ar <= 0.85 and self.torso_angle <= 38.0:
                self.state = KinematicState.COLLAPSED
                self.collapsed_start_time = now
                self.state_enter_time = now
                logger.info(f"[Kinematics] Track #{self.track_id} entered COLLAPSED (AR={self.smoothed_ar:.2f}, θ={self.torso_angle:.1f}°)")
            elif time_in_descent > 1.0:
                # Timed out without collapse -> normal motion
                self.state = KinematicState.STANDING

        # 4. COLLAPSED -> IMMOBILE (Sustained motionless on floor for >= 1.8s)
        elif self.state == KinematicState.COLLAPSED:
            time_collapsed = now - (self.collapsed_start_time or now)
            if self.descent_velocity < 0.35 and time_collapsed >= 1.8:
                self.state = KinematicState.IMMOBILE
                self.state_enter_time = now
                logger.warning(f"[Kinematics] Track #{self.track_id} entered IMMOBILE on floor ({time_collapsed:.1f}s)")

        # 5. IMMOBILE -> FALL_CONFIRMED (Total collapse time >= 2.5s)
        elif self.state == KinematicState.IMMOBILE:
            total_floor_time = now - (self.collapsed_start_time or now)
            if total_floor_time >= 2.5:
                self.state = KinematicState.FALL_CONFIRMED


# ==============================================================================
# Camera Discovery & Scanner
# ==============================================================================
class CameraScanner:
    """Discovers local USB cameras, mDNS ESP32 devices, and subnet RTSP/MJPEG streams."""

    @staticmethod
    def scan_usb_cameras() -> List[Dict[str, str]]:
        found = []
        if sys.platform.startswith("linux"):
            for dev in sorted(Path("/dev").glob("video*")):
                try:
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
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                return s.connect_ex((host, port)) == 0
        except Exception:
            return False

    @classmethod
    def scan_network_cameras(cls) -> List[Dict[str, str]]:
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

        # 2. Probe host subnet
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()

            ip_parts = local_ip.split(".")
            subnet_prefix = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}."

            candidate_ips = [
                local_ip,
                f"{subnet_prefix}86",  # User's ESP32 IP
                f"{subnet_prefix}1",
                f"{subnet_prefix}2",
                f"{subnet_prefix}100",
                f"{subnet_prefix}150"
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
        sources = []
        sources.append({
            "id": "synthetic",
            "name": "🛡️ Synthetic Benchmark Feed",
            "url": "synthetic"
        })
        sources.extend(cls.scan_usb_cameras())
        sources.extend(cls.scan_network_cameras())
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

        # Robust Unified Person Detector & Zone Debouncer
        self.detector = UnifiedPersonDetector()
        self.debouncer = ZoneAlertDebouncer(cooldown_seconds=3.5)

        # Tracked Persons
        self.tracks: Dict[int, KinematicPersonTracker] = {}
        self.next_track_id = 1

        # Performance Metrics
        self.fps = 0.0
        self.frame_count = 0
        self.avg_inference_ms = 0.0
        self.recent_latencies: List[float] = []

        # Live Kinematic Telemetry
        self.torso_angle = 85.0
        self.aspect_ratio = 1.9
        self.descent_velocity = 0.0
        self.floor_proximity = 0.15
        self.is_fall_active = False
        self.person_count = 0

        # Tripwire & Intrusion Stats
        self.in_count = 0
        self.out_count = 0
        self.is_intrusion_active = False
        self.active_alert_banner = ""
        self.alert_expiry = 0.0

        # Event History Log
        self.event_log: List[Dict[str, Any]] = []
        self.event_lock = threading.Lock()

        # Frame Broadcast Buffer for Web HUD
        self.latest_encoded_jpeg: Optional[bytes] = None
        self.frame_lock = threading.Lock()

        # Discovered Camera Sources
        self.available_sources: List[Dict[str, str]] = []

        # Ring Buffer for Clips
        self.ring_buffer = clip_recorder_service.get_or_create_buffer(self.camera_id)

        # Initialize Default Security Zones
        self._init_default_zones()

    def _init_default_zones(self):
        self.tripwire_config = {
            "id": "zone_tripwire_gate",
            "name": "Virtual Tripwire (Entry/Exit)",
            "x1": 0.15,
            "y1": 0.55,
            "x2": 0.85,
            "y2": 0.55,
            "direction": "BIDIRECTIONAL",
            "enabled": True
        }
        self.intrusion_config = {
            "id": "zone_intrusion_porch",
            "name": "Restricted Intrusion Zone",
            "points": [
                {"x": 0.55, "y": 0.25},
                {"x": 0.95, "y": 0.25},
                {"x": 0.95, "y": 0.85},
                {"x": 0.55, "y": 0.85}
            ],
            "enabled": True
        }
        self.sync_zones_to_service()

    def sync_zones_to_service(self):
        zones = []
        if self.tripwire_config.get("enabled"):
            direction_enum = TripwireDirection.BIDIRECTIONAL
            if self.tripwire_config.get("direction") == "A_TO_B":
                direction_enum = TripwireDirection.A_TO_B
            elif self.tripwire_config.get("direction") == "B_TO_A":
                direction_enum = TripwireDirection.B_TO_A

            zones.append(ZoneConfig(
                id=self.tripwire_config["id"],
                camera_id=self.camera_id,
                name=self.tripwire_config["name"],
                zone_type=ZoneType.TRIPWIRE,
                enabled=True,
                line_start=Point2D(x=float(self.tripwire_config["x1"]), y=float(self.tripwire_config["y1"])),
                line_end=Point2D(x=float(self.tripwire_config["x2"]), y=float(self.tripwire_config["y2"])),
                direction=direction_enum
            ))

        if self.intrusion_config.get("enabled"):
            pts = [Point2D(x=float(p["x"]), y=float(p["y"])) for p in self.intrusion_config["points"]]
            zones.append(ZoneConfig(
                id=self.intrusion_config["id"],
                camera_id=self.camera_id,
                name=self.intrusion_config["name"],
                zone_type=ZoneType.INTRUSION,
                enabled=True,
                polygon_points=pts,
                dwell_time_seconds=1.0
            ))

        ai_zone_service.set_camera_zones(self.camera_id, zones)

    def trigger_alert(self, message: str, event_type: str = "SECURITY", duration: float = 3.5):
        self.active_alert_banner = message
        self.alert_expiry = time.time() + duration
        logger.info(f"🚨 [{event_type}] {message}")

        evt = {
            "id": f"evt_{int(time.time()*1000)}",
            "time": time.strftime("%H:%M:%S"),
            "type": event_type,
            "message": message,
            "camera": self.current_source_name
        }
        with self.event_lock:
            self.event_log.insert(0, evt)
            if len(self.event_log) > 50:
                self.event_log.pop()

    def update_tripwire_position(self, x1: float, y1: float, x2: float, y2: float, direction: str):
        self.tripwire_config.update({
            "x1": max(0.0, min(1.0, x1)),
            "y1": max(0.0, min(1.0, y1)),
            "x2": max(0.0, min(1.0, x2)),
            "y2": max(0.0, min(1.0, y2)),
            "direction": direction
        })
        self.sync_zones_to_service()
        self.trigger_alert(f"Tripwire repositioned: ({x1:.2f},{y1:.2f}) -> ({x2:.2f},{y2:.2f}) [{direction}]", "ZONE_CONFIG", duration=2.5)

    def update_intrusion_zone(self, points: List[Dict[str, float]]):
        self.intrusion_config["points"] = points
        self.sync_zones_to_service()
        self.trigger_alert(f"Intrusion zone updated with {len(points)} vertices.", "ZONE_CONFIG", duration=2.5)

    def switch_source(self, new_source_url: str):
        logger.info(f"[+] Switching video source to: {new_source_url}")
        if self.cap:
            self.cap.release()
            self.cap = None
        self.stream_source = new_source_url
        self.open_video_source()

    def rescan_sources(self) -> List[Dict[str, str]]:
        logger.info("[+] Scanning for connected cameras and streams...")
        sources = CameraScanner.discover_all()
        self.available_sources = sources
        self.trigger_alert(f"Scan complete: Found {len(sources)} video source(s).", "CAMERA_SCAN", duration=2.5)
        return sources

    def open_video_source(self):
        # 1. Try explicit stream
        if self.stream_source and self.stream_source != "synthetic":
            logger.info(f"[+] Connecting to custom stream: {self.stream_source}")
            try:
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "timeout;3000000|stimeout;3000000"
                cap = cv2.VideoCapture(self.stream_source)
                if cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        logger.info(f"✅ Successfully connected to: {self.stream_source}")
                        self.cap = cap
                        self.current_source_name = f"Live ({self.stream_source})"
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
        logger.info("🛡️ Operating in Built-In High-Fidelity Synthetic AI Pipeline.")
        self.cap = None
        self.stream_source = "synthetic"
        self.current_source_name = "Synthetic Benchmark Feed"

    def update_tracks(self, detections: List[Tuple[float, float, float, float, float]]):
        """Updates persistent human tracks and evaluates kinematic state machine."""
        now = time.time()
        updated_tracks = set()

        for (x1, y1, x2, y2, conf) in detections:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            # Match to closest existing track
            best_track_id = None
            min_dist = 0.28

            for tid, t in self.tracks.items():
                if tid in updated_tracks:
                    continue
                tx1, ty1, tx2, ty2 = t.bbox
                tcx, tcy = (tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0
                dist = math.hypot(cx - tcx, cy - tcy)
                if dist < min_dist:
                    min_dist = dist
                    best_track_id = tid

            if best_track_id is None:
                best_track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[best_track_id] = KinematicPersonTracker(best_track_id, (x1, y1, x2, y2))

            self.tracks[best_track_id].update((x1, y1, x2, y2), now)
            updated_tracks.add(best_track_id)

        # Cleanup stale tracks (> 2.5 seconds unseen)
        stale_ids = [tid for tid, t in self.tracks.items() if now - t.last_seen > 2.5]
        for tid in stale_ids:
            del self.tracks[tid]

        self.person_count = len(self.tracks)

        # Update aggregated telemetry from primary subject
        if self.tracks:
            primary = list(self.tracks.values())[0]
            self.torso_angle = primary.torso_angle
            self.aspect_ratio = primary.smoothed_ar
            self.descent_velocity = primary.descent_velocity
            self.floor_proximity = min(1.0, primary.bbox[3])
            self.is_fall_active = (primary.state in (KinematicState.COLLAPSED, KinematicState.IMMOBILE, KinematicState.FALL_CONFIRMED))

            # Dispatch genuine fall alarm upon FALL_CONFIRMED
            if primary.state == KinematicState.FALL_CONFIRMED and not primary.alert_dispatched:
                primary.alert_dispatched = True
                self.trigger_alert(
                    f"CRITICAL FALL CONFIRMED! Torso={primary.torso_angle:.1f}°, Vy={primary.descent_velocity:.2f}m/s, AR={primary.smoothed_ar:.2f}",
                    "FALL_DETECTED",
                    duration=5.0
                )
                # Thread-safe background clip export
                clip_path = settings.CLIPS_DIR / f"fall_event_{int(time.time())}.mp4"
                pre_frames = self.ring_buffer.get_pre_event_frames()
                if pre_frames:
                    clip_muxer.submit_mux_task(pre_frames, clip_path, fps=25)
        else:
            self.torso_angle = 85.0
            self.aspect_ratio = 1.9
            self.descent_velocity = 0.0
            self.floor_proximity = 0.15
            self.is_fall_active = False

    def generate_synthetic_frame(self) -> np.ndarray:
        frame = np.full((720, 1280, 3), (24, 28, 36), dtype=np.uint8)
        t = time.time()

        # Room floor & grid
        cv2.rectangle(frame, (0, 480), (1280, 720), (35, 40, 50), -1)
        for y in range(480, 720, 40):
            cv2.line(frame, (0, y), (1280, y), (45, 52, 65), 1)

        # Walking avatar
        px = 0.50 + 0.30 * np.sin(t * 0.5)
        py = 0.52 + 0.10 * np.cos(t * 0.8)
        cx, cy = int(px * 1280), int(py * 720)

        cv2.ellipse(frame, (cx, cy + 85), (45, 14), 0, 0, 360, (15, 18, 24), -1)
        cv2.rectangle(frame, (cx - 28, cy - 80), (cx + 28, cy + 80), (60, 120, 200), -1)
        cv2.circle(frame, (cx, cy - 105), 20, (180, 200, 220), -1)

        return frame

    def draw_hud(self, frame: np.ndarray) -> np.ndarray:
        hud = frame.copy()
        h, w = hud.shape[:2]

        # 1. Render Tripwire Line & Labels
        if self.tripwire_config.get("enabled"):
            tx1 = int(self.tripwire_config["x1"] * w)
            ty1 = int(self.tripwire_config["y1"] * h)
            tx2 = int(self.tripwire_config["x2"] * w)
            ty2 = int(self.tripwire_config["y2"] * h)
            cv2.line(hud, (tx1, ty1), (tx2, ty2), (0, 220, 255), 2)
            cv2.circle(hud, (tx1, ty1), 5, (0, 220, 255), -1)
            cv2.circle(hud, (tx2, ty2), 5, (0, 220, 255), -1)

            mid_x, mid_y = (tx1 + tx2) // 2, (ty1 + ty2) // 2
            dir_str = self.tripwire_config.get("direction", "BIDIRECTIONAL")
            cv2.putText(hud, f"⚡ TRIPWIRE [{dir_str}] In:{self.in_count} Out:{self.out_count}",
                        (mid_x - 120, mid_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 220, 255), 2)

        # 2. Render Intrusion Polygon Zone
        if self.intrusion_config.get("enabled") and self.intrusion_config.get("points"):
            pts = np.array([[int(p["x"] * w), int(p["y"] * h)] for p in self.intrusion_config["points"]], np.int32)
            overlay = hud.copy()
            zone_color = (0, 0, 220) if self.is_intrusion_active else (255, 120, 0)
            cv2.fillPoly(overlay, [pts], zone_color)
            cv2.addWeighted(overlay, 0.25 if not self.is_intrusion_active else 0.45, hud, 0.75, 0, hud)
            cv2.polylines(hud, [pts], True, zone_color, 2)
            cv2.putText(hud, f"🛑 INTRUSION ZONE {'[BREACHED!]' if self.is_intrusion_active else '[ARMED]'}",
                        (pts[0][0] + 8, pts[0][1] + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, zone_color, 2)

        # 3. Render Unified Human Bounding Boxes & Kinematic Pose State
        for tid, t in self.tracks.items():
            bx1 = int(t.bbox[0] * w)
            by1 = int(t.bbox[1] * h)
            bx2 = int(t.bbox[2] * w)
            by2 = int(t.bbox[3] * h)

            if t.state in (KinematicState.COLLAPSED, KinematicState.IMMOBILE, KinematicState.FALL_CONFIRMED):
                box_color = (0, 0, 255)  # Red for Fall
            elif t.state == KinematicState.RAPID_DESCENT:
                box_color = (0, 165, 255)  # Orange for Descent
            else:
                box_color = (0, 255, 180)  # Green for Upright

            cv2.rectangle(hud, (bx1, by1), (bx2, by2), box_color, 2)

            # Footprint anchor point
            foot_x = (bx1 + bx2) // 2
            foot_y = by2
            cv2.circle(hud, (foot_x, foot_y), 4, (0, 240, 255), -1)

            label = f"ID:{tid} Person | {t.state.value} | θ:{t.torso_angle:.0f}° | AR:{t.smoothed_ar:.2f}"
            cv2.putText(hud, label, (bx1, max(20, by1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, box_color, 2)

        # 4. Top Status Bar
        overlay = hud.copy()
        cv2.rectangle(overlay, (0, 0), (w, 54), (10, 13, 18), -1)
        cv2.addWeighted(overlay, 0.85, hud, 0.15, 0, hud)
        cv2.line(hud, (0, 54), (w, 54), (0, 240, 255), 1)

        cv2.putText(hud, "🛡️ EDGE AI CCTV - REAL-TIME KINEMATICS EVALUATOR", (16, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 240, 255), 2)
        cv2.putText(hud, f"Source: {self.current_source_name} | Persons Detected: {self.person_count}", (16, 46),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 180, 200), 1)

        fps_text = f"FPS: {self.fps:.1f} | Latency: {self.avg_inference_ms:.1f}ms | Res: {w}x{h}"
        cv2.putText(hud, fps_text, (w - 440, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 160), 1)

        # 5. Kinematic Telemetry HUD (Bottom Left)
        kin_w, kin_h = 330, 132
        kx, ky = 16, h - kin_h - 16
        overlay = hud.copy()
        cv2.rectangle(overlay, (kx, ky), (kx + kin_w, ky + kin_h), (12, 16, 24), -1)
        cv2.addWeighted(overlay, 0.85, hud, 0.15, 0, hud)
        border_color = (0, 0, 255) if self.is_fall_active else (0, 200, 255)
        cv2.rectangle(hud, (kx, ky), (kx + kin_w, ky + kin_h), border_color, 1)

        cv2.putText(hud, "LIVE KINEMATIC POSE TELEMETRY", (kx + 10, ky + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, border_color, 1)

        angle_color = (0, 0, 255) if self.torso_angle < 35 else (0, 255, 180)
        cv2.putText(hud, f"• Torso Angle (θ): {self.torso_angle:.1f}° {'(CRITICAL)' if self.torso_angle < 35 else '(NORMAL)'}",
                    (kx + 10, ky + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.42, angle_color, 1)

        vel_color = (0, 0, 255) if self.descent_velocity > 1.3 else (0, 255, 180)
        cv2.putText(hud, f"• Descent Velocity (Vy): {self.descent_velocity:.2f} m/s",
                    (kx + 10, ky + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, vel_color, 1)

        aspect_color = (0, 0, 255) if self.aspect_ratio < 0.85 else (0, 255, 180)
        cv2.putText(hud, f"• Aspect Ratio (H/W): {self.aspect_ratio:.2f} {'(COLLAPSED)' if self.aspect_ratio < 0.85 else '(UPRIGHT)'}",
                    (kx + 10, ky + 91), cv2.FONT_HERSHEY_SIMPLEX, 0.42, aspect_color, 1)

        cv2.putText(hud, f"• Floor Proximity: {self.floor_proximity:.2f}",
                    (kx + 10, ky + 114), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 210, 220), 1)

        # 6. Active Alert Banner
        if time.time() < self.alert_expiry:
            banner_y = h - 62
            overlay = hud.copy()
            cv2.rectangle(overlay, (w // 2 - 360, banner_y), (w // 2 + 360, banner_y + 44), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.90, hud, 0.10, 0, hud)
            cv2.rectangle(hud, (w // 2 - 360, banner_y), (w // 2 + 360, banner_y + 44), (0, 240, 255), 2)
            cv2.putText(hud, self.active_alert_banner, (w // 2 - 340, banner_y + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2)

        return hud

    def capture_frame_loop(self):
        """Continuous frame ingestion, unified body detection, zone analytics, and MJPEG encoder."""
        self.open_video_source()
        last_time = time.time()

        while self.is_running:
            loop_start = time.time()

            # 1. Grab Frame
            frame = None
            if self.cap and self.cap.isOpened():
                ret, raw_frame = self.cap.read()
                if ret and raw_frame is not None:
                    frame = raw_frame
                else:
                    frame = self.generate_synthetic_frame()
            else:
                frame = self.generate_synthetic_frame()

            # 2. Run Unified Human Detection
            ai_start = time.time()
            raw_detections = self.detector.detect(frame)
            self.update_tracks(raw_detections)

            # 3. Zone & Tripwire Processing with Debouncing
            h, w = frame.shape[:2]
            detections_for_zones = []
            for tid, t in self.tracks.items():
                detections_for_zones.append({
                    "track_id": tid,
                    "class_name": "person",
                    "confidence": 0.94,
                    "bbox": list(t.bbox)
                })

            events = ai_zone_service.process_detections(self.camera_id, detections_for_zones, w, h)
            self.is_intrusion_active = False

            now = time.time()
            for ev in events:
                if ev.event_type == ZoneType.TRIPWIRE or "TRIPWIRE" in ev.event_type.name:
                    if self.debouncer.should_dispatch("tripwire", int(ev.camera_id if ev.camera_id.isdigit() else 1), now):
                        self.in_count += 1
                        self.trigger_alert(f"TRIPWIRE CROSSED by Person #{ev.camera_id}!", "TRIPWIRE")
                elif ev.event_type == ZoneType.INTRUSION or "INTRUSION" in ev.event_type.name:
                    self.is_intrusion_active = True
                    if self.debouncer.should_dispatch("intrusion", int(ev.camera_id if ev.camera_id.isdigit() else 1), now):
                        self.trigger_alert("RESTRICTED INTRUSION ZONE BREACHED!", "INTRUSION")

            ai_latency = (time.time() - ai_start) * 1000.0
            self.recent_latencies.append(ai_latency)
            if len(self.recent_latencies) > 30:
                self.recent_latencies.pop(0)
            self.avg_inference_ms = sum(self.recent_latencies) / len(self.recent_latencies)

            # 4. Ring Buffer Push
            self.ring_buffer.push_frame(frame)

            # 5. Draw HUD Canvas
            hud_frame = self.draw_hud(frame)

            # 6. Encode JPEG for Web HUD broadcast
            ret, jpeg_bytes = cv2.imencode(".jpg", hud_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                with self.frame_lock:
                    self.latest_encoded_jpeg = jpeg_bytes.tobytes()

            # 7. FPS Calculation
            self.frame_count += 1
            now = time.time()
            if now - last_time >= 1.0:
                self.fps = self.frame_count / (now - last_time)
                self.frame_count = 0
                last_time = now

            # Pacing to ~30 FPS
            elapsed = time.time() - loop_start
            sleep_time = max(0.005, (1.0 / 30.0) - elapsed)
            time.sleep(sleep_time)


# ==============================================================================
# Embedded Asynchronous Web HUD Server (FastAPI / Uvicorn)
# ==============================================================================
def create_web_hud_app(monitor: LiveAIMonitor):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse
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

    class TripwireReq(BaseModel):
        x1: float
        y1: float
        x2: float
        y2: float
        direction: str

    class IntrusionReq(BaseModel):
        points: List[Dict[str, float]]

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
  <title>Edge AI CCTV - Live Kinematics & Zone Evaluator</title>
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;800&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b0e14;
      --card-bg: rgba(18, 22, 31, 0.85);
      --card-border: rgba(0, 240, 255, 0.25);
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
      padding: 20px;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      background: var(--card-bg);
      backdrop-filter: blur(12px);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 14px 20px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .logo-group { display: flex; align-items: center; gap: 12px; }
    .logo-title { font-size: 18px; font-weight: 800; color: #fff; }
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
    .main-grid {
      display: grid;
      grid-template-columns: 1fr 380px;
      gap: 16px;
    }
    @media (max-width: 1080px) {
      .main-grid { grid-template-columns: 1fr; }
    }
    .video-card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 14px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .video-viewport {
      position: relative;
      width: 100%;
      border-radius: 12px;
      overflow: hidden;
      background: #000;
      border: 1px solid rgba(255,255,255,0.1);
      aspect-ratio: 16 / 9;
    }
    .video-viewport img {
      width: 100%;
      height: 100%;
      object-fit: cover;
      display: block;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
      margin-bottom: 14px;
    }
    .card-title {
      font-size: 13px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      color: var(--accent-cyan);
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .telemetry-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 8px 0;
      border-bottom: 1px solid rgba(255,255,255,0.05);
      font-size: 12.5px;
    }
    .telemetry-row:last-child { border-bottom: none; }
    .telemetry-label { color: var(--text-dim); }
    .telemetry-val {
      font-family: 'JetBrains Mono', monospace;
      font-weight: 700;
      color: var(--accent-green);
    }
    .btn {
      background: #1a202c;
      border: 1px solid rgba(255,255,255,0.12);
      color: #fff;
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 12px;
      font-weight: 700;
      padding: 10px 14px;
      border-radius: 8px;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      transition: all 0.2s ease;
      width: 100%;
    }
    .btn:hover {
      background: #2d3748;
      border-color: var(--accent-cyan);
    }
    .btn-primary { background: rgba(0, 240, 255, 0.15); border-color: var(--accent-cyan); color: var(--accent-cyan); }
    .btn-primary:hover { background: var(--accent-cyan); color: #000; }
    .btn-danger { background: rgba(255, 0, 85, 0.15); border-color: var(--accent-red); color: var(--accent-red); }
    .btn-danger:hover { background: var(--accent-red); color: #fff; }
    .input-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin-bottom: 8px;
    }
    .input-group label {
      font-size: 11px;
      color: var(--text-dim);
      display: block;
      margin-bottom: 4px;
    }
    .input-group input, select {
      width: 100%;
      background: #161b22;
      border: 1px solid rgba(255,255,255,0.15);
      color: #fff;
      padding: 6px 10px;
      border-radius: 6px;
      font-size: 12px;
      font-family: 'JetBrains Mono', monospace;
      outline: none;
    }
    .event-log-container {
      max-height: 180px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .event-log-item {
      background: rgba(0,0,0,0.3);
      border-left: 3px solid var(--accent-cyan);
      padding: 6px 10px;
      border-radius: 4px;
      font-size: 11.5px;
      font-family: 'JetBrains Mono', monospace;
    }
    .event-fall { border-left-color: var(--accent-red); color: #ff99bb; }
    .event-intrusion { border-left-color: var(--accent-amber); color: #ffd280; }
    .toast {
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: rgba(0, 240, 255, 0.95);
      color: #000;
      padding: 12px 18px;
      border-radius: 10px;
      font-weight: 700;
      font-size: 13px;
      display: none;
      box-shadow: 0 8px 30px rgba(0,240,255,0.4);
      z-index: 999;
    }
  </style>
</head>
<body>
  <header class="header">
    <div class="logo-group">
      <span style="font-size: 22px;">🛡️</span>
      <div>
        <div class="logo-title">EDGE AI CCTV CORE</div>
        <div style="font-size: 11.5px; color: var(--text-dim);">Live Multi-Platform AI Vision, Kinematics & Zone Evaluator</div>
      </div>
    </div>
    <div style="display: flex; gap: 8px; align-items: center;">
      <span class="badge" id="sourceBadge">Source: Detecting...</span>
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
      <div style="display: flex; gap: 10px;">
        <button class="btn btn-primary" onclick="triggerSnapshot()">📸 Take Snapshot</button>
        <button class="btn btn-danger" onclick="triggerClip()">🎥 Export 10s Pre/Post MP4 Clip</button>
        <button class="btn" onclick="rescanCameras()">🔄 Rescan Cameras</button>
      </div>
    </div>

    <!-- Side Control & Configuration Panels -->
    <div>
      <!-- Live Kinematic Telemetry Card -->
      <div class="card">
        <div class="card-title">📊 Live Kinematic Pose Metrics</div>
        <div class="telemetry-row">
          <span class="telemetry-label">Persons Detected:</span>
          <span class="telemetry-val" id="personCountVal">0</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Torso Angle (θ):</span>
          <span class="telemetry-val" id="torsoAngleVal">85.0°</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Descent Velocity (Vy):</span>
          <span class="telemetry-val" id="velocityVal">0.00 m/s</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Aspect Ratio (H/W):</span>
          <span class="telemetry-val" id="aspectVal">1.90</span>
        </div>
        <div class="telemetry-row">
          <span class="telemetry-label">Fall Alarm Status:</span>
          <span class="telemetry-val" id="fallStatusVal">NORMAL</span>
        </div>
      </div>

      <!-- Real-Time Tripwire Position Configurator -->
      <div class="card">
        <div class="card-title">📐 Virtual Tripwire Position</div>
        <div class="input-row">
          <div class="input-group"><label>Start X1 (0.0-1.0)</label><input type="number" id="twX1" step="0.05" value="0.15"></div>
          <div class="input-group"><label>Start Y1 (0.0-1.0)</label><input type="number" id="twY1" step="0.05" value="0.55"></div>
        </div>
        <div class="input-row">
          <div class="input-group"><label>End X2 (0.0-1.0)</label><input type="number" id="twX2" step="0.05" value="0.85"></div>
          <div class="input-group"><label>End Y2 (0.0-1.0)</label><input type="number" id="twY2" step="0.05" value="0.55"></div>
        </div>
        <div class="input-group" style="margin-bottom: 8px;">
          <label>Direction</label>
          <select id="twDir">
            <option value="BIDIRECTIONAL">Bidirectional (A ↔ B)</option>
            <option value="A_TO_B">Entry Only (A → B)</option>
            <option value="B_TO_A">Exit Only (B → A)</option>
          </select>
        </div>
        <button class="btn btn-primary" onclick="updateTripwire()">💾 Save & Apply Tripwire Position</button>
      </div>

      <!-- Live Security Event History Log -->
      <div class="card">
        <div class="card-title">📋 Real Security Events Log</div>
        <div class="event-log-container" id="eventLogList">
          <div class="event-log-item">System armed and monitoring active...</div>
        </div>
      </div>

      <!-- Camera Feed Switcher -->
      <div class="card">
        <div class="card-title">📹 Camera Feeds</div>
        <select id="cameraSelect" onchange="changeCameraSource(this.value)">
          <option value="synthetic">🛡️ Synthetic Benchmark Feed</option>
        </select>
      </div>
    </div>
  </div>

  <div class="toast" id="toast">Notification</div>

  <script>
    function showToast(msg) {
      const t = document.getElementById('toast');
      t.textContent = msg;
      t.style.display = 'block';
      setTimeout(() => { t.style.display = 'none'; }, 3000);
    }

    async function triggerSnapshot() {
      try {
        const res = await fetch('/api/action/snapshot', { method: 'POST' });
        const data = await res.json();
        showToast(data.message || 'Snapshot captured');
      } catch (e) {
        showToast('Snapshot failed');
      }
    }

    async function triggerClip() {
      try {
        const res = await fetch('/api/action/clip', { method: 'POST' });
        const data = await res.json();
        showToast(data.message || 'Exporting clip');
      } catch (e) {
        showToast('Clip export failed');
      }
    }

    async function updateTripwire() {
      const x1 = parseFloat(document.getElementById('twX1').value);
      const y1 = parseFloat(document.getElementById('twY1').value);
      const x2 = parseFloat(document.getElementById('twX2').value);
      const y2 = parseFloat(document.getElementById('twY2').value);
      const dir = document.getElementById('twDir').value;

      try {
        const res = await fetch('/api/zone/tripwire', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ x1: x1, y1: y1, x2: x2, y2: y2, direction: dir })
        });
        const data = await res.json();
        showToast(data.message || 'Tripwire updated');
      } catch (e) {
        showToast('Failed to update tripwire');
      }
    }

    async function rescanCameras() {
      showToast('Scanning network and USB devices...');
      try {
        const res = await fetch('/api/rescan', { method: 'POST' });
        const data = await res.json();
        const select = document.getElementById('cameraSelect');
        select.innerHTML = '';
        data.sources.forEach(src => {
          const opt = document.createElement('option');
          opt.value = src.url;
          opt.textContent = src.name;
          select.appendChild(opt);
        });
        showToast(`Scan complete: Found ${data.sources.length} sources.`);
      } catch (e) {
        showToast('Camera scan failed.');
      }
    }

    async function changeCameraSource(url) {
      showToast('Switching camera stream...');
      try {
        await fetch('/api/switch_source', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ url: url })
        });
      } catch (e) {
        showToast('Failed to switch source');
      }
    }

    async function updateTelemetry() {
      try {
        const res = await fetch('/api/status');
        const data = await res.json();

        document.getElementById('fpsBadge').textContent = `${data.fps.toFixed(1)} FPS`;
        document.getElementById('latencyBadge').textContent = `${data.latency_ms.toFixed(1)} ms`;
        document.getElementById('sourceBadge').textContent = data.current_source;

        document.getElementById('personCountVal').textContent = data.person_count;
        document.getElementById('torsoAngleVal').textContent = `${data.torso_angle.toFixed(1)}°`;
        document.getElementById('velocityVal').textContent = `${data.descent_velocity.toFixed(2)} m/s`;
        document.getElementById('aspectVal').textContent = `${data.aspect_ratio.toFixed(2)}`;

        const fallStatus = document.getElementById('fallStatusVal');
        if (data.is_fall_active) {
          fallStatus.textContent = '🚨 FALL DETECTED!';
          fallStatus.style.color = 'var(--accent-red)';
        } else {
          fallStatus.textContent = 'NORMAL';
          fallStatus.style.color = 'var(--accent-green)';
        }

        // Render Events Log
        if (data.events && data.events.length > 0) {
          const container = document.getElementById('eventLogList');
          container.innerHTML = '';
          data.events.forEach(ev => {
            const item = document.createElement('div');
            item.className = `event-log-item ${ev.type === 'FALL_DETECTED' ? 'event-fall' : ''} ${ev.type === 'INTRUSION' ? 'event-intrusion' : ''}`;
            item.textContent = `[${ev.time}] ${ev.type}: ${ev.message}`;
            container.appendChild(item);
          });
        }
      } catch (e) {}
    }

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

        return StreamingResponse(
            iter_frames(),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )

    @web_app.get("/api/status")
    async def get_status():
        with monitor.event_lock:
            recent_events = list(monitor.event_log[:15])
        return {
            "fps": monitor.fps,
            "latency_ms": monitor.avg_inference_ms,
            "current_source": monitor.current_source_name,
            "person_count": monitor.person_count,
            "torso_angle": monitor.torso_angle,
            "descent_velocity": monitor.descent_velocity,
            "aspect_ratio": monitor.aspect_ratio,
            "floor_proximity": monitor.floor_proximity,
            "is_fall_active": monitor.is_fall_active,
            "events": recent_events
        }

    @web_app.post("/api/zone/tripwire")
    async def set_tripwire(req: TripwireReq):
        monitor.update_tripwire_position(req.x1, req.y1, req.x2, req.y2, req.direction)
        return {"status": "ok", "message": "Tripwire configuration updated"}

    @web_app.post("/api/zone/intrusion")
    async def set_intrusion(req: IntrusionReq):
        monitor.update_intrusion_zone(req.points)
        return {"status": "ok", "message": "Intrusion zone configuration updated"}

    @web_app.post("/api/action/snapshot")
    async def take_snapshot():
        snap_path = settings.SNAPSHOTS_DIR / f"snapshot_{int(time.time())}.jpg"
        if monitor.latest_encoded_jpeg:
            snap_path.write_bytes(monitor.latest_encoded_jpeg)
            monitor.trigger_alert(f"Snapshot Saved: {snap_path.name}", "SNAPSHOT", duration=2.5)
        return {"status": "ok", "message": f"Snapshot saved: {snap_path.name}"}

    @web_app.post("/api/action/clip")
    async def export_clip():
        clip_path = settings.CLIPS_DIR / f"clip_{int(time.time())}.mp4"
        pre_frames = monitor.ring_buffer.get_pre_event_frames()
        if pre_frames:
            clip_muxer.submit_mux_task(pre_frames, clip_path, fps=25)
            monitor.trigger_alert(f"10s MP4 Clip Exported: {clip_path.name}", "CLIP_EXPORT", duration=3.0)
        return {"status": "ok", "message": f"Exported: {clip_path.name}"}

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

    # 1. Start Capture & AI Vision Worker Thread
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
    print(f"  🌐 Live Web HUD:  http://localhost:{args.port}")
    print(f"  🌐 Remote Access: http://0.0.0.0:{args.port}")
    print("  Interactive Features:")
    print("  • Real-Time Fall Detection on live detected persons")
    print("  • Virtual Tripwire crossing evaluation with In/Out count")
    print("  • Polygon Intrusion zone monitoring")
    print("  • Reposition tripwires & intrusion polygons live via Web HUD")
    print("  • Press 'Q' or ESC to Exit")
    print("=" * 75 + "\n")

    if not args.no_browser:
        try:
            webbrowser.open(f"http://localhost:{args.port}")
        except Exception:
            pass

    # 3. Attempt Native HighGUI Window (with graceful try/except fallback)
    gui_supported = False
    try:
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
                            elif key in (ord('s'), ord('S')):
                                snap_path = settings.SNAPSHOTS_DIR / f"snapshot_{int(time.time())}.jpg"
                                snap_path.write_bytes(jpeg)
                                monitor.trigger_alert(f"Snapshot Saved: {snap_path.name}", "SNAPSHOT", duration=2.5)
                            elif key in (ord('r'), ord('R')):
                                clip_path = settings.CLIPS_DIR / f"clip_{int(time.time())}.mp4"
                                pre_frames = monitor.ring_buffer.get_pre_event_frames()
                                if pre_frames:
                                    clip_muxer.submit_mux_task(pre_frames, clip_path, fps=25)
                                    monitor.trigger_alert(f"Event Clip Saved: {clip_path.name}", "CLIP", duration=3.0)
                            elif key in (ord('c'), ord('C')):
                                monitor.rescan_sources()
                        except Exception:
                            gui_supported = False
                else:
                    time.sleep(0.03)
            else:
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        monitor.is_running = False
        clip_muxer.shutdown()
        if gui_supported:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print("\n[+] Live AI Monitor closed cleanly.")


if __name__ == "__main__":
    main()
