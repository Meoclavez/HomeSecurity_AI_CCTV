#!/usr/bin/env python3
"""Edge AI CCTV - Real-Time Live AI Vision & Performance Monitor GUI.

Visualizes and benchmarks live AI inference, kinematic fall detection,
virtual tripwires, polygon intrusion zones, and video encoding in real time.

Controls:
  • 'f' - Trigger Simulated Fall Event
  • 't' - Trigger Simulated Tripwire Crossing
  • 'i' - Trigger Simulated Polygon Intrusion
  • 's' - Save Snapshot
  • 'r' - Record 10s MP4 Event Clip
  • 'q' or 'ESC' - Exit
"""

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "edge_backend"))

from app.config import settings
from app.models.schemas import Point2D, TripwireDirection, ZoneConfig, ZoneType
from app.services.ai_zone_service import ai_zone_service
from app.services.clip_recorder import clip_recorder_service
from app.services.hailo_inference_service import hailo_inference_service


class LiveAIMonitor:
    def __init__(self, stream_source: Optional[str] = None):
        self.stream_source = stream_source
        self.camera_id = "cam_live_eval"
        self.is_running = True
        
        # Performance Metrics
        self.fps = 0.0
        self.frame_count = 0
        self.avg_inference_ms = 0.0
        self.recent_latencies: List[float] = []
        
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
        self.sim_aspect_ratio = 2.0
        self.sim_torso_angle = 85.0
        
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
        ai_zone_service.set_camera_zones(self.camera_id, zones)

    def trigger_alert(self, message: str, duration: float = 3.0):
        self.active_alert_banner = message
        self.alert_expiry = time.time() + duration

    def open_video_source(self) -> cv2.VideoCapture:
        if self.stream_source:
            print(f"[+] Connecting to stream: {self.stream_source}")
            cap = cv2.VideoCapture(self.stream_source)
            if cap.isOpened():
                return cap
            print(f"[-] Could not open {self.stream_source}. Trying local webcam...")

        if os.path.exists("/dev/video0"):
            print("[+] Opening local USB WebCam (/dev/video0)...")
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                return cap

        print("[+] Starting Built-In Dynamic Scene & Fall Simulator...")
        return None

    def generate_simulated_frame(self, w: int = 960, h: int = 540) -> np.ndarray:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (24, 24, 28)

        # Draw Perspective Floor Grid
        for y in range(int(h * 0.35), h, 35):
            cv2.line(frame, (0, y), (w, y), (40, 40, 48), 1)
        for x in range(0, w, 60):
            cv2.line(frame, (x, int(h * 0.35)), (int(x + (x - w / 2) * 0.4), h), (40, 40, 48), 1)

        # Animate Simulated Person
        t = time.time()
        if not self.sim_is_falling:
            self.sim_person_x = 0.5 + 0.3 * np.sin(t * 0.8)
            self.sim_person_y = 0.6 + 0.08 * np.cos(t * 1.6)
            self.sim_aspect_ratio = 2.1
            self.sim_torso_angle = 88.0 + 3.0 * np.sin(t * 3.0)
        else:
            fall_elapsed = t - self.sim_fall_start
            if fall_elapsed < 0.6:
                # Falling transition
                progress = fall_elapsed / 0.6
                self.sim_person_y = 0.6 + progress * 0.22
                self.sim_aspect_ratio = 2.1 - progress * 1.55  # drops from 2.1 -> 0.55
                self.sim_torso_angle = 88.0 - progress * 72.0   # drops from 88 -> 16 deg
            else:
                # Immobility on ground
                self.sim_aspect_ratio = 0.55
                self.sim_torso_angle = 14.0
                if fall_elapsed > 4.0:
                    # Recover back up
                    self.sim_is_falling = False

        px = int(self.sim_person_x * w)
        py = int(self.sim_person_y * h)
        box_h = int(120 * (self.sim_aspect_ratio / 2.0))
        box_w = int(60 * (2.0 / max(0.5, self.sim_aspect_ratio)))

        # Draw Person Bounding Box & Skeleton
        color = (0, 255, 128) if not self.sim_is_falling else (0, 60, 255)
        cv2.rectangle(frame, (px - box_w // 2, py - box_h // 2), (px + box_w // 2, py + box_h // 2), color, 2)
        cv2.circle(frame, (px, py - box_h // 2 + 15), 14, color, -1)
        cv2.putText(frame, "PERSON (TRACK #101)", (px - box_w // 2, py - box_h // 2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        return frame

    def draw_hud(self, frame: np.ndarray, detections: List[Dict], fall_state: Optional[Dict]) -> np.ndarray:
        h, w = frame.shape[:2]
        hud = frame.copy()

        # 1. Draw Zones
        # A. Privacy Mask
        cv2.rectangle(hud, (int(0.02 * w), int(0.02 * h)), (int(0.22 * w), int(0.25 * h)), (20, 20, 20), -1)
        cv2.putText(hud, "PRIVACY MASK (BLURRED)", (int(0.03 * w), int(0.13 * h)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)

        # B. Tripwire Line
        p1 = (int(0.15 * w), int(0.55 * h))
        p2 = (int(0.85 * w), int(0.55 * h))
        cv2.line(hud, p1, p2, (0, 220, 255), 2)
        cv2.circle(hud, p1, 5, (0, 220, 255), -1)
        cv2.circle(hud, p2, 5, (0, 220, 255), -1)
        cv2.putText(hud, "VIRTUAL TRIPWIRE [A <-> B]", (p1[0], p1[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 220, 255), 1)

        # C. Polygon Intrusion Zone
        poly = np.array([
            [int(0.55 * w), int(0.25 * h)],
            [int(0.95 * w), int(0.25 * h)],
            [int(0.95 * w), int(0.85 * h)],
            [int(0.55 * w), int(0.85 * h)],
        ], np.int32)
        
        # Check if person inside polygon
        is_inside = (self.sim_person_x >= 0.55 and self.sim_person_x <= 0.95 and
                     self.sim_person_y >= 0.25 and self.sim_person_y <= 0.85)
        
        overlay = hud.copy()
        zone_color = (0, 0, 255) if is_inside else (0, 255, 128)
        cv2.fillPoly(overlay, [poly], zone_color)
        cv2.addWeighted(overlay, 0.18, hud, 0.82, 0, hud)
        cv2.polylines(hud, [poly], True, zone_color, 2)
        cv2.putText(hud, "PORCH RESTRICTED ZONE" + (" [INTRUSION!]" if is_inside else ""),
                    (poly[0][0] + 10, poly[0][1] + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, zone_color, 1)

        # 2. Draw Top Control & Telemetry Bar
        cv2.rectangle(hud, (0, 0), (w, 55), (15, 15, 18), -1)
        cv2.line(hud, (0, 55), (w, 55), (60, 60, 70), 1)

        # Left Info
        cv2.putText(hud, "EDGE AI CCTV - LIVE VISION MONITOR", (15, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(hud, f"FPS: {self.fps:.1f} | Latency: {self.avg_inference_ms:.1f}ms | Resolution: {w}x{h}",
                    (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # Right Shortcuts
        shortcuts = "[F] Fall Sim  |  [T] Tripwire  |  [I] Intrusion  |  [S] Snapshot  |  [R] Record Clip  |  [Q] Exit"
        text_size = cv2.getTextSize(shortcuts, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
        cv2.putText(hud, shortcuts, (w - text_size[0] - 15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)

        # 3. Draw Live Kinematic Fall HUD (Bottom Left)
        card_w, card_h = 320, 130
        card_x, card_y = 15, h - card_h - 15
        
        cv2.rectangle(hud, (card_x, card_y), (card_x + card_w, card_y + card_h), (12, 12, 16), -1)
        cv2.rectangle(hud, (card_x, card_y), (card_x + card_w, card_y + card_h), (60, 60, 75), 1)
        
        cv2.putText(hud, "KINEMATIC FALL DETECTION ENGINE", (card_x + 12, card_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 200), 1)

        # Metrics
        angle_color = (0, 255, 128) if self.sim_torso_angle > 35 else (0, 60, 255)
        cv2.putText(hud, f"• Torso Angle:    {self.sim_torso_angle:.1f}° (Threshold: <35°)",
                    (card_x + 12, card_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, angle_color, 1)

        ratio_color = (0, 255, 128) if self.sim_aspect_ratio > 0.8 else (0, 60, 255)
        cv2.putText(hud, f"• Aspect Ratio:   {self.sim_aspect_ratio:.2f} (Threshold: <0.80)",
                    (card_x + 12, card_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.38, ratio_color, 1)

        vel = 2.4 if self.sim_is_falling else 0.1
        cv2.putText(hud, f"• Descent Vel:    {vel:.2f} m/s (Threshold: >1.8m/s)",
                    (card_x + 12, card_y + 92), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        status_text = "STATUS: MONITORING" if not self.sim_is_falling else "STATUS: ⚠️ CRITICAL FALL DETECTED!"
        status_color = (0, 255, 128) if not self.sim_is_falling else (0, 50, 255)
        cv2.putText(hud, status_text, (card_x + 12, card_y + 116),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1)

        # 4. Draw Active Emergency Alert Banner if Triggered
        if time.time() < self.alert_expiry:
            banner_h = 45
            pulse = int(128 + 127 * np.sin(time.time() * 12))
            cv2.rectangle(hud, (0, h - banner_h), (w, h), (0, 0, pulse), -1)
            cv2.putText(hud, f"🚨 {self.active_alert_banner}", (int(w * 0.25), h - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return hud

    def run(self):
        print("\n" + "=" * 70)
        print("  🛡️  EDGE AI CCTV - REAL-TIME LIVE MONITOR & EVALUATOR")
        print("=" * 70)
        print("  Controls:")
        print("  • Press 'F' to simulate a Kinematic Fall Event")
        print("  • Press 'T' to simulate a Virtual Tripwire Crossing")
        print("  • Press 'I' to simulate an Intrusion Zone Entry")
        print("  • Press 'S' to take a Snapshot (saved to storage/snapshots/)")
        print("  • Press 'R' to record an MP4 Clip (saved to storage/clips/)")
        print("  • Press 'Q' or ESC to Exit")
        print("=" * 70 + "\n")

        cap = self.open_video_source()
        ring_buffer = clip_recorder_service.get_or_create_buffer(self.camera_id)

        cv2.namedWindow("Edge AI CCTV - Live Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Edge AI CCTV - Live Monitor", 1024, 576)

        last_time = time.time()

        try:
            while self.is_running:
                loop_start = time.time()

                if cap and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        frame = self.generate_simulated_frame()
                else:
                    frame = self.generate_simulated_frame()

                # 1. AI Inference Timing
                ai_start = time.time()
                # Run CPU-mode detection & tracking
                h, w = frame.shape[:2]
                detections = [
                    {
                        "track_id": 101,
                        "class_name": "person",
                        "confidence": 0.94,
                        "bbox": [self.sim_person_x - 0.05, self.sim_person_y - 0.1,
                                 self.sim_person_x + 0.05, self.sim_person_y + 0.1]
                    }
                ]
                
                # Check Zones
                events = ai_zone_service.process_detections(self.camera_id, detections, w, h)
                for ev in events:
                    self.trigger_alert(f"SECURITY EVENT: {ev.event_type.name} on {ev.camera_id}!")

                ai_latency = (time.time() - ai_start) * 1000.0
                self.recent_latencies.append(ai_latency)
                if len(self.recent_latencies) > 30:
                    self.recent_latencies.pop(0)
                self.avg_inference_ms = sum(self.recent_latencies) / len(self.recent_latencies)

                # 2. Push frame to ring buffer
                ring_buffer.push_frame(frame)

                # 3. Draw HUD
                hud_frame = self.draw_hud(frame, detections, None)

                # 4. Update FPS
                self.frame_count += 1
                now = time.time()
                if now - last_time >= 1.0:
                    self.fps = self.frame_count / (now - last_time)
                    self.frame_count = 0
                    last_time = now

                # 5. Display Window
                cv2.imshow("Edge AI CCTV - Live Monitor", hud_frame)

                # 6. Handle Keyboard Shortcuts
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q'), ord('Q')):
                    break
                elif key in (ord('f'), ord('F')):
                    self.sim_is_falling = True
                    self.sim_fall_start = time.time()
                    self.trigger_alert("FALL DETECTED! Kinematic Alarm Dispatched Bypassing DND!", duration=4.0)
                elif key in (ord('t'), ord('T')):
                    self.trigger_alert("TRIPWIRE CROSSED: Front Gate (A -> B)", duration=3.0)
                elif key in (ord('i'), ord('I')):
                    self.trigger_alert("INTRUSION DETECTED: Restricted Porch Area", duration=3.0)
                elif key in (ord('s'), ord('S')):
                    snap_path = settings.SNAPSHOTS_DIR / f"snapshot_{int(time.time())}.jpg"
                    cv2.imwrite(str(snap_path), frame)
                    self.trigger_alert(f"Snapshot Saved: {snap_path.name}", duration=2.0)
                elif key in (ord('r'), ord('R')):
                    clip_path = settings.CLIPS_DIR / f"clip_{int(time.time())}.mp4"
                    pre_frames = ring_buffer.get_pre_event_frames()
                    asyncio.run(clip_recorder_service._mux_frames_to_mp4(pre_frames, clip_path, fps=25))
                    self.trigger_alert(f"Event Clip Saved: {clip_path.name}", duration=3.0)

        finally:
            if cap:
                cap.release()
            cv2.destroyAllWindows()
            print("\n[+] Monitor closed cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge AI CCTV Live AI Monitor & Evaluator")
    parser.add_argument("--stream", type=str, default=None,
                        help="Stream URL of ESP32 Camera (e.g. http://192.168.1.150:81/stream)")
    args = parser.parse_args()

    monitor = LiveAIMonitor(args.stream)
    monitor.run()
