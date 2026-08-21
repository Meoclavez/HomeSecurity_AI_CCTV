"""AI Zone Service: Privacy Masking, Virtual Tripwires, Intrusion Zones, 
Hailo-8 Multi-Stream Scheduling, and Temporal State Machines.
"""

import time
import math
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import cv2

from app.config import settings
from app.models.schemas import (
    EventType,
    EventSeverity,
    SecurityEventCreate,
    BoundingBox,
    ZoneConfig,
    ZoneType,
    MaskMode,
    TripwireDirection,
)

logger = logging.getLogger("AIZoneService")


# ---------------- 1. Spatial Geometry & Math (PIP & Tripwire) ----------------

class PolygonGeometry:
    """High-performance 2D geometric operations for point-in-polygon and line intersection."""

    @staticmethod
    def point_in_polygon_raycasting(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """Ray-Casting Algorithm (Even-Odd rule / Jordan Curve Theorem)."""
        n = len(polygon)
        if n < 3:
            return False

        px, py = point
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if py > min(p1y, p2y):
                if py <= max(p1y, p2y):
                    if px <= max(p1x, p2x):
                        if p1y != p2y:
                            x_inters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or px <= x_inters:
                                inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    @staticmethod
    def _is_left(p0: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """2D Cross product determining if point P2 is left (>0), on (=0), or right (<0) of line P0->P1."""
        return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])

    @staticmethod
    def check_line_crossing(
        p_prev: Tuple[float, float],
        p_curr: Tuple[float, float],
        w_start: Tuple[float, float],
        w_end: Tuple[float, float]
    ) -> Optional[TripwireDirection]:
        """Evaluates if trajectory segment (p_prev -> p_curr) crossed tripwire (w_start -> w_end)."""
        d1 = PolygonGeometry._is_left(w_start, w_end, p_prev)
        d2 = PolygonGeometry._is_left(w_start, w_end, p_curr)
        c1 = PolygonGeometry._is_left(p_prev, p_curr, w_start)
        c2 = PolygonGeometry._is_left(p_prev, p_curr, w_end)

        if (d1 * d2 < 0.0) and (c1 * c2 < 0.0):
            if d1 > 0 and d2 < 0:
                return TripwireDirection.A_TO_B
            elif d1 < 0 and d2 > 0:
                return TripwireDirection.B_TO_A

        return None

    @staticmethod
    def get_bbox_footprint(bbox: BoundingBox) -> Tuple[float, float]:
        """Calculates ground plane contact footprint (bottom-center point of bounding box)."""
        footprint_x = (bbox.x_min + bbox.x_max) / 2.0
        footprint_y = bbox.y_max
        return (footprint_x, footprint_y)


# ---------------- 2. Privacy Masking Engine ----------------

class PrivacyMaskEngine:
    """Applies privacy masks to video frames before storage, streaming, and ring-buffer ingestion."""

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.masks: List[ZoneConfig] = []

    def update_masks(self, mask_configs: List[ZoneConfig]):
        self.masks = [m for m in mask_configs if m.zone_type == ZoneType.PRIVACY_MASK and m.enabled]

    def apply_privacy_masks(self, frame: np.ndarray) -> np.ndarray:
        """Obfuscates privacy regions on the edge in-place or returns masked copy."""
        if not self.masks or frame is None:
            return frame

        h, w = frame.shape[:2]
        output_frame = frame.copy()

        for mask_cfg in self.masks:
            if not mask_cfg.polygon_points or len(mask_cfg.polygon_points) < 3:
                continue

            pts = np.array(
                [[int(p.x * w), int(p.y * h)] for p in mask_cfg.polygon_points],
                dtype=np.int32
            )

            if mask_cfg.mask_mode == MaskMode.BLACKOUT:
                cv2.fillPoly(output_frame, [pts], (0, 0, 0))

            elif mask_cfg.mask_mode == MaskMode.COLOR:
                cv2.fillPoly(output_frame, [pts], mask_cfg.mask_color_bgr)

            elif mask_cfg.mask_mode == MaskMode.BLUR:
                poly_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(poly_mask, [pts], 255)
                k = mask_cfg.blur_kernel_size | 1
                blurred = cv2.GaussianBlur(output_frame, (k, k), 0)
                output_frame[poly_mask == 255] = blurred[poly_mask == 255]

            elif mask_cfg.mask_mode == MaskMode.MOSAIC:
                rx, ry, rw, rh = cv2.boundingRect(pts)
                if rw > 0 and rh > 0:
                    poly_mask = np.zeros((rh, rw), dtype=np.uint8)
                    pts_shifted = pts - np.array([rx, ry])
                    cv2.fillPoly(poly_mask, [pts_shifted], 255)

                    roi = output_frame[ry:ry + rh, rx:rx + rw]
                    scale = max(2, mask_cfg.mosaic_scale)
                    small_w, small_h = max(1, rw // scale), max(1, rh // scale)
                    small_roi = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
                    pixelated = cv2.resize(small_roi, (rw, rh), interpolation=cv2.INTER_NEAREST)

                    roi[poly_mask == 255] = pixelated[poly_mask == 255]
                    output_frame[ry:ry + rh, rx:rx + rw] = roi

        return output_frame


# ---------------- 3. Spatial Zone & Tripwire Tracker ----------------

@dataclass
class TrackSpatialState:
    track_id: int
    label: str
    last_position: Tuple[float, float]
    entry_timestamps: Dict[str, float] = field(default_factory=dict)
    last_tripwire_alerts: Dict[str, float] = field(default_factory=dict)
    last_seen: float = field(default_factory=time.time)


class ZoneAnalyticsTracker:
    """Maintains continuous spatial states for intrusion zones and directional tripwires."""

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.zones: Dict[str, ZoneConfig] = {}
        self.tracks: Dict[int, TrackSpatialState] = {}
        self.last_cleanup = time.time()

    def update_zones(self, zones: List[ZoneConfig]):
        self.zones = {z.id: z for z in zones if z.enabled}

    def process_detections(
        self,
        detections: List[Tuple[int, BoundingBox]]
    ) -> List[SecurityEventCreate]:
        events: List[SecurityEventCreate] = []
        now = time.time()

        if now - self.last_cleanup > 5.0:
            stale_ids = [tid for tid, t in self.tracks.items() if (now - t.last_seen) > 10.0]
            for tid in stale_ids:
                del self.tracks[tid]
            self.last_cleanup = now

        for track_id, bbox in detections:
            curr_pos = PolygonGeometry.get_bbox_footprint(bbox)

            if track_id not in self.tracks:
                self.tracks[track_id] = TrackSpatialState(
                    track_id=track_id,
                    label=bbox.label,
                    last_position=curr_pos,
                    last_seen=now
                )
                prev_pos = curr_pos
            else:
                prev_pos = self.tracks[track_id].last_position
                self.tracks[track_id].last_position = curr_pos
                self.tracks[track_id].last_seen = now

            track_state = self.tracks[track_id]

            # 1. Evaluate Tripwires
            for zone_id, zone in self.zones.items():
                if zone.zone_type != ZoneType.TRIPWIRE or not zone.line_start or not zone.line_end:
                    continue

                w_start = (zone.line_start.x, zone.line_start.y)
                w_end = (zone.line_end.x, zone.line_end.y)
                crossing = PolygonGeometry.check_line_crossing(prev_pos, curr_pos, w_start, w_end)

                if crossing:
                    is_valid_dir = (
                        zone.direction == TripwireDirection.BIDIRECTIONAL or
                        zone.direction == crossing
                    )
                    last_alert = track_state.last_tripwire_alerts.get(zone_id, 0.0)
                    if is_valid_dir and (now - last_alert > 3.0):
                        track_state.last_tripwire_alerts[zone_id] = now
                        events.append(
                            SecurityEventCreate(
                                camera_id=self.camera_id,
                                event_type=EventType.INTRUSION_DETECTED,
                                severity=EventSeverity.CRITICAL,
                                confidence=bbox.confidence,
                                bounding_box=bbox,
                                metadata={
                                    "zone_id": zone.id,
                                    "zone_name": zone.name,
                                    "analytics_type": "TRIPWIRE_LINE_CROSSING",
                                    "direction": crossing.value,
                                    "track_id": track_id
                                }
                            )
                        )

            # 2. Evaluate Polygon Intrusion & Loitering Zones
            for zone_id, zone in self.zones.items():
                if zone.zone_type != ZoneType.INTRUSION or not zone.polygon_points:
                    continue

                poly = [(p.x, p.y) for p in zone.polygon_points]
                is_inside = PolygonGeometry.point_in_polygon_raycasting(curr_pos, poly)

                if is_inside:
                    if zone_id not in track_state.entry_timestamps:
                        track_state.entry_timestamps[zone_id] = now

                    dwell_time = now - track_state.entry_timestamps[zone_id]
                    if dwell_time >= zone.dwell_time_seconds:
                        if dwell_time - zone.dwell_time_seconds < 1.0:
                            events.append(
                                SecurityEventCreate(
                                    camera_id=self.camera_id,
                                    event_type=EventType.INTRUSION_DETECTED,
                                    severity=EventSeverity.CRITICAL if zone.dwell_time_seconds == 0 else EventSeverity.WARNING,
                                    confidence=bbox.confidence,
                                    bounding_box=bbox,
                                    metadata={
                                        "zone_id": zone.id,
                                        "zone_name": zone.name,
                                        "analytics_type": "POLYGON_INTRUSION",
                                        "dwell_time_sec": round(dwell_time, 1),
                                        "track_id": track_id
                                    }
                                )
                            )
                else:
                    track_state.entry_timestamps.pop(zone_id, None)

        return events


# ---------------- 4. Temporal Security State Machines ----------------

class DoorStateMachine:
    """Monitors door open/closed status with debounce hysteresis and timeout alert."""

    def __init__(self, camera_id: str, timeout_seconds: float = settings.DOOR_OPEN_ALERT_TIMEOUT_SEC):
        self.camera_id = camera_id
        self.timeout_seconds = timeout_seconds
        self.is_open = False
        self.opened_at: Optional[float] = None
        self.alert_dispatched = False
        self.consecutive_open_frames = 0
        self.consecutive_closed_frames = 0
        self.hysteresis_threshold = 5

    def update(self, is_door_open: bool, bbox: Optional[BoundingBox] = None) -> Optional[SecurityEventCreate]:
        now = time.time()

        if is_door_open:
            self.consecutive_open_frames += 1
            self.consecutive_closed_frames = 0
        else:
            self.consecutive_closed_frames += 1
            self.consecutive_open_frames = 0

        if self.consecutive_open_frames >= self.hysteresis_threshold and not self.is_open:
            self.is_open = True
            self.opened_at = now
            self.alert_dispatched = False

        elif self.consecutive_closed_frames >= self.hysteresis_threshold and self.is_open:
            self.is_open = False
            self.opened_at = None
            self.alert_dispatched = False

        if self.is_open and self.opened_at:
            duration = now - self.opened_at
            if duration >= self.timeout_seconds and not self.alert_dispatched:
                self.alert_dispatched = True
                return SecurityEventCreate(
                    camera_id=self.camera_id,
                    event_type=EventType.DOOR_LEFT_OPEN,
                    severity=EventSeverity.WARNING,
                    confidence=0.95,
                    bounding_box=bbox,
                    metadata={
                        "duration_seconds": int(duration),
                        "timeout_threshold": int(self.timeout_seconds)
                    }
                )

        return None


class PackageTheftStateMachine:
    """Tracks stationary packages and generates theft alerts if package is displaced."""

    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.anchor_bbox: Optional[BoundingBox] = None
        self.package_first_seen: Optional[float] = None
        self.is_anchored = False
        self.theft_alert_sent = False

    def update(
        self,
        detected_packages: List[BoundingBox],
        detected_persons: List[BoundingBox]
    ) -> Optional[SecurityEventCreate]:
        now = time.time()

        if detected_packages and not self.is_anchored:
            pkg = detected_packages[0]
            if self.package_first_seen is None:
                self.package_first_seen = now
            elif now - self.package_first_seen >= 3.0:
                self.anchor_bbox = pkg
                self.is_anchored = True
                return SecurityEventCreate(
                    camera_id=self.camera_id,
                    event_type=EventType.PACKAGE_INTERACTION,
                    severity=EventSeverity.INFO,
                    confidence=pkg.confidence,
                    bounding_box=pkg,
                    metadata={"status": "PACKAGE_DELIVERED"}
                )

        if self.is_anchored and self.anchor_bbox:
            current_anchor_pkg = None
            for p in detected_packages:
                dx = abs((p.x_min + p.x_max) / 2.0 - (self.anchor_bbox.x_min + self.anchor_bbox.x_max) / 2.0)
                dy = abs((p.y_min + p.y_max) / 2.0 - (self.anchor_bbox.y_min + self.anchor_bbox.y_max) / 2.0)
                if math.sqrt(dx * dx + dy * dy) < 0.15:
                    current_anchor_pkg = p
                    break

            if current_anchor_pkg is None:
                is_person_near = len(detected_persons) > 0
                if is_person_near and not self.theft_alert_sent:
                    self.theft_alert_sent = True
                    self.is_anchored = False
                    return SecurityEventCreate(
                        camera_id=self.camera_id,
                        event_type=EventType.PACKAGE_INTERACTION,
                        severity=EventSeverity.CRITICAL,
                        confidence=0.92,
                        bounding_box=detected_persons[0] if detected_persons else self.anchor_bbox,
                        metadata={
                            "status": "PACKAGE_THEFT_DETECTED",
                            "anchor_coords": {
                                "x": (self.anchor_bbox.x_min + self.anchor_bbox.x_max) / 2.0,
                                "y": (self.anchor_bbox.y_min + self.anchor_bbox.y_max) / 2.0
                            }
                        }
                    )

        return None


# ---------------- 5. Multi-Stream Hailo Dynamic Scheduler ----------------

class StreamPriority(int):
    IDLE = 1
    MOTION = 2
    ALERT_ACTIVE = 3


class MultiStreamHailoScheduler:
    """Dynamic token-bucket credit scheduler balancing 4 to 16 camera sub-streams on Hailo-8."""

    def __init__(self, target_fps_budget: float = 240.0):
        self.target_fps_budget = target_fps_budget
        self.stream_states: Dict[str, int] = {}
        self.last_inference_times: Dict[str, float] = {}

    def register_stream(self, camera_id: str):
        self.stream_states[camera_id] = StreamPriority.IDLE
        self.last_inference_times[camera_id] = 0.0

    def update_stream_priority(self, camera_id: str, priority: int):
        self.stream_states[camera_id] = priority

    def should_infer_frame(self, camera_id: str) -> Tuple[bool, Tuple[int, int]]:
        """Returns (should_run_inference, (target_width, target_height))."""
        now = time.time()
        priority = self.stream_states.get(camera_id, StreamPriority.IDLE)
        last_t = self.last_inference_times.get(camera_id, 0.0)

        if priority == StreamPriority.ALERT_ACTIVE:
            min_interval = 1.0 / 25.0
            resolution = (640, 640)
        elif priority == StreamPriority.MOTION:
            min_interval = 1.0 / 10.0
            resolution = (640, 640)
        else:
            min_interval = 1.0 / 2.0
            resolution = (320, 320)

        if (now - last_t) >= min_interval:
            self.last_inference_times[camera_id] = now
            return True, resolution

        return False, resolution


# ---------------- 6. Master AI Zone Service ----------------

class AIZoneService:
    def __init__(self):
        self.privacy_engines: Dict[str, PrivacyMaskEngine] = {}
        self.zone_trackers: Dict[str, ZoneAnalyticsTracker] = {}
        self.door_machines: Dict[str, DoorStateMachine] = {}
        self.package_machines: Dict[str, PackageTheftStateMachine] = {}
        self.scheduler = MultiStreamHailoScheduler()

    def get_or_create_privacy_engine(self, camera_id: str) -> PrivacyMaskEngine:
        if camera_id not in self.privacy_engines:
            self.privacy_engines[camera_id] = PrivacyMaskEngine(camera_id)
            self.zone_trackers[camera_id] = ZoneAnalyticsTracker(camera_id)
            self.door_machines[camera_id] = DoorStateMachine(camera_id)
            self.package_machines[camera_id] = PackageTheftStateMachine(camera_id)
            self.scheduler.register_stream(camera_id)
        return self.privacy_engines[camera_id]

    def set_camera_zones(self, camera_id: str, zones: List[ZoneConfig]):
        privacy = self.get_or_create_privacy_engine(camera_id)
        privacy.update_masks(zones)
        self.zone_trackers[camera_id].update_zones(zones)

    def mask_frame(self, camera_id: str, frame: np.ndarray) -> np.ndarray:
        privacy = self.get_or_create_privacy_engine(camera_id)
        return privacy.apply_privacy_masks(frame)


ai_zone_service = AIZoneService()
