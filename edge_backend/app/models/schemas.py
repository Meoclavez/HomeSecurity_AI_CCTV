"""Pydantic schemas for Edge CCTV AI data validation, camera feeds, events, zones, DVR, and telemetry."""

from enum import Enum
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


# ---------------- Enums ----------------

class CameraStatus(str, Enum):
    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    DEGRADED = "DEGRADED"


class EventType(str, Enum):
    FALL_DETECTED = "FALL_DETECTED"
    INTRUSION_DETECTED = "INTRUSION_DETECTED"
    WEAPON_DETECTED = "WEAPON_DETECTED"
    PERIMETER_BREACH = "PERIMETER_BREACH"
    DOOR_LEFT_OPEN = "DOOR_LEFT_OPEN"
    PACKAGE_INTERACTION = "PACKAGE_INTERACTION"
    TAMPERING_DETECTED = "TAMPERING_DETECTED"


class EventSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    WARNING = "WARNING"
    INFO = "INFO"


class ZoneType(str, Enum):
    PRIVACY_MASK = "PRIVACY_MASK"
    TRIPWIRE = "TRIPWIRE"
    INTRUSION = "INTRUSION"
    DOOR = "DOOR"
    PACKAGE = "PACKAGE"


class MaskMode(str, Enum):
    BLACKOUT = "BLACKOUT"
    BLUR = "BLUR"
    MOSAIC = "MOSAIC"
    COLOR = "COLOR"


class TripwireDirection(str, Enum):
    A_TO_B = "A_TO_B"
    B_TO_A = "B_TO_A"
    BIDIRECTIONAL = "BIDIRECTIONAL"


# ---------------- Geometry & Zones ----------------

class Point2D(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0, description="Normalized X coordinate (0.0 to 1.0)")
    y: float = Field(..., ge=0.0, le=1.0, description="Normalized Y coordinate (0.0 to 1.0)")


class ZoneConfig(BaseModel):
    id: str
    camera_id: str
    name: str
    zone_type: ZoneType
    enabled: bool = True
    polygon_points: Optional[List[Point2D]] = Field(default_factory=list)
    line_start: Optional[Point2D] = None
    line_end: Optional[Point2D] = None
    direction: TripwireDirection = TripwireDirection.BIDIRECTIONAL
    mask_mode: MaskMode = MaskMode.BLACKOUT
    mask_color_bgr: tuple[int, int, int] = (0, 0, 0)
    blur_kernel_size: int = 51
    mosaic_scale: int = 16
    dwell_time_seconds: float = 0.0
    allowed_classes: List[str] = Field(default_factory=lambda: ["person", "car", "package"])


# ---------------- Vision & Kinematics ----------------

class BoundingBox(BaseModel):
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    label: str


class Keypoint(BaseModel):
    id: int
    name: str
    x: float
    y: float
    confidence: float


class KinematicTelemetry(BaseModel):
    hip_descent_velocity: float
    aspect_ratio_initial: float
    aspect_ratio_final: float
    transition_duration_ms: float
    immobility_duration_sec: float
    floor_proximity_score: float
    torso_angle_deg: Optional[float] = None


# Alias for backward compatibility
KinematicMetrics = KinematicTelemetry


# ---------------- Security Events ----------------

class SecurityEventBase(BaseModel):
    camera_id: str
    event_type: EventType
    severity: EventSeverity
    confidence: float
    bounding_box: Optional[BoundingBox] = None
    keypoints: Optional[List[Keypoint]] = None
    kinematics: Optional[KinematicTelemetry] = None
    metadata: Optional[Dict[str, Any]] = None


class SecurityEventCreate(SecurityEventBase):
    pass


class SecurityEvent(SecurityEventBase):
    id: str
    camera_name: str
    location: str
    timestamp: datetime
    clip_url: Optional[str] = None
    snapshot_url: Optional[str] = None
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class EventListResponse(BaseModel):
    events: List[SecurityEvent]
    total: int


# Alias for backward compatibility
SecurityEventListResponse = EventListResponse


# ---------------- Cameras & WebRTC ----------------

class CameraFeed(BaseModel):
    id: str
    name: str
    location: str
    webrtc_url: str
    status: str
    fps: int
    resolution: str
    is_ai_enabled: bool
    ai_models: List[str]
    dvr_enabled: bool = True
    dvr_retention_days: int = 7
    dvr_quota_gb: float = 100.0
    last_seen: Optional[datetime] = None

    class Config:
        from_attributes = True


class CameraListResponse(BaseModel):
    cameras: List[CameraFeed]
    total: Optional[int] = None


class WebRtcOffer(BaseModel):
    camera_id: str
    sdp: str
    type: str = "offer"


class WebRtcAnswer(BaseModel):
    camera_id: str
    sdp: str
    type: str = "answer"


# ---------------- 24-Hour Timeline & DVR ----------------

class TimelineSegment(BaseModel):
    id: str
    camera_id: str
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    file_size_bytes: int
    stream_url: str


class TimelineGap(BaseModel):
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    reason: str = "OFFLINE_OR_STREAM_DROP"


class TimelineEventMarker(BaseModel):
    id: str
    event_type: str
    severity: str
    confidence: float
    timestamp: datetime
    snapshot_url: Optional[str] = None
    clip_url: Optional[str] = None
    bounding_box: Optional[Dict[str, Any]] = None


class CameraTimelineResponse(BaseModel):
    camera_id: str
    camera_name: str
    date: str
    total_recorded_seconds: float
    total_segments: int
    hls_master_url: str
    segments: List[TimelineSegment]
    events: List[TimelineEventMarker]
    gaps: List[TimelineGap]


# ---------------- Custom Incident Export & Archives ----------------

class DVRExportRequest(BaseModel):
    start_time: datetime = Field(..., description="ISO 8601 start timestamp")
    end_time: datetime = Field(..., description="ISO 8601 end timestamp")
    title: str = Field(..., min_length=1, max_length=256, description="Title for the archived incident")
    description: Optional[str] = Field(None, max_length=512)


class IncidentArchiveResponse(BaseModel):
    id: str
    camera_id: str
    camera_name: str
    title: str
    description: Optional[str] = None
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    file_size_bytes: int
    status: str
    download_url: Optional[str] = None
    created_at: datetime


class IncidentArchiveListResponse(BaseModel):
    archives: List[IncidentArchiveResponse]
    total: int


# ---------------- Storage Health & Devices ----------------

class DiskSMARTInfo(BaseModel):
    device: str
    model: str
    serial_number: Optional[str] = None
    temperature_celsius: Optional[int] = None
    health_status: str = "PASSED"
    reallocated_sectors: Optional[int] = 0
    wear_level_percent: Optional[int] = None
    power_on_hours: Optional[int] = None
    is_ssd: bool = True


class CameraStorageQuota(BaseModel):
    camera_id: str
    camera_name: str
    used_bytes: int
    used_gb: float
    quota_gb: float
    segment_count: int
    oldest_segment: Optional[datetime] = None
    newest_segment: Optional[datetime] = None


class StorageHealthResponse(BaseModel):
    storage_root: str
    is_external_mount: bool
    total_gb: float
    used_gb: float
    free_gb: float
    used_percent: float
    smart_status: List[DiskSMARTInfo]
    camera_quotas: List[CameraStorageQuota]
    archives_used_gb: float


# ---------------- Device & Mute Schemas ----------------

class DeviceRegistration(BaseModel):
    device_token: str
    platform: str
    device_name: Optional[str] = None
    app_version: Optional[str] = None


# Alias for backward compatibility
DeviceTokenRegistration = DeviceRegistration


class MuteCameraRequest(BaseModel):
    duration_minutes: int = 5
