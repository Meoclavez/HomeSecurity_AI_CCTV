import os
import secrets
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Base Application Metadata
    APP_NAME: str = "Edge CCTV AI Surveillance Core"
    VERSION: str = "1.2.0"
    DEBUG: bool = False
    PORT: int = int(os.getenv("PORT", "8000"))
    HOST: str = os.getenv("HOST", "0.0.0.0")

    # Local Network & Edge Server Base URL
    EDGE_BASE_URL: str = os.getenv("EDGE_BASE_URL", "http://192.168.1.100:8000")
    ALLOWED_CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://192.168.1.100:8000",
        "https://edge-cctv.local",
    ]

    # Persistent Storage Paths (SQLite, Video Clips, Snapshots, DVR, Archives)
    STORAGE_DIR: Path = Path(os.getenv("STORAGE_DIR", "./storage")).resolve()
    SNAPSHOTS_DIR: Path = STORAGE_DIR / "snapshots"
    CLIPS_DIR: Path = STORAGE_DIR / "clips"
    DVR_DIR: Path = STORAGE_DIR / "dvr"
    ARCHIVES_DIR: Path = STORAGE_DIR / "archives"
    SQLITE_DB_PATH: Path = STORAGE_DIR / "cctv_core.db"
    DATABASE_PATH: Path = SQLITE_DB_PATH

    # Storage Retention & Purging Policies
    STORAGE_RETENTION_DAYS: int = int(os.getenv("STORAGE_RETENTION_DAYS", "7"))
    STORAGE_MAX_DISK_PERCENT: float = float(os.getenv("STORAGE_MAX_DISK_PERCENT", "85.0"))
    DVR_DEFAULT_RETENTION_DAYS: int = 7
    DVR_DEFAULT_QUOTA_GB: float = 100.0

    # Hardware Passthrough Paths (Intel N100 + Hailo-8 M.2)
    VAAPI_DEVICE: str = os.getenv("VAAPI_DEVICE", "/dev/dri/renderD128")
    HAILO_DEVICE: str = os.getenv("HAILO_DEVICE", "/dev/hailo0")
    HAILO_YOLO_HEF_PATH: str = os.getenv("HAILO_YOLO_HEF_PATH", "./models_hef/yolov8n.hef")
    HAILO_POSE_HEF_PATH: str = os.getenv("HAILO_POSE_HEF_PATH", "./models_hef/yolov8n_pose.hef")

    # Security, JWT & Service Secrets
    JWT_SECRET: str = os.getenv("JWT_SECRET", secrets.token_hex(32))
    JWT_ALGORITHM: str = "HS256"
    STREAM_TOKEN_EXPIRE_SECONDS: int = 86400  # 24 hours
    STREAM_TOKEN_EXPIRY_SECONDS: int = 86400
    CLIP_TOKEN_EXPIRY_SECONDS: int = 86400
    INTERNAL_SERVICE_KEY: str = os.getenv("INTERNAL_SERVICE_KEY", "edge_ai_vision_internal_secret")

    # Coturn TURN/STUN Relay Credentials
    COTURN_SECRET: str = os.getenv("COTURN_SECRET", "cctv_turn_super_secret_dynamic_key_change_me_in_prod")
    COTURN_REALM: str = os.getenv("COTURN_REALM", "cctv.local")
    COTURN_PUBLIC_IP: str = os.getenv("COTURN_PUBLIC_IP", "192.168.1.100")
    COTURN_PORT: int = int(os.getenv("COTURN_PORT", "3478"))

    # Kinematics & AI Event Thresholds
    CAMERA_ALERT_COOLDOWN_SEC: float = 30.0
    FALL_TRANSITION_MAX_MS: float = 800.0
    FALL_TORSO_HORIZONTAL_ANGLE: float = 35.0
    FALL_ASPECT_RATIO_END: float = 0.8
    FALL_VELOCITY_THRESHOLD_Y: float = 1.8
    FALL_IMMOBILITY_SECONDS: float = 5.0
    FALL_VELOCITY_THRESHOLD: float = 1.8
    FALL_ASPECT_RATIO_THRESHOLD: float = 0.8
    FALL_IMMOBILITY_TIME_SEC: float = 5.0
    FALL_TORSO_ANGLE_THRESHOLD: float = 35.0
    DOOR_OPEN_ALERT_TIMEOUT_SEC: float = 300.0

    # Buffer & Clip Settings
    PRE_EVENT_BUFFER_SECONDS: int = 5
    POST_EVENT_RECORD_SECONDS: int = 10
    RECORDING_FPS: int = 25

    # Push Notification Credentials
    FCM_SERVER_KEY: str = os.getenv("FCM_SERVER_KEY", "")
    APNS_KEY_ID: str = os.getenv("APNS_KEY_ID", "")
    APNS_TEAM_ID: str = os.getenv("APNS_TEAM_ID", "")
    APNS_BUNDLE_ID: str = os.getenv("APNS_BUNDLE_ID", "com.cctv.edgeAiCctv")

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()

# Ensure all directory paths exist
settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
settings.CLIPS_DIR.mkdir(parents=True, exist_ok=True)
settings.DVR_DIR.mkdir(parents=True, exist_ok=True)
settings.ARCHIVES_DIR.mkdir(parents=True, exist_ok=True)
