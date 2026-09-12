"""Feature toggle manager for camera AI features."""

import logging
from typing import Dict, Optional, Any
from app.models.schemas import CameraFeatureConfig

logger = logging.getLogger("FeatureManager")


class FeatureManager:
    """In-memory cache and coordinator for camera AI feature toggles."""

    def __init__(self):
        self._camera_features: Dict[str, CameraFeatureConfig] = {}

    def get_features(self, camera_id: str) -> CameraFeatureConfig:
        if camera_id not in self._camera_features:
            self._camera_features[camera_id] = CameraFeatureConfig()
        return self._camera_features[camera_id]

    def update_features(self, camera_id: str, config: CameraFeatureConfig) -> CameraFeatureConfig:
        self._camera_features[camera_id] = config
        logger.info(f"Updated feature toggles for camera {camera_id}: {config.dict() if hasattr(config, 'dict') else config}")
        return config

    def set_feature(self, camera_id: str, feature_name: str, enabled: bool) -> CameraFeatureConfig:
        cfg = self.get_features(camera_id)
        if hasattr(cfg, feature_name):
            setattr(cfg, feature_name, enabled)
            self._camera_features[camera_id] = cfg
        return cfg


feature_manager = FeatureManager()
