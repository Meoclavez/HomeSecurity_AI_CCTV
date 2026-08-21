"""Push notification service with SQLite persistent device token storage and dual-mode iOS fallback."""

import json
import logging
from typing import Dict, List, Optional
import httpx
from sqlalchemy import select

from app.config import settings
from app.database import async_session_factory
from app.models.db_models import DeviceTokenModel
from app.models.schemas import SecurityEvent, EventSeverity, EventType, DeviceTokenRegistration

logger = logging.getLogger("NotificationService")


class NotificationService:
    async def register_device(self, reg: DeviceTokenRegistration):
        """Register client device token into SQLite persistence."""
        async with async_session_factory() as session:
            stmt = select(DeviceTokenModel).where(DeviceTokenModel.device_token == reg.device_token)
            res = await session.execute(stmt)
            existing = res.scalar_one_or_none()

            if existing:
                existing.platform = reg.platform
                existing.device_name = reg.device_name
                existing.app_version = reg.app_version
            else:
                db_device = DeviceTokenModel(
                    device_token=reg.device_token,
                    platform=reg.platform,
                    device_name=reg.device_name,
                    app_version=reg.app_version
                )
                session.add(db_device)

            await session.commit()
            logger.info(f"Persisted {reg.platform} device token: {reg.device_token[:10]}... ({reg.device_name})")

    async def get_all_registered_devices(self) -> List[DeviceTokenModel]:
        """Fetch all registered devices from database."""
        async with async_session_factory() as session:
            stmt = select(DeviceTokenModel)
            res = await session.execute(stmt)
            return list(res.scalars().all())

    def build_apns_payload(self, event: SecurityEvent, use_critical_entitlement: bool = False) -> dict:
        """
        Construct Apple APNs payload.
        If use_critical_entitlement is False, uses 'time-sensitive' with custom siren audio & relevance-score 1.0,
        which functions immediately on all iOS 15+ devices without waiting for Apple form approvals.
        """
        is_critical = event.severity == EventSeverity.CRITICAL

        title = f"🚨 EMERGENCY: {event.event_type.value.replace('_', ' ').title()}" if is_critical else f"Alert: {event.event_type.value.replace('_', ' ').title()}"
        body = f"{event.location} ({event.camera_name}) — Urgent Attention Required"

        aps_dict = {
            "alert": {
                "title": title,
                "body": body,
            },
            "category": "CATEGORY_EMERGENCY_ALARM" if is_critical else "STANDARD_ALERT",
            "mutable-content": 1,
            "relevance-score": 1.0,  # Pushes notification to the top of iOS Notification Center
        }

        if is_critical and use_critical_entitlement:
            aps_dict["sound"] = {
                "critical": 1,
                "name": "emergency_siren.aiff",
                "volume": 1.0,
            }
            aps_dict["interruption-level"] = "critical"
        else:
            # High-priority Time-Sensitive fallback (No Apple review needed, breaks Focus mode)
            aps_dict["sound"] = "emergency_siren.aiff"
            aps_dict["interruption-level"] = "time-sensitive"

        return {
            "aps": aps_dict,
            "event_id": event.id,
            "event_type": event.event_type.value,
            "camera_id": event.camera_id,
            "camera_name": event.camera_name,
            "location": event.location,
            "is_emergency": "true" if is_critical else "false",
            "clip_url": event.clip_url or "",
            "snapshot_url": event.snapshot_url or "",
            "timestamp": event.timestamp.isoformat(),
        }

    def build_fcm_payload(self, event: SecurityEvent, token: str) -> dict:
        """Construct Android FCM High-Priority payload triggering USAGE_ALARM channel."""
        is_critical = event.severity == EventSeverity.CRITICAL

        title = f"🚨 EMERGENCY: {event.event_type.value.replace('_', ' ').title()}" if is_critical else f"Alert: {event.event_type.value.replace('_', ' ').title()}"
        body = f"{event.location} ({event.camera_name}) — Urgent Attention Required"

        return {
            "to": token,
            "priority": "high",
            "android": {
                "priority": "high",
                "notification": {
                    "channel_id": "cctv_emergency_channel" if is_critical else "cctv_standard_channel",
                    "sound": "emergency_siren",
                    "default_vibrate_timings": True,
                    "notification_priority": "PRIORITY_MAX" if is_critical else "PRIORITY_HIGH",
                }
            },
            "data": {
                "type": "CRITICAL_ALERT" if is_critical else "STANDARD_ALERT",
                "is_emergency": "true" if is_critical else "false",
                "title": title,
                "body": body,
                "event_id": event.id,
                "event_type": event.event_type.value,
                "camera_id": event.camera_id,
                "camera_name": event.camera_name,
                "location": event.location,
                "clip_url": event.clip_url or "",
                "snapshot_url": event.snapshot_url or "",
                "timestamp": event.timestamp.isoformat(),
            }
        }

    async def dispatch_event_notification(self, event: SecurityEvent):
        """Dispatch notifications to all registered Android and iOS devices."""
        devices = await self.get_all_registered_devices()
        logger.info(f"Dispatching notification for event {event.id} to {len(devices)} device(s)")

        apns_payload = self.build_apns_payload(event, use_critical_entitlement=bool(settings.APNS_KEY_ID))

        for device in devices:
            try:
                if device.platform.lower() == "ios":
                    await self._send_apns_push(device.device_token, apns_payload)
                elif device.platform.lower() == "android":
                    fcm_payload = self.build_fcm_payload(event, device.device_token)
                    await self._send_fcm_push(fcm_payload)
            except Exception as e:
                logger.error(f"Failed to dispatch to {device.platform} ({device.device_token[:10]}...): {e}")

    async def _send_fcm_push(self, payload: dict):
        if not settings.FCM_SERVER_KEY:
            logger.debug(f"[MOCK FCM PUSH] Payload: {json.dumps(payload, indent=2)}")
            return

        headers = {
            "Authorization": f"key={settings.FCM_SERVER_KEY}",
            "Content-Type": "application/json"
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post("https://fcm.googleapis.com/fcm/send", json=payload, headers=headers, timeout=5.0)
            logger.info(f"FCM Push Response: {resp.status_code}")

    async def _send_apns_push(self, token: str, payload: dict):
        if not settings.APNS_KEY_ID:
            logger.debug(f"[MOCK APNS PUSH] Token: {token} Payload: {json.dumps(payload, indent=2)}")
            return
        logger.info(f"Sending APNs Push to {token[:10]}...")
        headers = {
            "apns-topic": settings.APNS_BUNDLE_ID,
            "apns-push-type": "alert",
            "apns-priority": "10",
        }
        async with httpx.AsyncClient(http2=True) as client:
            try:
                # Assuming JWT token generation would happen here
                resp = await client.post(f"https://api.push.apple.com/3/device/{token}", json=payload, headers=headers)
                if resp.status_code == 410:
                    logger.warning(f"Device token {token} is no longer active.")
                    # TODO: Implement token cleanup
            except Exception as e:
                logger.error(f"APNs Push failed: {e}")


notification_service = NotificationService()
