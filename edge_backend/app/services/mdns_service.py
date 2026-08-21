"""mDNS / Bonjour auto-discovery broadcaster for Edge CCTV AI system."""

import socket
import logging
from typing import Optional
from zeroconf import IPVersion, ServiceInfo
from zeroconf.asyncio import AsyncZeroconf
from app.config import settings

logger = logging.getLogger("mDNSService")


class EdgeMDNSAdvertiser:
    def __init__(self):
        self.aiozc: Optional[AsyncZeroconf] = None
        self.service_info: Optional[ServiceInfo] = None

    def _get_local_lan_ip(self) -> str:
        """Detects host primary LAN IP address."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
        except Exception:
            ip = "127.0.0.1"
        finally:
            s.close()
        return ip

    async def start(self):
        """Register and broadcast mDNS service on local LAN."""
        try:
            lan_ip = self._get_local_lan_ip()
            logger.info(f"Registering mDNS service for Edge Server at IP: {lan_ip}")

            self.aiozc = AsyncZeroconf(ip_version=IPVersion.V4Only)

            properties = {
                "version": settings.VERSION,
                "server_name": settings.APP_NAME,
                "api_port": str(settings.PORT),
                "webrtc_port": "8555",
                "tls_enabled": "true",
                "device_id": "edge-n100-cctv-01"
            }

            self.service_info = ServiceInfo(
                type_="_cctv-edge._tcp.local.",
                name=f"Edge-CCTV-Core._cctv-edge._tcp.local.",
                addresses=[socket.inet_aton(lan_ip)],
                port=settings.PORT,
                properties=properties,
                server="edge-cctv.local."
            )

            await self.aiozc.register_service(self.service_info)
            logger.info("mDNS service '_cctv-edge._tcp.local.' broadcast active.")
        except Exception as e:
            logger.error(f"Failed to start mDNS service: {e}")

    async def stop(self):
        """Unregister mDNS service cleanly upon server shutdown."""
        if self.aiozc and self.service_info:
            try:
                logger.info("Unregistering mDNS service...")
                await self.aiozc.unregister_service(self.service_info)
                await self.aiozc.close()
                logger.info("mDNS service stopped cleanly.")
            except Exception as e:
                logger.warning(f"Error stopping mDNS: {e}")


mdns_advertiser = EdgeMDNSAdvertiser()
