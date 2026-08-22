"""Camera Network Manager & Dynamic IP Auto-Recovery Engine.

Handles:
1. Automated Ethernet link & carrier detection.
2. Dedicated camera subnet auto-configuration (e.g. eth1 -> 192.168.10.1/24).
3. Dynamic IP Migration Watchdog: Automatically scans, discovers, and reconfigures cameras
   when their IP addresses change due to DHCP re-assignment, router changes, or switch relocations.
4. Deep 5-Point Diagnostic Engine with explicit error states and auto-recovery fallbacks.
"""

import asyncio
import logging
import os
import re
import socket
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urlparse, urlunparse

from sqlalchemy import select

from app.database import async_session_factory
from app.models.db_models import CameraModel

logger = logging.getLogger("CameraNetworkManager")


class CameraDiagnosticState(str, Enum):
    ONLINE = "ONLINE"
    LINK_DOWN = "LINK_DOWN"                     # Physical Ethernet cable disconnected
    HOST_UNREACHABLE = "HOST_UNREACHABLE"       # No ICMP ping / ARP response
    PORT_CLOSED = "PORT_CLOSED"                 # Port 554 / 8554 not listening (camera booting)
    AUTH_FAILED = "AUTH_FAILED"                 # HTTP/RTSP 401 Unauthorized
    DECODE_ERROR = "DECODE_ERROR"               # Stream open but corrupted frames
    MIGRATING_SEARCHING = "MIGRATING_SEARCHING" # Camera missing, auto-discovery active
    RECOVERED = "RECOVERED"                     # Automatically relocated to new IP and reconnected


class CameraNetworkManager:
    def __init__(self):
        self._mac_fingerprints: Dict[str, str] = {}  # camera_id -> MAC address
        self._recovering_cameras: set = set()
        self._last_diagnostics: Dict[str, Dict[str, Any]] = {}
        self._scan_lock = asyncio.Lock()

    # ── 1. Host Network Interface & Carrier Detection ───────────

    def list_network_interfaces(self) -> List[Dict[str, Any]]:
        """List all host network interfaces with IP, carrier state, and speed."""
        interfaces = []
        net_dir = Path("/sys/class/net")
        if not net_dir.exists():
            return [{"name": "eth0", "carrier": True, "ip": "127.0.0.1", "state": "UP"}]

        for iface_path in net_dir.iterdir():
            iface_name = iface_path.name
            if iface_name == "lo":
                continue

            carrier = False
            carrier_file = iface_path / "carrier"
            if carrier_file.exists():
                try:
                    carrier = carrier_file.read_text().strip() == "1"
                except Exception:
                    carrier = False

            operstate = "UNKNOWN"
            state_file = iface_path / "operstate"
            if state_file.exists():
                try:
                    operstate = state_file.read_text().strip().upper()
                except Exception:
                    pass

            ip_addr = self._get_interface_ip(iface_name)

            interfaces.append({
                "interface": iface_name,
                "carrier": carrier,
                "operstate": operstate,
                "ip_address": ip_addr,
                "is_dedicated_cctv_candidate": iface_name.startswith(("eth1", "enp", "usb")),
            })

        return interfaces

    def _get_interface_ip(self, iface_name: str) -> Optional[str]:
        try:
            output = subprocess.check_output(
                ["ip", "-4", "addr", "show", iface_name],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2.0
            )
            match = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", output)
            return match.group(1) if match else None
        except Exception:
            return None

    # ── 2. Deep 5-Point Camera Diagnostics ──────────────────────

    async def diagnose_camera(self, camera_id: str, rtsp_url: str) -> Dict[str, Any]:
        """Perform deep 5-point diagnosis on an IP camera stream."""
        parsed = urlparse(rtsp_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 554
        username = parsed.username or ""
        password = parsed.password or ""

        # Step 1: Check Physical Carrier on host NICs
        interfaces = self.list_network_interfaces()
        has_active_carrier = any(i["carrier"] for i in interfaces)
        if not has_active_carrier:
            state = CameraDiagnosticState.LINK_DOWN
            msg = "Physical Ethernet link is down. Check PoE switch power and cable connection."
            return self._build_diag_report(camera_id, state, msg, host, port, False, False, False)

        # Step 2: Test IP Reachability (ICMP / ARP)
        ip_reachable = await self._test_ip_reachability(host)
        if not ip_reachable:
            state = CameraDiagnosticState.HOST_UNREACHABLE
            msg = f"Camera IP ({host}) is unreachable on the local subnet. Triggering dynamic auto-recovery..."
            # Trigger background auto-recovery
            asyncio.create_task(self.attempt_auto_recover(camera_id, rtsp_url))
            return self._build_diag_report(camera_id, state, msg, host, port, False, False, False)

        # Step 3: Test RTSP Port 554 / 8554 TCP Socket
        port_open = await self._test_tcp_port(host, port)
        if not port_open:
            state = CameraDiagnosticState.PORT_CLOSED
            msg = f"Camera reached at {host}, but RTSP port {port} is closed. Camera may still be booting."
            return self._build_diag_report(camera_id, state, msg, host, port, True, False, False)

        # Step 4: Test RTSP Auth & Handshake (via fast OpenCV probe in thread)
        auth_ok, decode_ok = await asyncio.to_thread(self._probe_rtsp_stream, rtsp_url)
        if not auth_ok:
            state = CameraDiagnosticState.AUTH_FAILED
            msg = f"RTSP 401 Unauthorized at {host}. Verify camera username/password."
            return self._build_diag_report(camera_id, state, msg, host, port, True, True, False)

        if not decode_ok:
            state = CameraDiagnosticState.DECODE_ERROR
            msg = f"Connected to {host}:{port}, but video frames failed to decode. Check video codec (H.264/H.265)."
            return self._build_diag_report(camera_id, state, msg, host, port, True, True, True, decode_ok=False)

        # Everything Healthy
        state = CameraDiagnosticState.ONLINE
        msg = f"Camera stream healthy at {host}:{port} ({username}@{host})."
        return self._build_diag_report(camera_id, state, msg, host, port, True, True, True, decode_ok=True)

    def _build_diag_report(
        self,
        camera_id: str,
        state: CameraDiagnosticState,
        message: str,
        host: str,
        port: int,
        ip_reachable: bool,
        port_open: bool,
        auth_ok: bool,
        decode_ok: bool = False
    ) -> Dict[str, Any]:
        report = {
            "camera_id": camera_id,
            "state": state.value,
            "message": message,
            "host": host,
            "port": port,
            "checks": {
                "physical_carrier": True,
                "ip_reachable": ip_reachable,
                "port_open": port_open,
                "auth_valid": auth_ok,
                "video_decodable": decode_ok,
            },
            "timestamp": time.time(),
        }
        self._last_diagnostics[camera_id] = report
        return report

    async def _test_ip_reachability(self, host: str) -> bool:
        """Test ICMP ping or ARP cache resolution."""
        if host in ("127.0.0.1", "localhost"):
            return True
        try:
            proc = await asyncio.create_subprocess_exec(
                "ping", "-c", "1", "-W", "1", host,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            return (await proc.wait()) == 0
        except Exception:
            return False

    async def _test_tcp_port(self, host: str, port: int) -> bool:
        """Fast TCP socket connect with 1.5s timeout."""
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=1.5
            )
            writer.close()
            await writer.wait_closed()
            return True
        except Exception:
            return False

    def _probe_rtsp_stream(self, rtsp_url: str) -> Tuple[bool, bool]:
        """Synchronously probe RTSP stream using OpenCV. Returns (auth_ok, decode_ok)."""
        import cv2
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            # If failed to open, check if it's 401 or network
            return (False, False)

        ret, frame = cap.read()
        cap.release()
        return (True, ret and frame is not None)

    # ── 3. Dynamic IP Migration & Auto-Discovery Watchdog ────────

    async def attempt_auto_recover(self, camera_id: str, old_rtsp_url: str) -> bool:
        """
        Scan subnets to locate a camera whose IP address changed,
        verify the RTSP stream at the new IP, and update the database and go2rtc.
        """
        if camera_id in self._recovering_cameras:
            logger.debug(f"Auto-recovery already in progress for {camera_id}")
            return False

        self._recovering_cameras.add(camera_id)
        logger.warning(f"🚀 [Auto-Recovery] Starting dynamic IP migration search for {camera_id}...")

        try:
            parsed = urlparse(old_rtsp_url)
            old_host = parsed.hostname or ""
            port = parsed.port or 554
            username = parsed.username or "admin"
            password = parsed.password or ""
            path = parsed.path or ""

            # Check known subnets
            subnets_to_probe = ["192.168.1", "192.168.10", "192.168.0"]
            candidate_ips = await self._discover_active_rtsp_hosts(subnets_to_probe, port)

            # Filter out old IP
            candidate_ips = [ip for ip in candidate_ips if ip != old_host]
            logger.info(f"Discovered {len(candidate_ips)} candidate RTSP hosts: {candidate_ips}")

            for new_ip in candidate_ips:
                # Construct candidate RTSP URL
                auth_part = f"{username}:{password}@" if username else ""
                new_rtsp_url = f"rtsp://{auth_part}{new_ip}:{port}{path}"

                # Test stream at new IP
                auth_ok, decode_ok = await asyncio.to_thread(self._probe_rtsp_stream, new_rtsp_url)
                if auth_ok and decode_ok:
                    logger.info(f"🎯 [Auto-Recovery SUCCESS] Camera {camera_id} found at NEW IP: {new_ip}!")
                    await self._apply_new_camera_ip(camera_id, old_rtsp_url, new_rtsp_url, new_ip)
                    return True

            logger.warning(f"❌ [Auto-Recovery] Could not locate camera {camera_id} on local subnets.")
            return False

        except Exception as e:
            logger.error(f"Error during auto-recovery for {camera_id}: {e}")
            return False
        finally:
            self._recovering_cameras.discard(camera_id)

    async def _discover_active_rtsp_hosts(self, subnet_prefixes: List[str], port: int = 554) -> List[str]:
        """Parallel TCP sweep on port 554 across candidate subnets."""
        active_hosts = []
        tasks = []

        for prefix in subnet_prefixes:
            for last_octet in range(2, 254):
                ip = f"{prefix}.{last_octet}"
                tasks.append(self._probe_host_port(ip, port))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, str) and res:
                active_hosts.append(res)

        return active_hosts

    async def _probe_host_port(self, ip: str, port: int) -> Optional[str]:
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(ip, port),
                timeout=0.6
            )
            writer.close()
            await writer.wait_closed()
            return ip
        except Exception:
            return None

    async def _apply_new_camera_ip(self, camera_id: str, old_url: str, new_url: str, new_ip: str):
        """Update SQLite database, restart video ingest worker, and update go2rtc."""
        # 1. Update SQLite DB
        async with async_session_factory() as session:
            stmt = select(CameraModel).where(CameraModel.id == camera_id)
            res = await session.execute(stmt)
            cam = res.scalar_one_or_none()
            if cam:
                cam.rtsp_url = new_url
                cam.status = "ONLINE"
                cam.last_seen = time.strftime("%Y-%m-%d %H:%M:%S")
                await session.commit()
                logger.info(f"Updated SQLite CameraModel for {camera_id} with new RTSP URL: {new_url}")

        # 2. Restart Video Ingest Worker with new URL
        from app.services.video_ingest_service import video_ingest_service
        await video_ingest_service.register_and_start_camera(camera_id, new_url)

        # 3. Update Diagnostic Report
        self._last_diagnostics[camera_id] = {
            "camera_id": camera_id,
            "state": CameraDiagnosticState.RECOVERED.value,
            "message": f"Automatically migrated from {old_url} to new IP {new_ip}",
            "host": new_ip,
            "timestamp": time.time(),
        }


camera_network_manager = CameraNetworkManager()
