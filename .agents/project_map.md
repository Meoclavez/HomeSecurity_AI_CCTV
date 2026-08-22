# Edge AI CCTV Surveillance System - Project Map

## 1. System Architecture Overview
The system is a 100% edge-processed CCTV AI surveillance platform designed for Intel N100 Mini PCs paired with Hailo M.2 PCIe AI accelerators (Hailo-8 / 8L), streaming to a cross-platform Flutter client accessible across **PC Web/Desktop, Android, and iOS**.

```
 [ IP Cameras (RTSP) ]
          │
          ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                  INTEL N100 + HAILO M.2 EDGE BACKEND                       │
 │                                                                            │
 │  1. Coturn TURN Relay (:3478/:5349): Dynamic HMAC-SHA1 secret auth for     │
 │     WebRTC across symmetric 4G/5G cellular NATs                            │
 │  2. Caddy Reverse Proxy (:443): Local CA TLS termination for HTTPS & WSS   │
 │  3. mDNS Zeroconf Broadcaster: Auto-advertises Edge box on LAN             │
 │  4. go2rtc Gateway (:1984/:8555): RTSP ingest, WebRTC (<300ms) & backchannel│
 │  5. QuickSync (VA-API /dev/dri): Hardware H.264/H.265 frame decoder        │
 │  6. 24/7 Segmented DVR Engine: 1-min zero-copy MP4 chunks & dynamic HLS   │
 │  7. Privacy Masking Engine: Blackout/blur/mosaic directly on source frames │
 │  8. HailoRT (/dev/hailo0): YOLOv8 detection, 17-keypoint pose kinematics,  │
 │     directional tripwires (A->B, B->A), polygon intrusion, door & package  │
 │     state machines with dynamic credit scheduler                           │
 │  9. SQLite Persistence (SQLAlchemy 2.0 Async + aiosqlite): Events,         │
 │     DVR segments, incident archives, and device tokens                     │
 └─────────────────────────────────────┬──────────────────────────────────────┘
                                       │ (HTTPS / WSS / WebRTC + Opus Mic)
                                       ▼
 ┌────────────────────────────────────────────────────────────────────────────┐
 │             CROSS-PLATFORM FLUTTER CLIENT (DESKTOP / WEB / MOBILE)         │
 │                                                                            │
 │  • Adaptive Navigation Shell: Mobile bottom nav, Tablet rail, PC sidebar   │
 │  • Live Multi-Cam Grid Wall: 1 to 16 adaptive camera tiles                 │
 │  • 24/7 Continuous DVR Player: 60fps timeline with pinch zoom & snapping   │
 │  • Visual Zone & Mask Canvas: Draw & drag polygons/tripwires on snapshots  │
 │  • System & SMART Storage Dashboard: Disk gauges, wear level, camera quota │
 │  • AI Incident & Alerts Center: Filtered playback & emergency alarms       │
 │  • Biometric Gate: FaceID / Fingerprint lock with 60s grace period         │
 │  • WebRTC 2-Way Audio Talkback: Native Opus microphone backchannel         │
 └────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Structure & Module Index

```
Edge_AI_CCTV/
├── .agents/
│   └── project_map.md                        # Current file: System architecture & API index
│
├── edge_backend/                             # Edge Mini PC Backend Services
│   ├── app/
│   │   ├── main.py                           # FastAPI app, SQLite lifecycle, mDNS, 24/7 DVR, cleaner task
│   │   ├── config.py                         # Settings, retention policies, Coturn keys, kinematics thresholds
│   │   ├── database.py                       # SQLAlchemy async SQLite session factory (aiosqlite)
│   │   ├── models/
│   │   │   ├── schemas.py                    # Pydantic schemas (events, zones, DVR, timeline, telemetry, setup)
│   │   │   └── db_models.py                  # SQLAlchemy ORM models (cameras, events, admin_users, system_setup)
│   │   ├── routes/
│   │   │   ├── health.py                     # Hardware & telemetry monitoring endpoint (/api/v1/health)
│   │   │   ├── setup.py                      # First-time setup wizard & auth pairing API (/api/v1/setup, /api/v1/auth)
│   │   │   ├── cameras.py                    # Camera CRUD, snapshot routes, device token registration
│   │   │   ├── events.py                     # AI event ingestion, SQLite persistence, clip streaming
│   │   │   ├── webrtc.py                     # WebRTC SDP offer/answer exchange & dynamic ICE servers
│   │   │   ├── dvr.py                        # 24-Hour Timeline, dynamic HLS playlists, incident exports, storage health
│   │   │   └── zones.py                      # Camera privacy masks, tripwires, intrusion zones & alert muting
│   │   └── services/
│   │       ├── camera_network_manager.py     # Plug-and-Play NIC DHCP, 5-point camera diagnostics & IP migration watchdog
│   │       ├── hailo_inference_service.py    # HailoRT PCIe (.hef) runner & kinematic fall engine
│   │       ├── ai_zone_service.py            # Privacy masking, tripwires, polygon PIP, state machines & scheduler
│   │       ├── dvr_recorder.py               # 24/7 continuous zero-copy segmenter, HLS, stitcher & SMART health
│   │       ├── video_ingest_service.py       # Threaded QuickSync VA-API frame grabber with source-level privacy masking
│   │       ├── clip_recorder.py              # JPEG ring-buffer pre/post MP4 generator & StorageCleaner
│   │       ├── notification_service.py       # APNs Critical Alert & FCM USAGE_ALARM persistent dispatcher
│   │       ├── turn_service.py               # RFC 5766 dynamic ephemeral TURN credentials generator
│   │       ├── mdns_service.py               # Bonjour/Zeroconf mDNS advertiser for LAN discovery
│   │       └── auth_service.py               # Tokenized session manager & path traversal sanitizer
│   ├── tests/
│   │   ├── test_api.py                       # REST API, auth, ICE, zones, timeline & storage tests
│   │   └── test_kinematics.py                # Kinematics, polygon ray-casting & tripwire crossing unit tests
│   ├── coturn/coturn.conf                    # Coturn TURN/STUN relay configuration (RFC 5766)
│   ├── Caddyfile                             # Caddy reverse proxy config (TLS termination for HTTPS/WSS)
│   ├── scripts/generate_certs.py             # Automated local TLS certificate generator with SAN extensions
│   ├── go2rtc.yaml                           # go2rtc Media Gateway config (localhost API binding)
│   ├── requirements.txt                      # Backend dependencies (FastAPI, SQLAlchemy, zeroconf, etc.)
│   ├── Dockerfile                            # Multi-stage container with HailoRT & VA-API
│   └── docker-compose.yml                    # Unified stack: coturn, go2rtc, edge_api, caddy, tailscale
│
└── mobile_app/                               # Cross-Platform Flutter Client (Desktop, Web, Mobile)
    ├── pubspec.yaml                          # Flutter package dependencies (local_auth, webrtc, notifications)
    └── lib/
        ├── main.dart                         # Bootstrap with AppShell & top-level background FCM isolate
        ├── core/
        │   ├── constants/api_constants.dart  # Endpoints, WebRTC STUN/TURN, WebSocket paths
        │   └── theme/app_theme.dart          # Modern high-contrast dark theme with GlassCard, CyberBadge, TelemetryChip
        ├── models/
        │   ├── camera_feed.dart              # Camera model (WebRTC URL, RTSP, DVR settings, zones)
        │   ├── security_event.dart           # AI event model (fall, intrusion, door, timestamps)
        │   └── zone_model.dart               # Polygon, tripwire & privacy mask models
        ├── services/
        │   ├── api_service.dart              # REST client with auto-failover and base URL switching
        │   ├── discovery_service.dart        # Universal mDNS discovery + parallel subnet sweep (Web-safe)
        │   ├── biometric_auth_service.dart   # FaceID / Fingerprint manager with 60s grace period
        │   ├── webrtc_service.dart           # WebRTC manager with H.264/Opus SDP prioritization & 2-way talkback
        │   └── notification_service.dart     # Push handler with lockscreen interactive actions (View, Mute, Call)
        ├── widgets/
        │   ├── biometric_gate.dart           # Biometric authentication screen wrapper
        │   ├── talkback_button.dart          # Push-to-Talk 2-way audio button with animated wave
        │   ├── timeline_models.dart          # Timeline recording segment and event pin models
        │   ├── timeline_painter.dart         # High-performance 60fps CustomPainter for 24h timeline
        │   ├── timeline_scrubber_widget.dart # Gesture scrubber with pinch zoom 1h-24h and magnetic snapping
        │   └── zone_canvas_painter.dart      # Interactive canvas painter for visual polygon/tripwire drawing
        └── screens/
            ├── app_shell.dart                # Master adaptive responsive AppShell (Mobile, Tablet, PC/Web)
            ├── login_screen.dart             # Multi-mode Login (Biometrics, PIN, QR Scan, Password)
            ├── setup_wizard_screen.dart      # 6-step First-Time Setup Wizard (Hardware, RTSP Scan, Admin)
            ├── settings_screen.dart          # System Settings & Hardware Telemetry Dashboard
            ├── multi_cam_grid_screen.dart    # Adaptive 1 to 16 camera live grid wall
            ├── live_view_screen.dart         # Fullscreen WebRTC live player with talkback & 24h timeline
            ├── dvr_playback_screen.dart      # 24/7 continuous DVR timeline player
            ├── events_center_screen.dart     # Filtered AI incident center
            ├── zone_editor_screen.dart       # Visual zone & privacy mask drawing canvas
            ├── clip_archives_screen.dart     # Incident video clip archives & export manager
            ├── storage_health_screen.dart    # System telemetry & SMART health gauges dashboard
            ├── clip_player_screen.dart       # Event video clip player with scrubber & anomaly tags
            └── emergency_alert_screen.dart   # Fullscreen emergency takeover modal with biometric dismiss
```

---

## 3. REST API Reference

### Zero-Trust & WebRTC Signaling (`/api/v1/webrtc`)
* `GET /api/v1/webrtc/ice-servers`: Dynamic RFC 5766 HMAC-SHA1 authenticated STUN/TURN ICE server list.
* `POST /api/v1/webrtc/offer`: Exchange SDP Offer/Answer with go2rtc (supports 2-way audio backchannel).
* `GET /api/v1/webrtc/token`: Short-lived stream authorization tokens.

### 24/7 Segmented DVR & 24-Hour Timeline (`/api/v1/dvr`, `/api/v1/cameras/{id}/timeline`)
* `GET /api/v1/cameras/{camera_id}/timeline?date=YYYY-MM-DD`: 24-hour recorded segments, gap intervals, and aligned AI event markers.
* `GET /api/v1/dvr/cameras/{camera_id}/hls/{date}/index.m3u8`: Dynamic HLS playlist with discontinuity tags across recording gaps.
* `GET /api/v1/dvr/segments/{segment_id}/video`: Stream 1-minute MP4 segment with HTTP 206 Partial Content support.
* `POST /api/v1/cameras/{camera_id}/export`: Lossless zero-copy stitching of custom time windows into standalone MP4s.
* `GET /api/v1/dvr/archives`: List and download archived incident video clips.
* `GET /api/v1/storage/health`: SMART drive health, temperature, external drive detection, and per-camera quotas.

### Camera Privacy & AI Zones (`/api/v1/cameras/{id}/zones`)
* `GET /api/v1/cameras/{camera_id}/zones`: List privacy masks, tripwires, and intrusion polygons.
* `POST /api/v1/cameras/{camera_id}/zones`: Create or update a zone configuration.
* `DELETE /api/v1/cameras/{camera_id}/zones/{zone_id}`: Delete a zone configuration.
* `POST /api/v1/cameras/{camera_id}/mute`: Mute camera alerts for X minutes (invoked from lockscreen action).
