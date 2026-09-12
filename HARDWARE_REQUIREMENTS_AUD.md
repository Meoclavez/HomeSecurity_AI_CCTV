# Edge AI CCTV - Hardware Requirements & BOM (AUD)

**Architecture:** 100% On-Premise Edge Surveillance (Zero Cloud Costs)  
**Primary Compute:** Intel N100 Mini PC using built-in Intel UHD iGPU (OpenVINO FP16)  
**Currency:** Australian Dollars (AUD) — Verified from live Australian retailers (Amazon AU, Umart, Scorptec, Core Electronics, Jaycar).

---

## 1. Necessary Items (Core Edge System)

| Item | Use / Need | Cost (AUD) |
| :--- | :--- | :--- |
| **Intel N100 Mini PC**<br>*(e.g., GMKtec G3 / Beelink S12 Pro)* | **Host Edge Server**: Quad-Core CPU (up to 3.4GHz, 6W TDP) with Intel UHD QuickSync hardware video decoding (VA-API H.264/H.265). Runs FastAPI backend, SQLite DB, DVR engine, and WebRTC streaming. | $249.00 |
| **RAM: 16 GB DDR4 / DDR5**<br>*(3200MHz / 4800MHz SODIMM)* | **System Memory**: Buffers incoming camera RTSP streams, maintains in-memory JPEG ring buffers, and caches SQLite event logs in RAM. | Included<br>*(with PC)* |
| **Storage: 512 GB M.2 NVMe SSD**<br>*(Crucial P3 Plus / Kingston NV2)* | **System Drive & 24/7 DVR**: Houses OS, AI models, SQLite database, incident video clip archives, and rolling continuous DVR recording segments. | Included<br>*(with PC)* |
| **Integrated GPU (Intel UHD 24 EUs)**<br>*(OpenVINO FP16 Engine)* | **AI Inference (1–4 Cameras)**: Built-in iGPU runs YOLO11 multi-class object detection and pose estimation directly in OpenVINO. **Zero additional cost** (Hailo module avoided). | **$0.00**<br>*(Included)* |
| **5-Port Gigabit PoE+ Switch**<br>*(TP-Link TL-SG1005P, 65W PoE)* | **Camera & Network Power**: Delivers data transmission and direct power to PoE IP cameras over single Cat6 cables without separate power adapters. | $65.00 |
| **Wi-Fi 6 Router / Access Point**<br>*(TP-Link Archer AX12 / AX23)* | **Local Wireless Gateway**: Serves encrypted LAN streams, Web HUD dashboard, and mDNS auto-discovery for mobile apps and ESP32 wireless cameras. | $79.00 |
| **PoE IP Camera (4MP / 5MP)**<br>*(Reolink RLC-520A / Dahua 4MP)* | **Primary Video Feed**: High-resolution outdoor RTSP video stream with infrared night vision and weatherproof (IP67) housing. | $89.00 |
| **Cabling & Power Infrastructure**<br>*(Cat6 Patch Leads 15m + PDU cord)* | **Physical Connectivity**: Connects IP cameras to PoE switch and edge host to the router. | $28.00 |
| **Total (Base Necessary System)** | **Complete Standalone Edge AI Surveillance Hub** | **~$510.00 AUD** |

---

## 2. Optional Upgrades & Components

| Item | Use / Need | Cost (AUD) |
| :--- | :--- | :--- |
| **Hailo-8L M.2 AI Module (13 TOPS)**<br>*(M.2 Key M / B+M 2242/2280)* | **Additional Cameras (4 to 8 Feeds)**: Offloads AI inference from host iGPU to dedicated NPU hardware; enables sub-10ms YOLO11 detection across multiple additional camera channels. | $125.00 |
| **Hailo-8 M.2 AI Module (26 TOPS)**<br>*(M.2 Key M 2280)* | **Commercial Multi-Camera Expansion (8 to 16 Feeds)**: Heavy-duty NPU upgrade running up to 16 concurrent camera feeds with YOLO11 detection and 17-keypoint pose tracking simultaneously. | $259.00 |
| **ESP32-S3 Camera Sentry Node**<br>*(Freenove S3 CAM / Seeed XIAO Sense)* | **Satellite Camera & IoT Sensor Hub**: Wireless Wi-Fi video stream (800x600 SVGA) with breakout GPIO pins for physical sentry sensors. | $22.00 |
| **PIR Motion Sensor (HC-SR501 / AM312)** | **Physical Intrusion Detection**: Passive infrared human body heat detector; triggers instant edge alert webhooks and 15s DVR incident clips. | $5.50 |
| **Ultrasonic Distance Sensor (HC-SR04P)**<br>*(3.3V Native Logic)* | **Proximity & Perimeter Breach**: Measures physical distance (2cm – 400cm); detects vehicles in driveway or people approaching doorways. | $6.50 |
| **Magnetic Door Reed Switches (MC-38, 2x)** | **Door / Gate Contact Monitoring**: Simple 2-wire magnetic switches connected to ESP32 pullup pins (monitors front door, back door, or lock state). | $7.00 |
| **Mini UPS Battery Backup (650VA / 360W)**<br>*(CyberPower UT650E / APC Back-UPS)* | **Power Outage Protection**: Provides 45–60 minutes of clean backup power to Edge PC, switch, and cameras during grid blackouts or wire tampering. | $99.00 |
| **2 TB Surveillance Storage Expansion**<br>*(WD Purple 2TB / Crucial P3 2TB)* | **Long-Term DVR Retention**: Expands continuous 24/7 video storage from 3–5 days to 30+ days across multiple camera channels. | $115.00 |
| **Active 5V Buzzer / Strobe Relay** | **Physical Alarm Deterrent**: Connects to ESP32 pin to trigger an audible local siren or strobe light upon perimeter breach. | $4.50 |
| **Weatherproof Sentry Enclosure**<br>*(ABS Junction Box IP65)* | **Outdoor Sensor Housing**: Houses ESP32-S3 module, PIR sensor, and ultrasonic transducer for outdoor weather protection. | $14.00 |
