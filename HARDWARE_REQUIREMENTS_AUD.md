# Edge AI CCTV - Hardware Requirements & BOM (AUD)

**System Architecture:** 100% On-Premise Edge Surveillance & AI Analytics  
**Primary Platform:** Intel N100 Mini PC + Hailo M.2 AI Accelerator / QuickSync VA-API  
**Currency:** Australian Dollars (AUD) — Verified from live Australian market retailers (Amazon AU, Umart, Scorptec, Core Electronics, Jaycar).

---

## 1. Necessary Items (Core Edge System)

| Item | Use / Need | Cost (AUD) |
| :--- | :--- | :--- |
| **Intel N100 Mini PC**<br>*(e.g., GMKtec NucBox G3 / Beelink S12 Pro)* | **Host Edge Server**: Quad-Core CPU (up to 3.4GHz, 6W TDP) with Intel UHD QuickSync (hardware VA-API H.264/H.265 video decoding). Runs backend, DVR engine, and WebRTC streaming. | $249.00 |
| **RAM: 16 GB DDR4 / DDR5**<br>*(3200MHz / 4800MHz SODIMM)* | **System Memory**: Buffers incoming RTSP streams, maintains JPEG pre/post-roll ring buffers, and caches SQLite event logs in RAM. *(Included with Mini PC)*. | Included<br>*(or $59.00)* |
| **Storage: 512 GB M.2 NVMe SSD**<br>*(Crucial P3 Plus / Kingston NV2)* | **System Drive & 24/7 DVR Buffer**: OS, AI models, SQLite database, incident video clip archives, and rolling continuous DVR segments. | Included<br>*(or $65.00)* |
| **AI Accelerator: Hailo-8L M.2 Module**<br>*(13 TOPS, M.2 Key M / B+M)* | **Dedicated NPU / GPU**: Offloads YOLO11 multi-class object detection and 17-keypoint pose estimation to sub-10ms hardware inference with zero host CPU load. | $125.00 |
| **Network: 5-Port Gigabit PoE+ Switch**<br>*(TP-Link TL-SG1005P, 65W PoE)* | **Camera & Edge Network**: Supplies power and high-bandwidth data transmission to PoE IP cameras over single Cat6 cables without separate power adapters. | $65.00 |
| **Wi-Fi 6 Router / Access Point**<br>*(TP-Link Archer AX12 / AX23)* | **Local Wireless Gateway**: Serves encrypted LAN streams, Web HUD dashboard, and mDNS discovery for mobile apps and ESP32 wireless cameras. | $79.00 |
| **PoE IP Camera (4MP / 5MP)**<br>*(Reolink RLC-520A or Dahua WizSense 4MP)* | **Primary Video Feed**: High-resolution RTSP video stream with night vision and weatherproof (IP67) outdoor housing. | $89.00 |
| **Cabling & Power Infrastructure**<br>*(Cat6 UTP Patch Leads 15m + PDU cord)* | **Connectivity**: Connects IP cameras to PoE switch and edge host to the router. | $28.00 |
| **Total (Base Necessary System)** | **Complete Standalone Edge AI Surveillance Hub** | **~$607.00 – $635.00 AUD** |

---

## 2. Optional Upgrades & Sensor Add-ons

| Item | Use / Need | Cost (AUD) |
| :--- | :--- | :--- |
| **ESP32-S3 Camera Sentry Node**<br>*(Freenove S3 CAM / Seeed XIAO S3 Sense)* | **Satellite Camera & IoT Hub**: Wireless Wi-Fi video stream (800x600 SVGA) with breakout GPIO pins for physical sentry sensors. | $22.00 |
| **PIR Motion Sensor (HC-SR501 / AM312)** | **Physical Intrusion Detection**: Passive infrared human body heat detector; triggers instant edge alert webhooks and 15s DVR incident clips. | $5.50 |
| **Ultrasonic Distance Sensor (HC-SR04P)**<br>*(3.3V Native Logic)* | **Proximity & Perimeter Breach**: Measures physical distance (2cm – 400cm); detects vehicles in driveway or people approaching doorways. | $6.50 |
| **Magnetic Door Reed Switches (MC-38, 2x)** | **Door / Gate Contact Monitoring**: Simple 2-wire magnetic switches connected to ESP32 pullup pins (monitors front door, back door, or lock state). | $7.00 |
| **Hailo-8 Upgrade (26 TOPS Module)**<br>*(Hailo-8 M.2 Key M 2280)* | **Heavy Multi-Camera AI Upgrade**: Upgrades NPU from 13 to 26 TOPS; handles 8 to 16 concurrent camera feeds with YOLO11 detection and pose tracking. | $259.00 |
| **Mini UPS Battery Backup (650VA / 360W)**<br>*(CyberPower UT650E / APC Back-UPS)* | **Power Outage Protection**: Provides 45–60 minutes of clean backup power to Edge PC, switch, and cameras during grid blackouts or tampering. | $99.00 |
| **2 TB Surveillance Hard Drive / SSD**<br>*(WD Purple 2TB / Crucial P3 2TB)* | **Long-Term DVR Storage**: Expands continuous 24/7 video retention from 3–5 days to 30+ days across multiple camera channels. | $115.00 |
| **Active 5V Buzzer / Strobe Relay** | **Physical Alarm Deterrent**: Connects to ESP32 pin to trigger an audible local alarm or strobe light upon perimeter breach. | $4.50 |
| **Weatherproof Sentry Enclosure**<br>*(ABS Junction Box IP65)* | **Outdoor Sensor Housing**: Houses ESP32-S3 module, PIR sensor, and ultrasonic transducer for outdoor weather protection. | $14.00 |
| **Total (Full Pro Upgrade Package)** | **Extended Multi-Camera Storage, High TOPS, Sensors & UPS** | **+$532.50 AUD** |
