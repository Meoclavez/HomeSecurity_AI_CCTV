# 📷 ESP32-S3 IP Camera & IoT Hardware Sensor Firmware

This firmware turns any **ESP32-S3 Camera module** (OV2640 / OV5640 / OV3660) into a dedicated high-performance RTSP & MJPEG IP security camera and multi-sensor IoT security node for the **Edge AI CCTV Surveillance System**.

---

## 🏗️ FreeRTOS Dual-Core Architecture

The firmware utilizes the ESP32-S3's dual-core Xtensa LX7 processor with strict task isolation:

* **Core 1 (Streaming Core):** High-priority MJPEG video capture engine running at 16 MHz XCLK with DMA double-buffering, cooperative yields, and zero-copy packet transmission. Delivers stable 25–30 FPS at SVGA/VGA resolutions.
* **Core 0 (Sensor Core):** Dedicated FreeRTOS background task (`sensorTask` via `xTaskCreatePinnedToCore`) polling hardware sensors every 200 ms with debouncing, distance calculation, state tracking, and automatic edge backend webhook dispatching. Polling never blocks video streaming.

---

## 🔌 Supported ESP32-S3 Camera Boards

| Board Profile | Default Status | Notes |
| :--- | :--- | :--- |
| `CAMERA_MODEL_FREENOVE_ESP32S3_CAM` | **Default** | Freenove ESP32-S3 WROOM CAM with built-in USB-C & micro-SD |
| `CAMERA_MODEL_XIAO_ESP32S3_SENSE` | Optional | Ultra-compact Seeed Studio module with expansion board |
| `CAMERA_MODEL_AI_THINKER_ESP32S3` | Optional | AI-Thinker ESP32-S3 CAM module |
| `CAMERA_MODEL_ESP32S3_EYE` | Optional | Espressif ESP32-S3-EYE / LilyGO T-Camera S3 |
| `CAMERA_MODEL_AI_THINKER_ESP32_CAM` | Legacy | Legacy ESP32 (Non-S3, single-core fallback) |

*(To select your board, uncomment the matching definition in [`camera_pins.h`](./camera_pins.h))*

---

## 🧰 Hardware Sensor Pinout Mapping

All sensor pin assignments are verified to be conflict-free with the camera parallel data bus (D0–D7), clock lines (XCLK, PCLK), synchronization lines (VSYNC, HREF), and I2C SCCB lines (SIOD, SIOC):

| Board Profile | PIR Motion (`PIN_PIR`) | Ultrasonic Trig (`PIN_US_TRIG`) | Ultrasonic Echo (`PIN_US_ECHO`) | Door 1 (`PIN_DOOR1`) | Door 2 (`PIN_DOOR2`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Freenove ESP32-S3 CAM** | **GPIO 1** | **GPIO 14** | **GPIO 21** | **GPIO 47** | **GPIO 3** |
| **XIAO ESP32S3 Sense** | **GPIO 1** | **GPIO 2** | **GPIO 3** | **GPIO 4** | **GPIO 5** |
| **AI-Thinker ESP32-S3** | **GPIO 4** | **GPIO 5** | **GPIO 6** | **GPIO 7** | **GPIO 8** |
| **AI-Thinker ESP32-CAM (Legacy)**| **GPIO 13** | **GPIO 14** | **GPIO 15** | **GPIO 12** | **GPIO 16** |

---

## ⚠️ Critical Hardware Wiring Instructions & Warnings

### 1. HC-SR04 Ultrasonic Distance Sensor (3.3V Logic Level Warning)
> **HIGH VOLTAGE WARNING:** Standard HC-SR04 ultrasonic modules require **5V VCC** and output a **5V logic pulse** on their `ECHO` pin.
> Connecting the 5V `ECHO` pin directly to an ESP32-S3 GPIO pin will permanently fry the SoC's internal ESD protection diodes and damage the chip!

To safely connect an HC-SR04 ultrasonic sensor:
* **Option A (Voltage Divider - Recommended for standard HC-SR04):**
  Place a simple resistive voltage divider on the `ECHO` line:
  ```text
  HC-SR04 ECHO Pin ────[ 1.0 kΩ ]────┬────> ESP32 PIN_US_ECHO (GPIO 21)
                                     │
                                 [ 2.0 kΩ ]
                                     │
                                    GND
  ```
  This steps down the 5.0V echo signal to approximately **3.33V**, which is safe for ESP32 GPIO pins.
* **Option B (3.3V Native Ultrasonic Sensors):**
  Use a 3.3V-tolerant sensor such as the **HC-SR04P**, **RCWL-1601**, or **US-015**, which can run directly on 3.3V VCC and output native 3.3V logic.

### 2. PIR Motion Sensor (AM312 or HC-SR501)
* **AM312 (Mini PIR):** Connect `VCC` to **3.3V**, `GND` to **GND**, and `OUT` to `PIN_PIR` (e.g. GPIO 1).
* **HC-SR501:** Connect `VCC` to **5V**, `GND` to **GND**, and `OUT` to `PIN_PIR` (the HC-SR501 output is natively 3.3V logic level).
* Configured as digital input (`pinMode(PIN_PIR, INPUT)`). Output is `HIGH` when motion is detected and `LOW` when idle.

### 3. Magnetic Door Contact Reed Switches (Door 1 & Door 2)
* Connect one terminal of the reed switch to **GND**.
* Connect the other terminal directly to `PIN_DOOR1` (e.g. GPIO 47) or `PIN_DOOR2` (e.g. GPIO 3).
* The firmware configures these pins with internal pull-up resistors (`INPUT_PULLUP`):
  * **Door Closed:** Magnet is adjacent to switch $\rightarrow$ contacts closed to GND $\rightarrow$ GPIO reads `LOW` (`door_open = false`).
  * **Door Opened:** Magnet pulls away $\rightarrow$ contacts open $\rightarrow$ internal pullup drives GPIO `HIGH` (`door_open = true`).
* Includes software debouncing (2 consecutive cycles / 400 ms) to eliminate mechanical contact bounce.

---

## 📡 HTTP Endpoints & API Reference

The ESP32-S3 hosts an embedded HTTP server on **port 80** and high-speed video streaming on **port 81**:

### 1. GET `/sensors` — Live Sensor Telemetry
Returns current real-time telemetry from all hardware sensors and active runtime toggles.

* **Method:** `GET`
* **URL:** `http://<esp32-ip>/sensors`
* **Response:** `200 OK` (`application/json`)
```json
{
  "camera_id": "cam_living_room",
  "pir_motion": true,
  "distance_cm": 42.6,
  "door1_open": false,
  "door2_open": false,
  "toggles": {
    "pir": true,
    "ultrasonic": true,
    "door1": true,
    "door2": true
  }
}
```

* **Field Definitions:**
  * `camera_id`: Configured identifier for this camera unit.
  * `pir_motion`: `true` if motion is actively detected by the PIR sensor.
  * `distance_cm`: Distance measured by HC-SR04 in centimeters (`-1.0` if out of range or timeout).
  * `door1_open` / `door2_open`: `true` if door reed switch is open (disconnected from GND).
  * `toggles`: Real-time enable/disable status for each individual sensor.

### 2. POST `/sensors/config` — Dynamic Sensor Configuration
Updates sensor enable/disable toggles, distance thresholds, or backend integration parameters dynamically without rebooting the ESP32.

* **Method:** `POST`
* **URL:** `http://<esp32-ip>/sensors/config`
* **Headers:** `Content-Type: application/json`
* **Payload Example:**
```json
{
  "pir": true,
  "ultrasonic": true,
  "door1": true,
  "door2": false,
  "threshold_cm": 60.0,
  "camera_id": "cam_living_room",
  "edge_backend_ip": "192.168.1.100"
}
```
* **Response:** `200 OK` (`application/json`)
```json
{
  "status": "success",
  "camera_id": "cam_living_room",
  "distance_threshold_cm": 60.0,
  "toggles": {
    "pir": true,
    "ultrasonic": true,
    "door1": true,
    "door2": false
  }
}
```

### 3. Video & System Telemetry Endpoints
| Endpoint | Method | Port | Description |
| :--- | :---: | :---: | :--- |
| `http://<esp32-ip>:81/stream` | `GET` | 81 | High-speed dedicated MJPEG stream (25–30 FPS) |
| `http://<esp32-ip>/stream` | `GET` | 80 | Standard MJPEG stream |
| `http://<esp32-ip>/capture` | `GET` | 80 | High-resolution single JPEG snapshot |
| `http://<esp32-ip>/status` | `GET` | 80 | Device telemetry (heap, PSRAM, RSSI, IP) |
| `http://<esp32-ip>/` | `GET` | 80 | Interactive web portal with live player & links |

---

## 🚨 Alert Webhook Dispatcher

When an active intrusion or breach occurs, the Core 0 `sensorTask` automatically formats and dispatches an HTTP POST webhook directly to the Edge AI backend:

* **Target URL:** `http://<edge_backend_ip>:8000/api/v1/events/trigger`
* **Security Header:** `X-Edge-API-Key: <edge_api_key>`
* **Event Severity Mapping:**
  * **Door Opened:** `event_type = "DOOR_LEFT_OPEN"`, `severity = "HIGH"`, `confidence = 1.0`
  * **PIR Motion Detected:** `event_type = "INTRUSION_DETECTED"`, `severity = "WARNING"`, `confidence = 0.95`
  * **Ultrasonic Distance < Threshold (default 50 cm):** `event_type = "PERIMETER_BREACH"`, `severity = "WARNING"`, `confidence = 0.90`
* **Flood Prevention & Rate Limiting:**
  * **Hardware Debouncing:** Magnetic door contacts and ultrasonic distance readings require 2 consecutive matching 200 ms cycles (400 ms total) before triggering.
  * **5-Second Alert Cooldown:** Enforces a strict 5000 ms silence interval between consecutive webhooks to eliminate notification storms while keeping the edge server responsive.

---

## ⚡ Flashing Instructions

### Method 1: Arduino IDE
1. Install the **esp32 by Espressif Systems** board package (v2.0.14+ or v3.0.0+).
2. Select **Board**: `ESP32S3 Dev Module`
3. Configure:
   * **PSRAM:** `OPI PSRAM` ⚠️ *(Required for double-buffered DMA streaming)*
   * **Flash Size:** `8MB` or `16MB`
   * **Partition Scheme:** `Huge APP (3MB No OTA/1MB SPIFFS)` or `8M with spiffs`
   * **CPU Frequency:** `240MHz (WiFi)`
4. Set Wi-Fi credentials in `esp32_s3_cctv_cam.ino`:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```
5. Click **Upload**.

### Method 2: PlatformIO
```bash
cd esp32_cam_firmware
pio run -t upload
```

---

## 🧪 Testing the ESP32 Stream Locally on PC

Run the automated test runner and pass your ESP32's stream URL:

```bash
./scripts/run_local_test.sh http://<esp32-ip>:81/stream
```

The system will ingest the ESP32 video, run YOLOv8 object detection, execute the kinematic fall detection engine, evaluate virtual tripwires/intrusion zones, and record verified MP4 clips in `storage/clips/`.
