// ==============================================================================
// Edge AI CCTV - ESP32-S3 High-Speed IP Camera Firmware (DMA-Optimized)
// ==============================================================================
// Features:
// 1. Dual-Port Video Streaming:
//    - Port 81 High-Speed MJPEG: http://<esp32-ip>:81/stream
//    - Port 80 Web Portal & Stream: http://<esp32-ip>/stream & http://<esp32-ip>/
//    - Single-Frame Snapshot:    http://<esp32-ip>/capture
//    - JSON Status Telemetry:    http://<esp32-ip>/status
// 2. Anti-Overflow DMA Engine:
//    - 16 MHz XCLK to stabilize PSRAM DMA bursts and eliminate FB-OVF
//    - CAMERA_GRAB_LATEST mode with cooperative task yielding (vTaskDelay)
//    - Double-buffered PSRAM with optimized single-pass chunk header transmission
// 3. mDNS Auto-Discovery: http://esp32-cctv.local
// 4. Compatible with Edge AI CCTV Core, OpenCV, go2rtc, VLC, and browsers.
// ==============================================================================

#include "esp_camera.h"
#include <WiFi.h>
#include <ESPmDNS.h>
#include <WiFiClient.h>
#include <HTTPClient.h>
#include "esp_http_server.h"
#include "camera_pins.h"

// ── Wi-Fi Configuration ───────────────────────────────────────────────────────
// Set your Wi-Fi credentials here (or connect to Edge Mini PC's Wi-Fi hotspot)
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Camera Device Name & mDNS Hostname
const char* hostname = "esp32-cctv";

// ── Edge AI Backend & Hardware Configuration ──────────────────────────────────
String camera_id = "cam_living_room";
String edge_backend_ip = "192.168.1.100";
int edge_backend_port = 8000;
String edge_api_key = "edge_ai_vision_internal_secret";
float us_distance_threshold_cm = 50.0f;

// ── Sensor Feature Flags (Runtime Configurable) ────────────────────────────────
volatile bool pir_enabled = true;
volatile bool ultrasonic_enabled = true;
volatile bool door1_enabled = true;
volatile bool door2_enabled = true;

// ── Sensor Telemetry Cache (Read by /sensors) ──────────────────────────────────
volatile bool current_pir_motion = false;
volatile float current_distance_cm = -1.0f;
volatile bool current_door1_open = false;
volatile bool current_door2_open = false;

// Task Handle for Core 0 Sensor Poller
TaskHandle_t sensorTaskHandle = NULL;

// Alert Cooldown and Debounce Timers
const unsigned long ALERT_COOLDOWN_MS = 5000;
unsigned long last_alert_time = 0;

// HTTP Stream Server Handlers
httpd_handle_t stream_httpd = NULL;
httpd_handle_t camera_httpd = NULL;

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;

// ── MJPEG Streaming Handler (Anti-Overflow & DMA-Paced) ────────────────────────
static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  char part_buf[128];

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;

  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "X-Framerate", "30");
  httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");
  httpd_resp_set_hdr(req, "Pragma", "no-cache");

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("[-] Camera capture failed, retrying...");
      vTaskDelay(pdMS_TO_TICKS(10));
      continue;
    }

    // Consolidated single boundary + header chunk to minimize TCP context switches
    size_t hlen = snprintf(part_buf, sizeof(part_buf),
      "\r\n--" PART_BOUNDARY "\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n",
      fb->len);
    
    res = httpd_resp_send_chunk(req, part_buf, hlen);
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    }

    // Return frame buffer immediately so DMA engine can reuse it without overflowing
    esp_camera_fb_return(fb);
    fb = NULL;

    if (res != ESP_OK) {
      // Client disconnected
      break;
    }

    // Cooperative yield to allow FreeRTOS Wi-Fi and DMA tasks to run without stalling
    vTaskDelay(pdMS_TO_TICKS(2));
  }
  return res;
}

// ── Snapshot Capture Handler ─────────────────────────────────────────────────
static esp_err_t capture_handler(httpd_req_t *req) {
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return res;
}

// ── Status Telemetry Handler ─────────────────────────────────────────────────
static esp_err_t status_handler(httpd_req_t *req) {
  char json_buf[320];
  snprintf(json_buf, sizeof(json_buf),
    "{\"status\":\"online\",\"device\":\"ESP32-S3-CAM\",\"ip\":\"%s\","
    "\"free_heap\":%u,\"free_psram\":%u,\"rssi\":%d,\"mjpeg_port\":81,"
    "\"stream_url\":\"http://%s:81/stream\",\"snapshot_url\":\"http://%s/capture\"}",
    WiFi.localIP().toString().c_str(),
    ESP.getFreeHeap(),
    ESP.getFreePsram(),
    WiFi.RSSI(),
    WiFi.localIP().toString().c_str(),
    WiFi.localIP().toString().c_str()
  );

  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, json_buf, strlen(json_buf));
}

// ── Web Portal Landing Page Handler ──────────────────────────────────────────
static esp_err_t index_handler(httpd_req_t *req) {
  static const char index_html[] =
    "<!DOCTYPE html><html><head><meta charset='utf-8'><title>ESP32-S3 Edge CCTV Camera</title>"
    "<meta name='viewport' content='width=device-width, initial-scale=1'>"
    "<style>"
    "body{margin:0;background:#0d1117;color:#e6edf3;font-family:system-ui,-apple-system,sans-serif;display:flex;flex-direction:column;align-items:center;padding:20px;}"
    "h1{color:#58a6ff;margin-bottom:8px;font-size:24px;}"
    ".badge{background:#238636;color:#fff;padding:3px 8px;border-radius:12px;font-size:12px;font-weight:bold;margin-left:8px;}"
    ".card{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:16px;max-width:840px;width:100%;box-shadow:0 8px 24px rgba(0,0,0,0.5);margin-top:16px;}"
    "img{width:100%;border-radius:8px;background:#010409;border:1px solid #30363d;}"
    ".links{display:flex;gap:12px;margin-top:16px;flex-wrap:wrap;}"
    "a{background:#21262d;color:#58a6ff;padding:8px 16px;border-radius:6px;text-decoration:none;border:1px solid #30363d;font-size:14px;font-weight:600;}"
    "a:hover{background:#30363d;border-color:#8b949e;}"
    "</style></head><body>"
    "<h1>🛡️ ESP32-S3 CCTV Camera <span class='badge'>ONLINE</span></h1>"
    "<div class='card'>"
    "<img src='/stream' alt='Live Video Stream' />"
    "<div class='links'>"
    "<a href='/stream' target='_blank'>🎥 Open Stream (Port 80)</a>"
    "<a href=':81/stream' target='_blank'>⚡ High-Speed Stream (Port 81)</a>"
    "<a href='/capture' target='_blank'>📸 Capture Snapshot</a>"
    "<a href='/status' target='_blank'>📊 Device Telemetry</a>"
    "<a href='/sensors' target='_blank'>📡 Sensor Telemetry</a>"
    "</div></div></body></html>";

  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, index_html, strlen(index_html));
}

// ── Lightweight Zero-Dependency JSON Helpers ──────────────────────────────────
static bool parseJsonBool(const char* json, const char* key, bool* outVal) {
  char searchKey[64];
  snprintf(searchKey, sizeof(searchKey), "\"%s\"", key);
  const char* p = strstr(json, searchKey);
  if (!p) return false;
  p += strlen(searchKey);
  while (*p == ' ' || *p == ':' || *p == '\t' || *p == '\r' || *p == '\n') p++;
  if (strncmp(p, "true", 4) == 0 || *p == '1') {
    *outVal = true;
    return true;
  } else if (strncmp(p, "false", 5) == 0 || *p == '0') {
    *outVal = false;
    return true;
  }
  return false;
}

static bool parseJsonFloat(const char* json, const char* key, float* outVal) {
  char searchKey[64];
  snprintf(searchKey, sizeof(searchKey), "\"%s\"", key);
  const char* p = strstr(json, searchKey);
  if (!p) return false;
  p += strlen(searchKey);
  while (*p == ' ' || *p == ':' || *p == '\t' || *p == '\r' || *p == '\n') p++;
  char* endptr = NULL;
  float val = strtof(p, &endptr);
  if (endptr != p) {
    *outVal = val;
    return true;
  }
  return false;
}

static bool parseJsonString(const char* json, const char* key, char* outVal, size_t maxLen) {
  char searchKey[64];
  snprintf(searchKey, sizeof(searchKey), "\"%s\"", key);
  const char* p = strstr(json, searchKey);
  if (!p) return false;
  p += strlen(searchKey);
  while (*p == ' ' || *p == ':' || *p == '\t' || *p == '\r' || *p == '\n') p++;
  if (*p == '\"') {
    p++;
    size_t i = 0;
    while (*p && *p != '\"' && i < maxLen - 1) {
      outVal[i++] = *p++;
    }
    outVal[i] = '\0';
    return true;
  }
  return false;
}

// ── Ultrasonic Pulse Measurement ──────────────────────────────────────────────
static float readUltrasonicDistance() {
  #if defined(PIN_US_TRIG) && defined(PIN_US_ECHO) && PIN_US_TRIG >= 0 && PIN_US_ECHO >= 0
    digitalWrite(PIN_US_TRIG, LOW);
    delayMicroseconds(2);
    digitalWrite(PIN_US_TRIG, HIGH);
    delayMicroseconds(10);
    digitalWrite(PIN_US_TRIG, LOW);

    unsigned long duration = pulseIn(PIN_US_ECHO, HIGH, 30000); // 30ms timeout (~5.1m)
    if (duration == 0) {
      return -1.0f; // Timeout or out of range
    }
    return (float)duration * 0.0343f / 2.0f;
  #else
    return -1.0f;
  #endif
}

// ── Alert Webhook Dispatcher ─────────────────────────────────────────────────
static void dispatchAlertWebhook(const char* event_type, const char* severity, float confidence, const char* sensor_name, const char* extra_json) {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }

  WiFiClient client;
  HTTPClient http;
  String url = "http://" + edge_backend_ip + ":" + String(edge_backend_port) + "/api/v1/events/trigger";

  if (http.begin(client, url)) {
    http.addHeader("Content-Type", "application/json");
    http.addHeader("X-Edge-API-Key", edge_api_key);
    http.setTimeout(2000); // 2000ms max timeout to avoid stalling sensor polling

    char payload[384];
    snprintf(payload, sizeof(payload),
      "{\"camera_id\":\"%s\",\"event_type\":\"%s\",\"severity\":\"%s\",\"confidence\":%.2f,"
      "\"metadata\":{\"source\":\"esp32_iot_sensor\",\"sensor\":\"%s\",%s}}",
      camera_id.c_str(),
      event_type,
      severity,
      confidence,
      sensor_name,
      extra_json
    );

    int httpCode = http.POST((uint8_t*)payload, strlen(payload));
    if (httpCode > 0) {
      Serial.printf("[+] Webhook alert dispatched [%s -> %s], response: %d\n", sensor_name, event_type, httpCode);
    } else {
      Serial.printf("[-] Webhook alert failed [%s]: %s\n", sensor_name, http.errorToString(httpCode).c_str());
    }
    http.end();
  }
}

// ── Dedicated Core 0 Sensor Task (FreeRTOS) ──────────────────────────────────
static void sensorTask(void *pvParameters) {
  Serial.printf("[+] FreeRTOS sensorTask running on Core %d\n", xPortGetCoreID());

  // Configure sensor GPIOs
  #if defined(PIN_PIR) && PIN_PIR >= 0
    pinMode(PIN_PIR, INPUT);
  #endif

  #if defined(PIN_US_TRIG) && PIN_US_TRIG >= 0
    pinMode(PIN_US_TRIG, OUTPUT);
    digitalWrite(PIN_US_TRIG, LOW);
  #endif

  #if defined(PIN_US_ECHO) && PIN_US_ECHO >= 0
    pinMode(PIN_US_ECHO, INPUT);
  #endif

  #if defined(PIN_DOOR1) && PIN_DOOR1 >= 0
    pinMode(PIN_DOOR1, INPUT_PULLUP);
  #endif

  #if defined(PIN_DOOR2) && PIN_DOOR2 >= 0
    pinMode(PIN_DOOR2, INPUT_PULLUP);
  #endif

  int door1_debounce = 0;
  int door2_debounce = 0;
  int us_debounce = 0;

  while (true) {
    // 1. Digital read for PIR motion sensor
    if (pir_enabled) {
      #if defined(PIN_PIR) && PIN_PIR >= 0
        current_pir_motion = (digitalRead(PIN_PIR) == HIGH);
      #else
        current_pir_motion = false;
      #endif
    } else {
      current_pir_motion = false;
    }

    // 2. Ultrasonic pulse measurement (HC-SR04 pulseIn with 30ms timeout)
    if (ultrasonic_enabled) {
      current_distance_cm = readUltrasonicDistance();
    } else {
      current_distance_cm = -1.0f;
    }

    // 3. Door 1 reed switch (INPUT_PULLUP: LOW = Closed/Connected to GND, HIGH = Open)
    if (door1_enabled) {
      #if defined(PIN_DOOR1) && PIN_DOOR1 >= 0
        bool raw_d1 = (digitalRead(PIN_DOOR1) == HIGH);
        if (raw_d1 != current_door1_open) {
          door1_debounce++;
          if (door1_debounce >= 2) { // 2 cycles * 200ms = 400ms debounce
            current_door1_open = raw_d1;
            door1_debounce = 0;
          }
        } else {
          door1_debounce = 0;
        }
      #else
        current_door1_open = false;
      #endif
    } else {
      current_door1_open = false;
    }

    // 4. Door 2 reed switch (INPUT_PULLUP: LOW = Closed/Connected to GND, HIGH = Open)
    if (door2_enabled) {
      #if defined(PIN_DOOR2) && PIN_DOOR2 >= 0
        bool raw_d2 = (digitalRead(PIN_DOOR2) == HIGH);
        if (raw_d2 != current_door2_open) {
          door2_debounce++;
          if (door2_debounce >= 2) {
            current_door2_open = raw_d2;
            door2_debounce = 0;
          }
        } else {
          door2_debounce = 0;
        }
      #else
        current_door2_open = false;
      #endif
    } else {
      current_door2_open = false;
    }

    // 5. Ultrasonic proximity debounce
    bool us_triggered = false;
    if (ultrasonic_enabled && current_distance_cm > 0.0f && current_distance_cm < us_distance_threshold_cm) {
      us_debounce++;
      if (us_debounce >= 2) { // 2 consecutive readings < threshold
        us_triggered = true;
      }
    } else {
      us_debounce = 0;
    }

    // 6. Alert Webhook Dispatcher with 5-Second Cooldown & Debounce Enforcement
    unsigned long now = millis();
    if (now - last_alert_time >= ALERT_COOLDOWN_MS) {
      // Priority 1: Door Opened (or remains open)
      if ((door1_enabled && current_door1_open) || (door2_enabled && current_door2_open)) {
        char extra[96];
        snprintf(extra, sizeof(extra), "\"door1_open\":%s,\"door2_open\":%s",
          current_door1_open ? "true" : "false",
          current_door2_open ? "true" : "false"
        );
        dispatchAlertWebhook("DOOR_LEFT_OPEN", "HIGH", 1.0f, "REED_SWITCH", extra);
        last_alert_time = now;
      }
      // Priority 2: PIR Motion Detected
      else if (pir_enabled && current_pir_motion) {
        char extra[48];
        snprintf(extra, sizeof(extra), "\"pir_motion\":true");
        dispatchAlertWebhook("INTRUSION_DETECTED", "WARNING", 0.95f, "PIR", extra);
        last_alert_time = now;
      }
      // Priority 3: Ultrasonic Distance < Threshold
      else if (us_triggered) {
        char extra[96];
        snprintf(extra, sizeof(extra), "\"distance_cm\":%.1f,\"threshold_cm\":%.1f",
          current_distance_cm, us_distance_threshold_cm
        );
        dispatchAlertWebhook("PERIMETER_BREACH", "WARNING", 0.90f, "ULTRASONIC", extra);
        last_alert_time = now;
      }
    }

    // Yield Core 0 for 200ms without blocking Core 1's MJPEG streaming loop
    vTaskDelay(pdMS_TO_TICKS(200));
  }
}

// ── Sensors Telemetry Handler (GET /sensors) ─────────────────────────────────
static esp_err_t sensors_handler(httpd_req_t *req) {
  char json_buf[384];
  snprintf(json_buf, sizeof(json_buf),
    "{\"camera_id\":\"%s\",\"pir_motion\":%s,\"distance_cm\":%.1f,\"door1_open\":%s,\"door2_open\":%s,"
    "\"toggles\":{\"pir\":%s,\"ultrasonic\":%s,\"door1\":%s,\"door2\":%s}}",
    camera_id.c_str(),
    current_pir_motion ? "true" : "false",
    current_distance_cm,
    current_door1_open ? "true" : "false",
    current_door2_open ? "true" : "false",
    pir_enabled ? "true" : "false",
    ultrasonic_enabled ? "true" : "false",
    door1_enabled ? "true" : "false",
    door2_enabled ? "true" : "false"
  );

  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, json_buf, strlen(json_buf));
}

// ── Sensors Configuration Handler (POST /sensors/config) ─────────────────────
static esp_err_t sensors_config_handler(httpd_req_t *req) {
  char buf[512];
  int total_len = req->content_len;
  if (total_len >= sizeof(buf)) {
    httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Payload too large");
    return ESP_FAIL;
  }

  int cur_len = 0;
  int received = 0;
  while (cur_len < total_len) {
    received = httpd_req_recv(req, buf + cur_len, total_len - cur_len);
    if (received <= 0) {
      if (received == HTTPD_SOCK_ERR_TIMEOUT) {
        httpd_resp_send_408(req);
      }
      return ESP_FAIL;
    }
    cur_len += received;
  }
  buf[cur_len] = '\0';

  bool val;
  if (parseJsonBool(buf, "pir", &val) || parseJsonBool(buf, "pir_enabled", &val)) {
    pir_enabled = val;
  }
  if (parseJsonBool(buf, "ultrasonic", &val) || parseJsonBool(buf, "ultrasonic_enabled", &val)) {
    ultrasonic_enabled = val;
  }
  if (parseJsonBool(buf, "door1", &val) || parseJsonBool(buf, "door1_enabled", &val)) {
    door1_enabled = val;
  }
  if (parseJsonBool(buf, "door2", &val) || parseJsonBool(buf, "door2_enabled", &val)) {
    door2_enabled = val;
  }

  float fval;
  if (parseJsonFloat(buf, "threshold", &fval) || parseJsonFloat(buf, "threshold_cm", &fval) || parseJsonFloat(buf, "distance_threshold_cm", &fval)) {
    if (fval > 0.0f) us_distance_threshold_cm = fval;
  }

  char strVal[64];
  if (parseJsonString(buf, "camera_id", strVal, sizeof(strVal))) {
    camera_id = String(strVal);
  }
  if (parseJsonString(buf, "edge_backend_ip", strVal, sizeof(strVal))) {
    edge_backend_ip = String(strVal);
  }
  if (parseJsonString(buf, "edge_api_key", strVal, sizeof(strVal))) {
    edge_api_key = String(strVal);
  }

  char resp[384];
  snprintf(resp, sizeof(resp),
    "{\"status\":\"success\",\"camera_id\":\"%s\",\"distance_threshold_cm\":%.1f,"
    "\"toggles\":{\"pir\":%s,\"ultrasonic\":%s,\"door1\":%s,\"door2\":%s}}",
    camera_id.c_str(),
    us_distance_threshold_cm,
    pir_enabled ? "true" : "false",
    ultrasonic_enabled ? "true" : "false",
    door1_enabled ? "true" : "false",
    door2_enabled ? "true" : "false"
  );

  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, resp, strlen(resp));
}

// ── CORS Options Preflight Handler ───────────────────────────────────────────
static esp_err_t cors_options_handler(httpd_req_t *req) {
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Headers", "Content-Type, X-Edge-API-Key");
  httpd_resp_send(req, NULL, 0);
  return ESP_OK;
}

// ── HTTP Server Initializer ──────────────────────────────────────────────────
void startCameraServer() {
  // Main Web, Snapshot & Sensor Server on Port 80
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.ctrl_port = 32768;
  config.max_uri_handlers = 12;
  config.lru_purge_enable = true;
  config.send_wait_timeout = 2;
  config.recv_wait_timeout = 2;

  httpd_uri_t index_uri           = { .uri = "/",               .method = HTTP_GET,     .handler = index_handler,          .user_ctx = NULL };
  httpd_uri_t stream80_uri        = { .uri = "/stream",         .method = HTTP_GET,     .handler = stream_handler,         .user_ctx = NULL };
  httpd_uri_t capture_uri         = { .uri = "/capture",        .method = HTTP_GET,     .handler = capture_handler,        .user_ctx = NULL };
  httpd_uri_t status_uri          = { .uri = "/status",         .method = HTTP_GET,     .handler = status_handler,         .user_ctx = NULL };
  httpd_uri_t sensors_get_uri     = { .uri = "/sensors",        .method = HTTP_GET,     .handler = sensors_handler,        .user_ctx = NULL };
  httpd_uri_t sensors_opt_uri     = { .uri = "/sensors",        .method = HTTP_OPTIONS, .handler = cors_options_handler,   .user_ctx = NULL };
  httpd_uri_t sensors_cfg_uri     = { .uri = "/sensors/config", .method = HTTP_POST,    .handler = sensors_config_handler, .user_ctx = NULL };
  httpd_uri_t sensors_cfg_opt_uri = { .uri = "/sensors/config", .method = HTTP_OPTIONS, .handler = cors_options_handler,   .user_ctx = NULL };

  if (httpd_start(&camera_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &index_uri);
    httpd_register_uri_handler(camera_httpd, &stream80_uri);
    httpd_register_uri_handler(camera_httpd, &capture_uri);
    httpd_register_uri_handler(camera_httpd, &status_uri);
    httpd_register_uri_handler(camera_httpd, &sensors_get_uri);
    httpd_register_uri_handler(camera_httpd, &sensors_opt_uri);
    httpd_register_uri_handler(camera_httpd, &sensors_cfg_uri);
    httpd_register_uri_handler(camera_httpd, &sensors_cfg_opt_uri);
  }

  // Dedicated High-Speed Stream Server on Port 81
  config.server_port = 81;
  config.ctrl_port = 32769;

  httpd_uri_t stream81_uri = { .uri = "/stream", .method = HTTP_GET, .handler = stream_handler, .user_ctx = NULL };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream81_uri);
  }
}

// ── Arduino Setup ────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(false); // Reduce UART flooding
  Serial.println();
  Serial.println("=================================================");
  Serial.println("   Edge AI CCTV - ESP32-S3 IP Camera Initializing ");
  Serial.println("=================================================");

  #if defined(LED_GPIO_NUM) && LED_GPIO_NUM >= 0
    pinMode(LED_GPIO_NUM, OUTPUT);
    digitalWrite(LED_GPIO_NUM, LOW);
  #endif

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  
  // 16 MHz XCLK frequency stabilizes PSRAM DMA bus timing and eliminates FB-OVF
  config.xclk_freq_hz = 16000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;

  // Frame size & buffer allocation based on PSRAM availability
  if (psramFound()) {
    Serial.printf("[+] PSRAM Detected: %d bytes free\n", ESP.getFreePsram());
    config.frame_size = FRAMESIZE_SVGA; // 800x600 (ideal for Edge AI kinematics)
    config.jpeg_quality = 12;           // 10-14 gives crisp detail with zero DMA overflow
    config.fb_count = 2;                // Double buffering
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    Serial.println("[-] No PSRAM detected. Falling back to DRAM VGA mode.");
    config.frame_size = FRAMESIZE_VGA;  // 640x480
    config.jpeg_quality = 14;
    config.fb_count = 1;
    config.fb_location = CAMERA_FB_IN_DRAM;
  }

  // Camera Sensor Initialization
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[-] Camera init failed with error 0x%x\n", err);
    return;
  }
  Serial.println("[+] Camera sensor initialized successfully.");

  // Sensor Image Tuning
  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_brightness(s, 1);     // -2 to 2
    s->set_contrast(s, 1);       // -2 to 2
    s->set_saturation(s, 0);     // -2 to 2
    s->set_whitebal(s, 1);       // Auto White Balance
    s->set_awb_gain(s, 1);       // Auto WB Gain
    s->set_wb_mode(s, 0);        // Auto Mode
    s->set_exposure_ctrl(s, 1);  // Auto Exposure
    s->set_aec2(s, 1);           // Auto Exposure Calc
    s->set_gain_ctrl(s, 1);      // Auto Gain
    s->set_agc_gain(s, 0);       // AGC
    s->set_gainceiling(s, (gainceiling_t)2);
    s->set_bpc(s, 1);            // Bad Pixel Correction
    s->set_wpc(s, 1);            // White Pixel Correction
    s->set_raw_gma(s, 1);        // Gamma Correction
    s->set_lenc(s, 1);           // Lens Correction
    s->set_hmirror(s, 0);        // Horizontal Mirror
    s->set_vflip(s, 0);          // Vertical Flip
  }

  // Spawn Dedicated FreeRTOS Sensor Polling Task on Core 0 (isolated from Core 1 video loop)
  xTaskCreatePinnedToCore(
    sensorTask,
    "sensorTask",
    8192,
    NULL,
    1,
    &sensorTaskHandle,
    0
  );
  Serial.println("[+] FreeRTOS sensorTask pinned to Core 0 created.");

  // Connect to Wi-Fi
  Serial.printf("[+] Connecting to Wi-Fi: %s", ssid);
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false); // Disable Wi-Fi sleep for lowest latency streaming
  WiFi.begin(ssid, password);

  int retry = 0;
  while (WiFi.status() != WL_CONNECTED && retry < 30) {
    delay(500);
    Serial.print(".");
    #if defined(LED_GPIO_NUM) && LED_GPIO_NUM >= 0
      digitalWrite(LED_GPIO_NUM, !digitalRead(LED_GPIO_NUM));
    #endif
    retry++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println();
    Serial.println("=================================================");
    Serial.println("  ✅ ESP32-S3 Camera Connected to Network! ");
    Serial.println("=================================================");
    Serial.printf("  • Web Portal:   http://%s\n", WiFi.localIP().toString().c_str());
    Serial.printf("  • MJPEG Stream: http://%s:81/stream\n", WiFi.localIP().toString().c_str());
    Serial.printf("  • Alt Stream:   http://%s/stream\n", WiFi.localIP().toString().c_str());
    Serial.printf("  • Snapshot URL: http://%s/capture\n", WiFi.localIP().toString().c_str());
    Serial.printf("  • Sensor Data:  http://%s/sensors\n", WiFi.localIP().toString().c_str());
    Serial.printf("  • mDNS Address: http://%s.local\n", hostname);
    Serial.println("=================================================");

    // Register mDNS
    if (MDNS.begin(hostname)) {
      MDNS.addService("http", "tcp", 80);
      MDNS.addService("cctv-stream", "tcp", 81);
      Serial.println("[+] mDNS responder started: esp32-cctv.local");
    }

    #if defined(LED_GPIO_NUM) && LED_GPIO_NUM >= 0
      digitalWrite(LED_GPIO_NUM, HIGH); // Solid ON
    #endif

    // Start Streaming Web Server
    startCameraServer();
    Serial.println("[+] Video stream server active and ready for Edge AI ingestion.");
  } else {
    Serial.println("\n[-] Wi-Fi connection timed out. Check SSID and password.");
  }
}

// ── Arduino Loop ─────────────────────────────────────────────────────────────
void loop() {
  vTaskDelay(pdMS_TO_TICKS(1000)); // FreeRTOS server daemon handles streaming
}
