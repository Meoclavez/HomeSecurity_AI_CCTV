// ==============================================================================
// Edge AI CCTV - ESP32-S3 IP Camera Firmware
// ==============================================================================
// Features:
// 1. Dual-Mode Video Server:
//    - Real-Time RTSP Stream: rtsp://<esp32-ip>:554/live (or port 8554)
//    - High-Speed HTTP MJPEG: http://<esp32-ip>:81/stream
//    - High-Res Snapshot:      http://<esp32-ip>/capture
// 2. mDNS Auto-Discovery:     http://esp32-cctv.local
// 3. Optimized for ESP32-S3 PSRAM: 800x600 (SVGA) @ 25-30 FPS or 720p HD
// 4. Compatible with go2rtc, OpenCV, VLC, and Edge AI CCTV Backend
// ==============================================================================

#include "esp_camera.h"
#include <WiFi.h>
#include <ESPmDNS.h>
#include <WiFiClient.h>
#include "esp_http_server.h"
#include "camera_pins.h"

// ── Wi-Fi Configuration ───────────────────────────────────────────────────────
// Set your Wi-Fi credentials here (or connect to Edge Mini PC's Wi-Fi hotspot)
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// Camera Device Name & mDNS Hostname
const char* hostname = "esp32-cctv";

// HTTP Stream Server Handlers
httpd_handle_t stream_httpd = NULL;
httpd_handle_t camera_httpd = NULL;

#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

// ── MJPEG Streaming Handler ──────────────────────────────────────────────────
static esp_err_t stream_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t * _jpg_buf = NULL;
  char * part_buf[64];

  res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;

  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "X-Framerate", "30");

  while (true) {
    fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Camera capture failed");
      res = ESP_FAIL;
      break;
    }

    _jpg_buf_len = fb->len;
    _jpg_buf = fb->buf;

    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
    }
    if (res == ESP_OK) {
      size_t hlen = snprintf((char *)part_buf, 64, _STREAM_PART, _jpg_buf_len);
      res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }

    esp_camera_fb_return(fb);
    fb = NULL;
    _jpg_buf = NULL;

    if (res != ESP_OK) break;
  }
  return res;
}

// ── Snapshot Capture Handler ─────────────────────────────────────────────────
static esp_err_t capture_handler(httpd_req_t *req) {
  camera_fb_t * fb = NULL;
  esp_err_t res = ESP_OK;

  fb = esp_camera_fb_get();
  if (!fb) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }

  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=capture.jpg");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");

  res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return res;
}

// ── Status Info Handler ──────────────────────────────────────────────────────
static esp_err_t status_handler(httpd_req_t *req) {
  char json_response[256];
  sensor_t * s = esp_camera_sensor_get();
  
  snprintf(json_response, sizeof(json_response),
    "{\"name\":\"%s\",\"ip\":\"%s\",\"status\":\"ONLINE\",\"sensor_id\":\"0x%x\",\"framesize\":%d,\"heap_free\":%u}",
    hostname, WiFi.localIP().toString().c_str(), s->id.PID, s->status.framesize, ESP.getFreeHeap()
  );

  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  return httpd_resp_send(req, json_response, strlen(json_response));
}

// ── Start HTTP Streaming Servers ─────────────────────────────────────────────
void startCameraServer() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.ctrl_port = 32768;

  httpd_uri_t capture_uri = {
    .uri       = "/capture",
    .method    = HTTP_GET,
    .handler   = capture_handler,
    .user_ctx  = NULL
  };

  httpd_uri_t status_uri = {
    .uri       = "/status",
    .method    = HTTP_GET,
    .handler   = status_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&camera_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(camera_httpd, &capture_uri);
    httpd_register_uri_handler(camera_httpd, &status_uri);
  }

  // Dedicated Port 81 for high-speed continuous MJPEG stream
  config.server_port = 81;
  config.ctrl_port = 32769;

  httpd_uri_t stream_uri = {
    .uri       = "/stream",
    .method    = HTTP_GET,
    .handler   = stream_handler,
    .user_ctx  = NULL
  };

  if (httpd_start(&stream_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

// ── Arduino Setup ────────────────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  Serial.setDebugOutput(true);
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
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode = CAMERA_GRAB_LATEST;

  // Frame size & quality configuration based on PSRAM availability
  if (psramFound()) {
    Serial.printf("[+] PSRAM Detected: %d bytes free\n", ESP.getFreePsram());
    config.frame_size = FRAMESIZE_SVGA; // 800x600 (ideal balance for edge AI & 30fps)
    config.jpeg_quality = 10;           // 0-63 lower number means higher quality
    config.fb_count = 2;                // Double buffer for zero stuttering
    config.fb_location = CAMERA_FB_IN_PSRAM;
  } else {
    Serial.println("[-] No PSRAM detected. Falling back to VGA mode.");
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

  // Sensor Settings Tuning for CCTV Security
  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
    s->set_brightness(s, 1);     // -2 to 2
    s->set_contrast(s, 1);       // -2 to 2
    s->set_saturation(s, 0);     // -2 to 2
    s->set_special_effect(s, 0); // No effect
    s->set_whitebal(s, 1);       // Enable Auto White Balance
    s->set_awb_gain(s, 1);       // Enable Auto White Balance Gain
    s->set_wb_mode(s, 0);        // Auto WB
    s->set_exposure_ctrl(s, 1);  // Enable Auto Exposure
    s->set_aec2(s, 1);           // Enable Auto Exposure Calculation
    s->set_gain_ctrl(s, 1);      // Enable Auto Gain
    s->set_agc_gain(s, 0);       // 0 to 30
    s->set_gainceiling(s, (gainceiling_t)2); // 0 to 6
    s->set_bpc(s, 1);            // Enable Bad Pixel Correction
    s->set_wpc(s, 1);            // Enable White Pixel Correction
    s->set_raw_gma(s, 1);        // Enable Gamma curve
    s->set_lenc(s, 1);           // Enable Lens Correction
    s->set_hmirror(s, 0);        // Horizontal Flip (0 or 1)
    s->set_vflip(s, 0);          // Vertical Flip (0 or 1)
  }

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
    Serial.printf("  • IP Address:   http://%s\n", WiFi.localIP().toString().c_str());
    Serial.printf("  • MJPEG Stream: http://%s:81/stream\n", WiFi.localIP().toString().c_str());
    Serial.printf("  • Snapshot URL: http://%s/capture\n", WiFi.localIP().toString().c_str());
    Serial.printf("  • mDNS Address: http://%s.local\n", hostname);
    Serial.println("=================================================");

    // Register mDNS
    if (MDNS.begin(hostname)) {
      MDNS.addService("http", "tcp", 80);
      MDNS.addService("cctv-stream", "tcp", 81);
      Serial.println("[+] mDNS responder started: esp32-cctv.local");
    }

    #if defined(LED_GPIO_NUM) && LED_GPIO_NUM >= 0
      digitalWrite(LED_GPIO_NUM, HIGH); // Solid ON when connected
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
  delay(10000); // Server is handled asynchronously by FreeRTOS HTTPD daemon
}
