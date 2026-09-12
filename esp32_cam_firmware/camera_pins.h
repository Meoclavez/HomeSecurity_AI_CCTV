// ==============================================================================
// Edge AI CCTV - ESP32-S3 Camera Pin Definitions & Board Profiles
// ==============================================================================

#pragma once

// Select your ESP32-S3 Camera Model:
#define CAMERA_MODEL_FREENOVE_ESP32S3_CAM  // Default (Very popular ESP32-S3 CAM)
// #define CAMERA_MODEL_XIAO_ESP32S3_SENSE
// #define CAMERA_MODEL_AI_THINKER_ESP32S3
// #define CAMERA_MODEL_ESP32S3_EYE
// #define CAMERA_MODEL_AI_THINKER_ESP32_CAM // Legacy ESP32 (Non-S3)

#if defined(CAMERA_MODEL_FREENOVE_ESP32S3_CAM)
  #define PWDN_GPIO_NUM     -1
  #define RESET_GPIO_NUM    -1
  #define XCLK_GPIO_NUM     15
  #define SIOD_GPIO_NUM     4
  #define SIOC_GPIO_NUM     5

  #define Y9_GPIO_NUM       16
  #define Y8_GPIO_NUM       17
  #define Y7_GPIO_NUM       18
  #define Y6_GPIO_NUM       12
  #define Y5_GPIO_NUM       10
  #define Y4_GPIO_NUM       8
  #define Y3_GPIO_NUM       9
  #define Y2_GPIO_NUM       11
  #define VSYNC_GPIO_NUM    6
  #define HREF_GPIO_NUM     7
  #define PCLK_GPIO_NUM     13
  #define LED_GPIO_NUM      2 // On-board status LED

  // IoT Hardware Sensor Pins (Freenove ESP32-S3 WROOM CAM)
  #define PIN_PIR           1   // PIR Motion Sensor (Digital Input)
  #define PIN_US_TRIG       14  // HC-SR04 Ultrasonic Trigger (Digital Output)
  #define PIN_US_ECHO       21  // HC-SR04 Ultrasonic Echo (Digital Input)
  #define PIN_DOOR1         47  // Door 1 Reed Switch (INPUT_PULLUP)
  #define PIN_DOOR2         3   // Door 2 Reed Switch (INPUT_PULLUP)

#elif defined(CAMERA_MODEL_XIAO_ESP32S3_SENSE)
  #define PWDN_GPIO_NUM     -1
  #define RESET_GPIO_NUM    -1
  #define XCLK_GPIO_NUM     10
  #define SIOD_GPIO_NUM     40
  #define SIOC_GPIO_NUM     39

  #define Y9_GPIO_NUM       48
  #define Y8_GPIO_NUM       11
  #define Y7_GPIO_NUM       12
  #define Y6_GPIO_NUM       14
  #define Y5_GPIO_NUM       16
  #define Y4_GPIO_NUM       18
  #define Y3_GPIO_NUM       17
  #define Y2_GPIO_NUM       15
  #define VSYNC_GPIO_NUM    38
  #define HREF_GPIO_NUM     47
  #define PCLK_GPIO_NUM     13
  #define LED_GPIO_NUM      21

  // IoT Hardware Sensor Pins (Seeed Studio XIAO ESP32S3 Sense)
  #define PIN_PIR           1   // PIR Motion Sensor (Digital Input)
  #define PIN_US_TRIG       2   // HC-SR04 Ultrasonic Trigger (Digital Output)
  #define PIN_US_ECHO       3   // HC-SR04 Ultrasonic Echo (Digital Input)
  #define PIN_DOOR1         4   // Door 1 Reed Switch (INPUT_PULLUP)
  #define PIN_DOOR2         5   // Door 2 Reed Switch (INPUT_PULLUP)

#elif defined(CAMERA_MODEL_AI_THINKER_ESP32S3)
  #define PWDN_GPIO_NUM     -1
  #define RESET_GPIO_NUM    -1
  #define XCLK_GPIO_NUM     39
  #define SIOD_GPIO_NUM     21
  #define SIOC_GPIO_NUM     46

  #define Y9_GPIO_NUM       40
  #define Y8_GPIO_NUM       38
  #define Y7_GPIO_NUM       37
  #define Y6_GPIO_NUM       36
  #define Y5_GPIO_NUM       35
  #define Y4_GPIO_NUM       34
  #define Y3_GPIO_NUM       33
  #define Y2_GPIO_NUM       47
  #define VSYNC_GPIO_NUM    48
  #define HREF_GPIO_NUM     45
  #define PCLK_GPIO_NUM     41
  #define LED_GPIO_NUM      2

  // IoT Hardware Sensor Pins (AI-Thinker ESP32-S3 CAM)
  #define PIN_PIR           4   // PIR Motion Sensor (Digital Input)
  #define PIN_US_TRIG       5   // HC-SR04 Ultrasonic Trigger (Digital Output)
  #define PIN_US_ECHO       6   // HC-SR04 Ultrasonic Echo (Digital Input)
  #define PIN_DOOR1         7   // Door 1 Reed Switch (INPUT_PULLUP)
  #define PIN_DOOR2         8   // Door 2 Reed Switch (INPUT_PULLUP)

#elif defined(CAMERA_MODEL_ESP32S3_EYE)
  #define PWDN_GPIO_NUM     -1
  #define RESET_GPIO_NUM    -1
  #define XCLK_GPIO_NUM     15
  #define SIOD_GPIO_NUM     4
  #define SIOC_GPIO_NUM     5

  #define Y9_GPIO_NUM       16
  #define Y8_GPIO_NUM       17
  #define Y7_GPIO_NUM       18
  #define Y6_GPIO_NUM       12
  #define Y5_GPIO_NUM       10
  #define Y4_GPIO_NUM       8
  #define Y3_GPIO_NUM       9
  #define Y2_GPIO_NUM       11
  #define VSYNC_GPIO_NUM    6
  #define HREF_GPIO_NUM     7
  #define PCLK_GPIO_NUM     13
  #define LED_GPIO_NUM      2

  // IoT Hardware Sensor Pins (ESP32-S3-EYE / Fallback)
  #define PIN_PIR           1   // PIR Motion Sensor (Digital Input)
  #define PIN_US_TRIG       14  // HC-SR04 Ultrasonic Trigger (Digital Output)
  #define PIN_US_ECHO       21  // HC-SR04 Ultrasonic Echo (Digital Input)
  #define PIN_DOOR1         47  // Door 1 Reed Switch (INPUT_PULLUP)
  #define PIN_DOOR2         3   // Door 2 Reed Switch (INPUT_PULLUP)

#elif defined(CAMERA_MODEL_AI_THINKER_ESP32_CAM)
  #define PWDN_GPIO_NUM     32
  #define RESET_GPIO_NUM    -1
  #define XCLK_GPIO_NUM      0
  #define SIOD_GPIO_NUM     26
  #define SIOC_GPIO_NUM     27

  #define Y9_GPIO_NUM       35
  #define Y8_GPIO_NUM       34
  #define Y7_GPIO_NUM       39
  #define Y6_GPIO_NUM       36
  #define Y5_GPIO_NUM       21
  #define Y4_GPIO_NUM       19
  #define Y3_GPIO_NUM       18
  #define Y2_GPIO_NUM        5
  #define VSYNC_GPIO_NUM    25
  #define HREF_GPIO_NUM     23
  #define PCLK_GPIO_NUM     22
  #define LED_GPIO_NUM      33 // Flashlight / status LED

  // IoT Hardware Sensor Pins (Legacy AI-Thinker ESP32-CAM Breakout)
  #define PIN_PIR           13  // PIR Motion Sensor (Digital Input)
  #define PIN_US_TRIG       14  // HC-SR04 Ultrasonic Trigger (Digital Output)
  #define PIN_US_ECHO       15  // HC-SR04 Ultrasonic Echo (Digital Input)
  #define PIN_DOOR1         12  // Door 1 Reed Switch (INPUT_PULLUP)
  #define PIN_DOOR2         16  // Door 2 Reed Switch (INPUT_PULLUP)
#endif

// Fallback Sensor Pins if custom profile has missing definitions
#ifndef PIN_PIR
  #define PIN_PIR           1
  #define PIN_US_TRIG       14
  #define PIN_US_ECHO       21
  #define PIN_DOOR1         47
  #define PIN_DOOR2         3
#endif
