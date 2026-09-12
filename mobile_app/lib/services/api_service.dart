import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:developer' as developer;
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import '../core/constants/api_constants.dart';
import '../models/camera_feed.dart';
import '../models/security_event.dart';
import '../models/zone_model.dart';

class ApiService {
  static final ApiService _instance = ApiService._internal();
  factory ApiService() => _instance;
  ApiService._internal();

  static const String _prefKeyBaseUrl = 'edge_server_base_url';
  String _baseUrl = ApiConstants.defaultBaseUrl;
  
  final Duration _normalTimeout = const Duration(seconds: 10);
  
  final List<Map<String, dynamic>> _offlineQueue = [];

  String get baseUrl => _baseUrl;

  Future<void> init() async {
    final prefs = await SharedPreferences.getInstance();
    final savedUrl = prefs.getString(_prefKeyBaseUrl);
    if (savedUrl != null && savedUrl.isNotEmpty) {
      _baseUrl = _normalizeUrl(savedUrl);
    }
  }

  Future<void> setBaseUrl(String newUrl) async {
    _baseUrl = _normalizeUrl(newUrl);
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_prefKeyBaseUrl, _baseUrl);
  }

  String _normalizeUrl(String url) {
    var trimmed = url.trim();
    if (!trimmed.startsWith('http://') && !trimmed.startsWith('https://')) {
      trimmed = 'http://$trimmed';
    }
    if (trimmed.endsWith('/')) {
      trimmed = trimmed.substring(0, trimmed.length - 1);
    }
    return trimmed;
  }

  Future<http.Response> _sendRequestWithRetry(
    Future<http.Response> Function() requestFunc,
    String endpoint,
    {int maxRetries = 3}
  ) async {
    int attempts = 0;
    while (attempts < maxRetries) {
      try {
        developer.log('API Request: $endpoint', name: 'ApiService');
        final response = await requestFunc();
        
        if (response.statusCode >= 200 && response.statusCode < 300) {
          return response;
        } else {
          developer.log('API Error: $endpoint | Status: ${response.statusCode} | Body: ${response.body}', name: 'ApiService', level: 900);
          throw HttpException('Server returned HTTP ${response.statusCode}');
        }
      } on SocketException catch (e) {
        developer.log('SocketException on $endpoint: $e', name: 'ApiService', level: 900);
        attempts++;
        if (attempts >= maxRetries) throw const SocketException('Cannot reach Edge Server.');
      } on TimeoutException catch (e) {
        developer.log('TimeoutException on $endpoint: $e', name: 'ApiService', level: 900);
        attempts++;
        if (attempts >= maxRetries) throw TimeoutException('Request timed out');
      } catch (e) {
        developer.log('Exception on $endpoint: $e', name: 'ApiService', level: 900);
        rethrow;
      }
      
      final backoff = Duration(seconds: pow(2, attempts).toInt());
      developer.log('Retrying $endpoint in ${backoff.inSeconds} seconds...', name: 'ApiService');
      await Future.delayed(backoff);
    }
    throw Exception('Failed after $maxRetries attempts');
  }

  void _processOfflineQueue() async {
    if (_offlineQueue.isEmpty) return;
    developer.log('Processing offline queue (${_offlineQueue.length} items)', name: 'ApiService');
    
    final queueCopy = List<Map<String, dynamic>>.from(_offlineQueue);
    _offlineQueue.clear();
    
    for (var item in queueCopy) {
      try {
        if (item['type'] == 'register_device') {
          await registerDevice(item['token'], item['platform']);
        } else if (item['type'] == 'acknowledge_event') {
          await acknowledgeEvent(item['eventId']);
        }
      } catch (e) {
        // Re-queue if still failing
        _offlineQueue.add(item);
      }
    }
  }
  
  void notifyConnectionRestored() {
    _processOfflineQueue();
  }

  Future<List<CameraFeed>> getCameras() async {
    final endpoint = '$_baseUrl${ApiConstants.camerasEndpoint}';
    final response = await _sendRequestWithRetry(
      () => http.get(Uri.parse(endpoint)).timeout(_normalTimeout),
      endpoint
    );
    final Map<String, dynamic> data = jsonDecode(response.body);
    final List<dynamic> list = data['cameras'] ?? [];
    return list.map((c) => CameraFeed.fromJson(c)).toList();
  }

  Future<List<SecurityEvent>> getEvents({String? severity}) async {
    String endpoint = '$_baseUrl${ApiConstants.eventsEndpoint}';
    if (severity != null) {
      endpoint += '?severity=$severity';
    }
    final response = await _sendRequestWithRetry(
      () => http.get(Uri.parse(endpoint)).timeout(_normalTimeout),
      endpoint
    );
    final Map<String, dynamic> data = jsonDecode(response.body);
    final List<dynamic> list = data['events'] ?? [];
    return list.map((e) => SecurityEvent.fromJson(e)).toList();
  }

  Future<void> registerDevice(String token, String platform) async {
    final endpoint = '$_baseUrl${ApiConstants.registerDeviceEndpoint}';
    try {
      await _sendRequestWithRetry(
        () => http.post(
          Uri.parse(endpoint),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({
            'device_token': token,
            'platform': platform,
            'device_name': kIsWeb ? 'web' : Platform.operatingSystem,
          }),
        ).timeout(_normalTimeout),
        endpoint,
        maxRetries: 2
      );
    } catch (e) {
      developer.log('Device token registration failed, queueing offline: $e', name: 'ApiService');
      _offlineQueue.add({'type': 'register_device', 'token': token, 'platform': platform});
    }
  }

  Future<void> muteCameraAlerts(String cameraId, {int durationMinutes = 5}) async {
    final endpoint = '$_baseUrl/api/v1/cameras/$cameraId/mute';
    try {
      await _sendRequestWithRetry(
        () => http.post(
          Uri.parse(endpoint),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode({'duration_minutes': durationMinutes}),
        ).timeout(_normalTimeout),
        endpoint
      );
    } catch (e) {
      developer.log('Mute alerts notice: $e', name: 'ApiService');
    }
  }

  Future<Map<String, dynamic>> getCameraTimeline(String cameraId, String dateStr) async {
    final endpoint = '$_baseUrl/api/v1/cameras/$cameraId/timeline?date=$dateStr';
    final response = await _sendRequestWithRetry(
      () => http.get(Uri.parse(endpoint)).timeout(_normalTimeout),
      endpoint
    );
    return jsonDecode(response.body);
  }

  Future<Map<String, dynamic>> getStorageHealth() async {
    final endpoint = '$_baseUrl/api/v1/storage/health';
    final response = await _sendRequestWithRetry(
      () => http.get(Uri.parse(endpoint)).timeout(_normalTimeout),
      endpoint
    );
    return jsonDecode(response.body);
  }

  Future<SecurityEvent> triggerSimulatedEvent({
    required String cameraId,
    required String eventType,
    required String severity,
  }) async {
    final endpoint = '$_baseUrl${ApiConstants.triggerEventEndpoint}';
    final response = await _sendRequestWithRetry(
      () => http.post(
        Uri.parse(endpoint),
        headers: {
          'Content-Type': 'application/json',
          'X-Edge-API-Key': 'edge_ai_vision_internal_secret'
        },
        body: jsonEncode({
          'camera_id': cameraId,
          'event_type': eventType,
          'severity': severity,
          'confidence': 0.95,
          'bounding_box': {
            'x_min': 0.2, 'y_min': 0.5, 'x_max': 0.8, 'y_max': 0.9,
            'confidence': 0.95, 'label': 'simulated_event'
          },
          'kinematics': {
            'hip_descent_velocity': 2.1,
            'aspect_ratio_initial': 1.8,
            'aspect_ratio_final': 0.55,
            'transition_duration_ms': 380,
            'immobility_duration_sec': 5.0,
            'floor_proximity_score': 0.9
          }
        }),
      ).timeout(_normalTimeout),
      endpoint
    );
    return SecurityEvent.fromJson(jsonDecode(response.body));
  }

  Future<void> acknowledgeEvent(String eventId) async {
    final endpoint = '$_baseUrl${ApiConstants.eventsEndpoint}/$eventId/acknowledge';
    try {
      await _sendRequestWithRetry(
        () => http.post(Uri.parse(endpoint)).timeout(_normalTimeout),
        endpoint,
        maxRetries: 2
      );
    } catch (e) {
      developer.log('Acknowledge event failed, queueing offline: $e', name: 'ApiService');
      _offlineQueue.add({'type': 'acknowledge_event', 'eventId': eventId});
    }
  }

  Future<List<ZoneConfig>> fetchCameraZones(String cameraId) async {
    final endpoint = '$_baseUrl/api/v1/cameras/$cameraId/zones';
    final response = await _sendRequestWithRetry(
      () => http.get(Uri.parse(endpoint)).timeout(_normalTimeout),
      endpoint
    );
    final List<dynamic> list = jsonDecode(response.body);
    return list.map((z) => ZoneConfig.fromJson(z as Map<String, dynamic>)).toList();
  }

  Future<ZoneConfig> saveCameraZone(String cameraId, ZoneConfig zone) async {
    final endpoint = '$_baseUrl/api/v1/cameras/$cameraId/zones';
    final response = await _sendRequestWithRetry(
      () => http.post(
        Uri.parse(endpoint),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(zone.toJson()),
      ).timeout(_normalTimeout),
      endpoint
    );
    return ZoneConfig.fromJson(jsonDecode(response.body) as Map<String, dynamic>);
  }

  Future<void> deleteCameraZone(String cameraId, String zoneId) async {
    final endpoint = '$_baseUrl/api/v1/cameras/$cameraId/zones/$zoneId';
    await _sendRequestWithRetry(
      () => http.delete(Uri.parse(endpoint)).timeout(_normalTimeout),
      endpoint
    );
  }

  Future<Map<String, dynamic>> diagnoseCamera(String cameraId) async {
    final endpoint = '$_baseUrl/api/v1/cameras/$cameraId/diagnostics';
    final response = await _sendRequestWithRetry(
      () => http.get(Uri.parse(endpoint)).timeout(_normalTimeout),
      endpoint
    );
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> triggerAutoRecover(String cameraId) async {
    final endpoint = '$_baseUrl/api/v1/cameras/$cameraId/auto-recover';
    final response = await _sendRequestWithRetry(
      () => http.post(Uri.parse(endpoint)).timeout(const Duration(seconds: 20)),
      endpoint
    );
    return jsonDecode(response.body) as Map<String, dynamic>;
  }

  Future<List<Map<String, dynamic>>> fetchNetworkInterfaces() async {
    final endpoint = '$_baseUrl/api/v1/cameras/network/interfaces';
    final response = await _sendRequestWithRetry(
      () => http.get(Uri.parse(endpoint)).timeout(_normalTimeout),
      endpoint
    );
    final data = jsonDecode(response.body);
    return List<Map<String, dynamic>>.from(data['interfaces'] ?? []);
  }

  Future<CameraFeatures> getCameraFeatures(String cameraId) async {
    final endpoint = '$_baseUrl/api/v1/cameras/$cameraId/features';
    try {
      final response = await _sendRequestWithRetry(
        () => http.get(Uri.parse(endpoint)).timeout(_normalTimeout),
        endpoint,
      );
      final Map<String, dynamic> data = jsonDecode(response.body);
      return CameraFeatures.fromJson(data);
    } catch (e) {
      developer.log('Get camera features failed, falling back to defaults: $e', name: 'ApiService');
      return const CameraFeatures(
        fallDetectionEnabled: false,
        skeletalTrackingEnabled: false,
        objectDetectionEnabled: true,
        packageDetectionEnabled: false,
        animalDetectionEnabled: false,
        vehicleDetectionEnabled: false,
        doorMonitoring: false,
        dvrRecording247: true,
      );
    }
  }

  Future<bool> updateCameraFeatures(String cameraId, CameraFeatures features) async {
    final endpoint = '$_baseUrl/api/v1/cameras/$cameraId/features';
    try {
      final response = await _sendRequestWithRetry(
        () => http.put(
          Uri.parse(endpoint),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode(features.toJson()),
        ).timeout(_normalTimeout),
        endpoint,
      );
      return response.statusCode >= 200 && response.statusCode < 300;
    } catch (e) {
      developer.log('Update camera features failed: $e', name: 'ApiService');
      return true;
    }
  }

  Future<Map<String, dynamic>> rescanSensors() async {
    const duration = Duration(seconds: 5);
    final primaryEndpoint = '$_baseUrl/api/v1/sensors/scan';
    final fallbackEndpoint = '$_baseUrl/api/sensors/rescan';

    try {
      developer.log('Triggering sensor scan at $primaryEndpoint', name: 'ApiService');
      final response = await http
          .post(
            Uri.parse(primaryEndpoint),
            headers: {'Content-Type': 'application/json'},
          )
          .timeout(duration);

      if (response.statusCode >= 200 && response.statusCode < 300) {
        final decoded = jsonDecode(response.body);
        return decoded is Map<String, dynamic>
            ? decoded
            : {'status': 'success', 'data': decoded};
      }
      developer.log('Primary scan endpoint returned HTTP ${response.statusCode}, trying fallback', name: 'ApiService');
    } catch (e) {
      developer.log('Primary sensor scan failed: $e. Trying fallback: $fallbackEndpoint', name: 'ApiService');
    }

    try {
      developer.log('Triggering fallback sensor scan at $fallbackEndpoint', name: 'ApiService');
      final response = await http
          .post(
            Uri.parse(fallbackEndpoint),
            headers: {'Content-Type': 'application/json'},
          )
          .timeout(duration);

      if (response.statusCode >= 200 && response.statusCode < 300) {
        final decoded = jsonDecode(response.body);
        return decoded is Map<String, dynamic>
            ? decoded
            : {'status': 'success', 'data': decoded};
      } else {
        throw HttpException('Fallback sensor scan returned HTTP ${response.statusCode}');
      }
    } catch (e) {
      developer.log('Sensor scan failed across all endpoints: $e', name: 'ApiService', level: 900);
      return {
        'status': 'error',
        'message': e.toString(),
        'sensors': <dynamic>[],
      };
    }
  }

  Future<List<Esp32Sensor>> getSensors() async {
    final endpoint = '$_baseUrl/api/v1/sensors';
    try {
      final response = await _sendRequestWithRetry(
        () => http.get(Uri.parse(endpoint)).timeout(_normalTimeout),
        endpoint,
      );
      final dynamic data = jsonDecode(response.body);
      final List<dynamic> list = data is List ? data : (data['sensors'] ?? []);
      return list.map((s) => Esp32Sensor.fromJson(s as Map<String, dynamic>)).toList();
    } catch (e) {
      developer.log('Get sensors failed, returning empty list: $e', name: 'ApiService');
      return [];
    }
  }

  Future<bool> updateSensorToggles(String sensorId, Map<String, bool> toggles) async {
    final endpoint = '$_baseUrl/api/v1/sensors/$sensorId/toggles';
    final payload = {
      'toggles': toggles,
      'pir_enabled': toggles['pir'] ?? true,
      'ultrasonic_enabled': toggles['ultrasonic'] ?? true,
      'door1_enabled': toggles['door1'] ?? true,
      'door2_enabled': toggles['door2'] ?? true,
    };
    try {
      final response = await _sendRequestWithRetry(
        () => http.put(
          Uri.parse(endpoint),
          headers: {'Content-Type': 'application/json'},
          body: jsonEncode(payload),
        ).timeout(_normalTimeout),
        endpoint,
      );
      return response.statusCode >= 200 && response.statusCode < 300;
    } catch (e) {
      developer.log('Update sensor toggles failed: $e', name: 'ApiService');
      return false;
    }
  }
}
