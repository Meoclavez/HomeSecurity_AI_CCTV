import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import '../core/constants/api_constants.dart';
import '../models/camera_feed.dart';
import '../models/security_event.dart';

class ApiService {
  static final ApiService _instance = ApiService._internal();
  factory ApiService() => _instance;
  ApiService._internal();

  static const String _prefKeyBaseUrl = 'edge_server_base_url';
  String _baseUrl = ApiConstants.defaultBaseUrl;
  final Duration _timeout = const Duration(seconds: 6);

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

  Future<List<CameraFeed>> getCameras() async {
    try {
      final response = await http
          .get(Uri.parse('$_baseUrl${ApiConstants.camerasEndpoint}'))
          .timeout(_timeout);

      if (response.statusCode == 200) {
        final Map<String, dynamic> data = jsonDecode(response.body);
        final List<dynamic> list = data['cameras'] ?? [];
        return list.map((c) => CameraFeed.fromJson(c)).toList();
      } else {
        throw HttpException('Server returned HTTP ${response.statusCode}');
      }
    } on SocketException {
      throw const SocketException('Cannot reach Edge Mini PC. Verify local Wi-Fi or VPN connection.');
    } on TimeoutException {
      throw TimeoutException('Connection timed out while reaching $_baseUrl');
    }
  }

  Future<List<SecurityEvent>> getEvents({String? severity}) async {
    String url = '$_baseUrl${ApiConstants.eventsEndpoint}';
    if (severity != null) {
      url += '?severity=$severity';
    }
    try {
      final response = await http.get(Uri.parse(url)).timeout(_timeout);
      if (response.statusCode == 200) {
        final Map<String, dynamic> data = jsonDecode(response.body);
        final List<dynamic> list = data['events'] ?? [];
        return list.map((e) => SecurityEvent.fromJson(e)).toList();
      } else {
        throw HttpException('Server returned HTTP ${response.statusCode}');
      }
    } on SocketException {
      throw const SocketException('Cannot reach Edge Server.');
    } on TimeoutException {
      throw TimeoutException('Request timed out');
    }
  }

  Future<void> registerDevice(String token, String platform) async {
    try {
      await http.post(
        Uri.parse('$_baseUrl${ApiConstants.registerDeviceEndpoint}'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'device_token': token,
          'platform': platform,
          'device_name': Platform.operatingSystem,
        }),
      ).timeout(const Duration(seconds: 4));
    } catch (e) {
      debugPrint('Device token registration notice: $e');
    }
  }

  Future<void> muteCameraAlerts(String cameraId, {int durationMinutes = 5}) async {
    try {
      await http.post(
        Uri.parse('$_baseUrl/api/v1/cameras/$cameraId/mute'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'duration_minutes': durationMinutes}),
      ).timeout(_timeout);
    } catch (e) {
      debugPrint('Mute alerts notice: $e');
    }
  }

  Future<Map<String, dynamic>> getCameraTimeline(String cameraId, String dateStr) async {
    final response = await http
        .get(Uri.parse('$_baseUrl/api/v1/cameras/$cameraId/timeline?date=$dateStr'))
        .timeout(_timeout);

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to load timeline: ${response.body}');
    }
  }

  Future<Map<String, dynamic>> getStorageHealth() async {
    final response = await http
        .get(Uri.parse('$_baseUrl/api/v1/storage/health'))
        .timeout(_timeout);

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception('Failed to load storage health: ${response.body}');
    }
  }

  Future<SecurityEvent> triggerSimulatedEvent({
    required String cameraId,
    required String eventType,
    required String severity,
  }) async {
    final response = await http.post(
      Uri.parse('$_baseUrl${ApiConstants.triggerEventEndpoint}'),
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
    ).timeout(_timeout);

    if (response.statusCode == 200) {
      return SecurityEvent.fromJson(jsonDecode(response.body));
    } else {
      throw Exception('Failed to trigger event: ${response.body}');
    }
  }

  Future<void> acknowledgeEvent(String eventId) async {
    try {
      await http.post(Uri.parse('$_baseUrl${ApiConstants.eventsEndpoint}/$eventId/acknowledge')).timeout(_timeout);
    } catch (_) {}
  }
}
