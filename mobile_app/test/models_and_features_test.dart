import 'package:flutter_test/flutter_test.dart';
import 'package:edge_ai_cctv/models/camera_feed.dart';

void main() {
  group('CameraFeatures Model Tests', () {
    test('Default values are set properly', () {
      const features = CameraFeatures();
      expect(features.fallDetectionEnabled, false);
      expect(features.skeletalTrackingEnabled, false);
      expect(features.objectDetectionEnabled, true);
      expect(features.packageDetectionEnabled, false);
      expect(features.animalDetectionEnabled, false);
      expect(features.vehicleDetectionEnabled, false);
      expect(features.doorMonitoring, false);
      expect(features.dvrRecording247, true);
    });

    test('fromJson and toJson round-trip', () {
      final json = {
        'fall_detection_enabled': true,
        'skeletal_tracking_enabled': true,
        'object_detection_enabled': false,
        'package_detection_enabled': true,
        'animal_detection_enabled': false,
        'vehicle_detection_enabled': true,
        'door_monitoring': true,
        'dvr_recording_247': false,
      };

      final features = CameraFeatures.fromJson(json);
      expect(features.fallDetectionEnabled, true);
      expect(features.skeletalTrackingEnabled, true);
      expect(features.objectDetectionEnabled, false);
      expect(features.packageDetectionEnabled, true);
      expect(features.animalDetectionEnabled, false);
      expect(features.vehicleDetectionEnabled, true);
      expect(features.doorMonitoring, true);
      expect(features.dvrRecording247, false);

      final outJson = features.toJson();
      expect(outJson['fall_detection_enabled'], true);
      expect(outJson['skeletal_tracking_enabled'], true);
      expect(outJson['object_detection_enabled'], false);
      expect(outJson['dvr_recording_247'], false);
    });

    test('copyWith updates specific properties', () {
      const features = CameraFeatures();
      final updated = features.copyWith(
        fallDetectionEnabled: true,
        skeletalTrackingEnabled: true,
      );

      expect(updated.fallDetectionEnabled, true);
      expect(updated.skeletalTrackingEnabled, true);
      expect(updated.objectDetectionEnabled, true); // preserved
      expect(updated.dvrRecording247, true); // preserved
    });
  });

  group('CameraFeed Model Tests', () {
    test('CameraFeed attaches CameraFeatures', () {
      final json = {
        'id': 'cam_front',
        'name': 'Front Door',
        'location': 'Exterior',
        'rtsp_url': 'rtsp://test',
        'webrtc_url': 'http://test',
        'status': 'ONLINE',
        'fps': 30,
        'resolution': '1080p',
        'is_ai_enabled': true,
        'ai_models': ['yolov8', 'pose'],
        'features': {
          'fall_detection_enabled': true,
          'skeletal_tracking_enabled': false,
        },
      };

      final cam = CameraFeed.fromJson(json);
      expect(cam.id, 'cam_front');
      expect(cam.features, isNotNull);
      expect(cam.features!.fallDetectionEnabled, true);
      expect(cam.features!.skeletalTrackingEnabled, false);
    });
  });

  group('Esp32Sensor Model Tests', () {
    test('fromJson parses raw and backend schema', () {
      final json = {
        'id': 'esp32_01',
        'name': 'Porch Sentry',
        'ip_address': '192.168.1.150',
        'camera_id': 'cam_front',
        'pir_motion': true,
        'distance_cm': 42.5,
        'door1_open': true,
        'door2_open': false,
        'toggles': {
          'pir': true,
          'ultrasonic': true,
          'door1': false,
          'door2': true,
        },
        'last_heartbeat': '2026-09-12T12:00:00.000Z',
      };

      final sensor = Esp32Sensor.fromJson(json);
      expect(sensor.id, 'esp32_01');
      expect(sensor.name, 'Porch Sentry');
      expect(sensor.ipAddress, '192.168.1.150');
      expect(sensor.cameraId, 'cam_front');
      expect(sensor.pirMotion, true);
      expect(sensor.distanceCm, 42.5);
      expect(sensor.door1Open, true);
      expect(sensor.door2Open, false);
      expect(sensor.toggles['pir'], true);
      expect(sensor.toggles['door1'], false);
      expect(sensor.lastHeartbeat, isNotNull);

      final outJson = sensor.toJson();
      expect(outJson['pir_motion'], true);
      expect(outJson['distance_cm'], 42.5);
      expect(outJson['toggles']['door1'], false);
    });

    test('fromJson parses backend enabled_sensors and sensor_states', () {
      final backendJson = {
        'id': 'esp32_node_backend',
        'name': 'Backyard Node',
        'ip_address': '192.168.1.155',
        'associated_camera_id': 'cam_backyard',
        'enabled_sensors': {
          'pir_enabled': true,
          'ultrasonic_enabled': false,
          'door1_enabled': true,
          'door2_enabled': false,
        },
        'sensor_states': {
          'pir_motion': false,
          'distance_cm': 120.0,
          'door1_open': false,
          'door2_open': true,
        },
      };

      final sensor = Esp32Sensor.fromJson(backendJson);
      expect(sensor.cameraId, 'cam_backyard');
      expect(sensor.toggles['pir'], true);
      expect(sensor.toggles['ultrasonic'], false);
      expect(sensor.toggles['door1'], true);
      expect(sensor.toggles['door2'], false);
      expect(sensor.distanceCm, 120.0);
      expect(sensor.door2Open, true);
    });

    test('copyWith updates sensor toggles', () {
      final sensor = Esp32Sensor(
        id: 'esp32_test',
        name: 'Test Node',
        ipAddress: '192.168.1.1',
      );

      final updated = sensor.copyWith(
        toggles: {'pir': false, 'ultrasonic': true, 'door1': true, 'door2': true},
        pirMotion: true,
      );

      expect(updated.toggles['pir'], false);
      expect(updated.pirMotion, true);
      expect(sensor.toggles['pir'], true); // immutability test
    });
  });
}
