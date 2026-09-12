class CameraFeatures {
  final bool fallDetectionEnabled;
  final bool skeletalTrackingEnabled;
  final bool objectDetectionEnabled;
  final bool packageDetectionEnabled;
  final bool animalDetectionEnabled;
  final bool vehicleDetectionEnabled;
  final bool doorMonitoring;
  final bool dvrRecording247;

  const CameraFeatures({
    this.fallDetectionEnabled = false,
    this.skeletalTrackingEnabled = false,
    this.objectDetectionEnabled = true,
    this.packageDetectionEnabled = false,
    this.animalDetectionEnabled = false,
    this.vehicleDetectionEnabled = false,
    this.doorMonitoring = false,
    this.dvrRecording247 = true,
  });

  factory CameraFeatures.fromJson(Map<String, dynamic> json) {
    return CameraFeatures(
      fallDetectionEnabled: json['fall_detection_enabled'] ?? json['fallDetectionEnabled'] ?? false,
      skeletalTrackingEnabled: json['skeletal_tracking_enabled'] ?? json['skeletalTrackingEnabled'] ?? false,
      objectDetectionEnabled: json['object_detection_enabled'] ?? json['objectDetectionEnabled'] ?? true,
      packageDetectionEnabled: json['package_detection_enabled'] ?? json['packageDetectionEnabled'] ?? false,
      animalDetectionEnabled: json['animal_detection_enabled'] ?? json['animalDetectionEnabled'] ?? false,
      vehicleDetectionEnabled: json['vehicle_detection_enabled'] ?? json['vehicleDetectionEnabled'] ?? false,
      doorMonitoring: json['door_monitoring'] ?? json['doorMonitoring'] ?? false,
      dvrRecording247: json['dvr_recording_247'] ?? json['dvrRecording247'] ?? true,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'fall_detection_enabled': fallDetectionEnabled,
      'skeletal_tracking_enabled': skeletalTrackingEnabled,
      'object_detection_enabled': objectDetectionEnabled,
      'package_detection_enabled': packageDetectionEnabled,
      'animal_detection_enabled': animalDetectionEnabled,
      'vehicle_detection_enabled': vehicleDetectionEnabled,
      'door_monitoring': doorMonitoring,
      'dvr_recording_247': dvrRecording247,
    };
  }

  CameraFeatures copyWith({
    bool? fallDetectionEnabled,
    bool? skeletalTrackingEnabled,
    bool? objectDetectionEnabled,
    bool? packageDetectionEnabled,
    bool? animalDetectionEnabled,
    bool? vehicleDetectionEnabled,
    bool? doorMonitoring,
    bool? dvrRecording247,
  }) {
    return CameraFeatures(
      fallDetectionEnabled: fallDetectionEnabled ?? this.fallDetectionEnabled,
      skeletalTrackingEnabled: skeletalTrackingEnabled ?? this.skeletalTrackingEnabled,
      objectDetectionEnabled: objectDetectionEnabled ?? this.objectDetectionEnabled,
      packageDetectionEnabled: packageDetectionEnabled ?? this.packageDetectionEnabled,
      animalDetectionEnabled: animalDetectionEnabled ?? this.animalDetectionEnabled,
      vehicleDetectionEnabled: vehicleDetectionEnabled ?? this.vehicleDetectionEnabled,
      doorMonitoring: doorMonitoring ?? this.doorMonitoring,
      dvrRecording247: dvrRecording247 ?? this.dvrRecording247,
    );
  }
}

class CameraFeed {
  final String id;
  final String name;
  final String location;
  final String rtspUrl;
  final String webrtcUrl;
  final String status;
  final int fps;
  final String resolution;
  final bool isAiEnabled;
  final List<String> aiModels;

  final String? diagnosticState;
  final String? errorMessage;
  final bool isAutoRecovering;
  final CameraFeatures? features;

  CameraFeed({
    required this.id,
    required this.name,
    required this.location,
    required this.rtspUrl,
    required this.webrtcUrl,
    required this.status,
    required this.fps,
    required this.resolution,
    required this.isAiEnabled,
    required this.aiModels,
    this.diagnosticState,
    this.errorMessage,
    this.isAutoRecovering = false,
    this.features,
  });

  factory CameraFeed.fromJson(Map<String, dynamic> json) {
    return CameraFeed(
      id: json['id'] ?? '',
      name: json['name'] ?? '',
      location: json['location'] ?? '',
      rtspUrl: json['rtsp_url'] ?? '',
      webrtcUrl: json['webrtc_url'] ?? '',
      status: json['status'] ?? 'ONLINE',
      fps: json['fps'] ?? 30,
      resolution: json['resolution'] ?? '1920x1080',
      isAiEnabled: json['is_ai_enabled'] ?? true,
      aiModels: List<String>.from(json['ai_models'] ?? []),
      diagnosticState: json['diagnostic_state'],
      errorMessage: json['error_message'],
      isAutoRecovering: json['is_auto_recovering'] ?? false,
      features: json['features'] != null
          ? CameraFeatures.fromJson(json['features'] as Map<String, dynamic>)
          : null,
    );
  }

  CameraFeed copyWith({
    String? id,
    String? name,
    String? location,
    String? rtspUrl,
    String? webrtcUrl,
    String? status,
    int? fps,
    String? resolution,
    bool? isAiEnabled,
    List<String>? aiModels,
    String? diagnosticState,
    String? errorMessage,
    bool? isAutoRecovering,
    CameraFeatures? features,
  }) {
    return CameraFeed(
      id: id ?? this.id,
      name: name ?? this.name,
      location: location ?? this.location,
      rtspUrl: rtspUrl ?? this.rtspUrl,
      webrtcUrl: webrtcUrl ?? this.webrtcUrl,
      status: status ?? this.status,
      fps: fps ?? this.fps,
      resolution: resolution ?? this.resolution,
      isAiEnabled: isAiEnabled ?? this.isAiEnabled,
      aiModels: aiModels ?? this.aiModels,
      diagnosticState: diagnosticState ?? this.diagnosticState,
      errorMessage: errorMessage ?? this.errorMessage,
      isAutoRecovering: isAutoRecovering ?? this.isAutoRecovering,
      features: features ?? this.features,
    );
  }

  bool get isOnline => status.toUpperCase() == 'ONLINE';
}

class Esp32Sensor {
  final String id;
  final String name;
  final String ipAddress;
  final String? cameraId;
  final bool pirMotion;
  final double distanceCm;
  final bool door1Open;
  final bool door2Open;
  final Map<String, bool> toggles;
  final DateTime? lastHeartbeat;

  Esp32Sensor({
    required this.id,
    required this.name,
    required this.ipAddress,
    this.cameraId,
    this.pirMotion = false,
    this.distanceCm = 0.0,
    this.door1Open = false,
    this.door2Open = false,
    Map<String, bool>? toggles,
    this.lastHeartbeat,
  }) : toggles = toggles ?? {
          'pir': true,
          'ultrasonic': true,
          'door1': true,
          'door2': true,
        };

  factory Esp32Sensor.fromJson(Map<String, dynamic> json) {
    Map<String, bool> parsedToggles = {};
    if (json['toggles'] != null && json['toggles'] is Map) {
      (json['toggles'] as Map).forEach((k, v) {
        parsedToggles[k.toString()] = v == true;
      });
    } else if (json['enabled_sensors'] != null && json['enabled_sensors'] is Map) {
      final es = json['enabled_sensors'] as Map;
      parsedToggles['pir'] = (es['pir_enabled'] ?? es['pir'] ?? true) == true;
      parsedToggles['ultrasonic'] = (es['ultrasonic_enabled'] ?? es['ultrasonic'] ?? true) == true;
      parsedToggles['door1'] = (es['door1_enabled'] ?? es['door1'] ?? true) == true;
      parsedToggles['door2'] = (es['door2_enabled'] ?? es['door2'] ?? true) == true;
    } else {
      parsedToggles = {
        'pir': true,
        'ultrasonic': true,
        'door1': true,
        'door2': true,
      };
    }

    DateTime? heartbeat;
    if (json['last_heartbeat'] != null) {
      heartbeat = DateTime.tryParse(json['last_heartbeat'].toString());
    } else if (json['lastHeartbeat'] != null) {
      heartbeat = DateTime.tryParse(json['lastHeartbeat'].toString());
    }

    final states = (json['sensor_states'] is Map) ? json['sensor_states'] as Map : {};

    return Esp32Sensor(
      id: json['id'] ?? '',
      name: json['name'] ?? 'ESP32 Sentry Node',
      ipAddress: json['ip_address'] ?? json['ipAddress'] ?? '',
      cameraId: json['camera_id'] ?? json['cameraId'] ?? json['associated_camera_id'],
      pirMotion: json['pir_motion'] ?? json['pirMotion'] ?? states['pir_motion'] ?? false,
      distanceCm: (json['distance_cm'] ?? json['distanceCm'] ?? states['distance_cm'] ?? 0.0).toDouble(),
      door1Open: json['door1_open'] ?? json['door1Open'] ?? states['door1_open'] ?? false,
      door2Open: json['door2_open'] ?? json['door2Open'] ?? states['door2_open'] ?? false,
      toggles: parsedToggles,
      lastHeartbeat: heartbeat,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'name': name,
      'ip_address': ipAddress,
      'camera_id': cameraId,
      'pir_motion': pirMotion,
      'distance_cm': distanceCm,
      'door1_open': door1Open,
      'door2_open': door2Open,
      'toggles': toggles,
      'last_heartbeat': lastHeartbeat?.toIso8601String(),
    };
  }

  Esp32Sensor copyWith({
    String? id,
    String? name,
    String? ipAddress,
    String? cameraId,
    bool? pirMotion,
    double? distanceCm,
    bool? door1Open,
    bool? door2Open,
    Map<String, bool>? toggles,
    DateTime? lastHeartbeat,
  }) {
    return Esp32Sensor(
      id: id ?? this.id,
      name: name ?? this.name,
      ipAddress: ipAddress ?? this.ipAddress,
      cameraId: cameraId ?? this.cameraId,
      pirMotion: pirMotion ?? this.pirMotion,
      distanceCm: distanceCm ?? this.distanceCm,
      door1Open: door1Open ?? this.door1Open,
      door2Open: door2Open ?? this.door2Open,
      toggles: toggles != null ? Map<String, bool>.from(toggles) : Map<String, bool>.from(this.toggles),
      lastHeartbeat: lastHeartbeat ?? this.lastHeartbeat,
    );
  }
}
