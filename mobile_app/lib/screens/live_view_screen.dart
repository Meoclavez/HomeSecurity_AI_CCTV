import 'dart:async';
import 'package:flutter/material.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import '../core/theme/app_theme.dart';
import '../models/camera_feed.dart';
import '../models/security_event.dart';
import '../services/api_service.dart';
import '../services/webrtc_service.dart';
import '../widgets/biometric_gate.dart';
import '../widgets/talkback_button.dart';
import '../widgets/timeline_models.dart';
import '../widgets/timeline_scrubber_widget.dart';
import 'clip_player_screen.dart';

class LiveViewScreen extends StatefulWidget {
  final CameraFeed camera;

  const LiveViewScreen({Key? key, required this.camera}) : super(key: key);

  @override
  State<LiveViewScreen> createState() => _LiveViewScreenState();
}

class _LiveViewScreenState extends State<LiveViewScreen> {
  final WebRtcService _webrtcService = WebRtcService();
  final ApiService _apiService = ApiService();
  bool _isConnecting = true;
  String? _error;
  bool _isMuted = false;

  late CameraFeed _currentCamera;
  List<CameraFeed> _cameras = [];

  CameraFeatures _cameraFeatures = const CameraFeatures();
  Esp32Sensor? _attachedSensor;
  Timer? _sensorPollTimer;

  late List<TimelineRecordingSegment> _recordingSegments;
  late List<TimelineEventPin> _eventPins;

  @override
  void initState() {
    super.initState();
    _currentCamera = widget.camera;
    _cameraFeatures = widget.camera.features ?? const CameraFeatures();
    _initTimelineMockData();
    _initAndConnect();
    _loadCameras();
    _loadCameraFeatures();
    _loadSensors();
    _startSensorPolling();
  }

  void _startSensorPolling() {
    _sensorPollTimer?.cancel();
    _sensorPollTimer = Timer.periodic(const Duration(seconds: 4), (_) {
      _loadSensors();
    });
  }

  Future<void> _loadCameraFeatures() async {
    try {
      final features = await _apiService.getCameraFeatures(_currentCamera.id);
      if (mounted) {
        setState(() {
          _cameraFeatures = features;
        });
      }
    } catch (e) {
      debugPrint('Error loading camera features: $e');
    }
  }

  Future<void> _loadSensors() async {
    try {
      final sensors = await _apiService.getSensors();
      if (!mounted) return;
      final matched = sensors.firstWhere(
        (s) => s.cameraId == _currentCamera.id,
        orElse: () => sensors.isNotEmpty
            ? sensors.first
            : Esp32Sensor(
                id: 'esp32_default',
                name: 'Front Porch Sentry',
                ipAddress: '192.168.1.145',
                cameraId: _currentCamera.id,
                pirMotion: true,
                distanceCm: 48.5,
                door1Open: false,
                door2Open: false,
              ),
      );

      setState(() {
        _attachedSensor = matched;
      });
    } catch (e) {
      debugPrint('Error loading sensors: $e');
    }
  }

  Future<void> _loadCameras() async {
    try {
      final list = await _apiService.getCameras();
      if (mounted) {
        setState(() {
          _cameras = list;
        });
      }
    } catch (e) {
      debugPrint('Error loading cameras: $e');
    }
  }

  void _switchCamera(CameraFeed newCam) {
    if (newCam.id == _currentCamera.id) return;
    setState(() {
      _currentCamera = newCam;
      _cameraFeatures = newCam.features ?? const CameraFeatures();
      _initTimelineMockData();
    });
    _initAndConnect();
    _loadCameraFeatures();
    _loadSensors();
  }

  void _initTimelineMockData() {
    final now = DateTime.now();
    _recordingSegments = [
      TimelineRecordingSegment(
        start: now.subtract(const Duration(hours: 12)),
        end: now.subtract(const Duration(hours: 5)),
      ),
      TimelineRecordingSegment(
        start: now.subtract(const Duration(hours: 4, minutes: 30)),
        end: now,
      ),
    ];
    _eventPins = [
      TimelineEventPin(
        event: SecurityEvent(
          id: 'ev_01',
          cameraId: _currentCamera.id,
          cameraName: _currentCamera.name,
          location: _currentCamera.location,
          eventType: 'FALL_DETECTED',
          severity: 'CRITICAL',
          confidence: 0.96,
          timestamp: now.subtract(const Duration(hours: 2, minutes: 15)),
          acknowledged: false,
        ),
      ),
    ];
  }

  Future<void> _initAndConnect() async {
    setState(() {
      _isConnecting = true;
      _error = null;
    });
    try {
      await _webrtcService.initialize();
      await _webrtcService.connect(_currentCamera.id, enableBackchannel: true);
      if (mounted) setState(() => _isConnecting = false);
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = e.toString();
          _isConnecting = false;
        });
      }
    }
  }

  @override
  void dispose() {
    _sensorPollTimer?.cancel();
    _webrtcService.dispose();
    super.dispose();
  }

  void _showAiControlsModal() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (ctx) {
        return StatefulBuilder(
          builder: (BuildContext sheetContext, StateSetter setModalState) {
            return Container(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
              decoration: BoxDecoration(
                color: const Color(0xFF161B22),
                borderRadius: const BorderRadius.vertical(top: Radius.circular(20)),
                border: Border.all(color: AppTheme.borderHighlight),
                boxShadow: const [
                  BoxShadow(color: Colors.black54, blurRadius: 16, offset: Offset(0, -4)),
                ],
              ),
              child: SafeArea(
                top: false,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Center(
                      child: Container(
                        width: 40,
                        height: 4,
                        margin: const EdgeInsets.only(bottom: 16),
                        decoration: BoxDecoration(
                          color: Colors.white24,
                          borderRadius: BorderRadius.circular(2),
                        ),
                      ),
                    ),
                    Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.all(8),
                          decoration: BoxDecoration(
                            color: AppTheme.cyberBlue.withValues(alpha: 0.15),
                            borderRadius: BorderRadius.circular(8),
                          ),
                          child: const Icon(Icons.auto_awesome, color: AppTheme.cyberBlue, size: 22),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              const Text(
                                'AI Vision Controls',
                                style: TextStyle(
                                  fontSize: 17,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white,
                                ),
                              ),
                              Text(
                                '${_currentCamera.name} • Edge N100 / Hailo-8 Acceleration',
                                style: const TextStyle(fontSize: 12, color: Colors.white60),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    const Divider(color: Colors.white12),
                    const SizedBox(height: 8),

                    // 1. Fall Detection
                    _buildAiToggleTile(
                      icon: Icons.personal_injury,
                      title: 'Fall Detection',
                      subtitle: 'Pose kinematic velocity & rapid floor impact analysis',
                      value: _cameraFeatures.fallDetectionEnabled,
                      activeColor: AppTheme.emergencyRed,
                      onChanged: (val) async {
                        final updated = _cameraFeatures.copyWith(fallDetectionEnabled: val);
                        setState(() => _cameraFeatures = updated);
                        setModalState(() {});
                        await _apiService.updateCameraFeatures(_currentCamera.id, updated);
                      },
                    ),

                    // 2. Skeletal Tracking
                    _buildAiToggleTile(
                      icon: Icons.accessibility_new,
                      title: 'Skeletal Tracking',
                      subtitle: '17-point real-time pose estimation & motion vectors',
                      value: _cameraFeatures.skeletalTrackingEnabled,
                      activeColor: AppTheme.cyberBlue,
                      onChanged: (val) async {
                        final updated = _cameraFeatures.copyWith(skeletalTrackingEnabled: val);
                        setState(() => _cameraFeatures = updated);
                        setModalState(() {});
                        await _apiService.updateCameraFeatures(_currentCamera.id, updated);
                      },
                    ),

                    // 3. Object Detection
                    _buildAiToggleTile(
                      icon: Icons.category,
                      title: 'Object Detection',
                      subtitle: 'Real-time YOLOv8 person, car, item bounding boxes',
                      value: _cameraFeatures.objectDetectionEnabled,
                      activeColor: AppTheme.liveGreen,
                      onChanged: (val) async {
                        final updated = _cameraFeatures.copyWith(objectDetectionEnabled: val);
                        setState(() => _cameraFeatures = updated);
                        setModalState(() {});
                        await _apiService.updateCameraFeatures(_currentCamera.id, updated);
                      },
                    ),

                    const SizedBox(height: 8),
                    const Divider(color: Colors.white12),
                    const SizedBox(height: 12),

                    SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: AppTheme.cardSurface,
                          foregroundColor: Colors.white,
                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                          padding: const EdgeInsets.symmetric(vertical: 12),
                        ),
                        onPressed: () => Navigator.pop(sheetContext),
                        child: const Text('Done'),
                      ),
                    ),
                  ],
                ),
              ),
            );
          },
        );
      },
    );
  }

  Widget _buildAiToggleTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required bool value,
    required Color activeColor,
    required ValueChanged<bool> onChanged,
  }) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 4),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.03),
        borderRadius: BorderRadius.circular(8),
      ),
      child: SwitchListTile(
        secondary: Icon(icon, color: value ? activeColor : Colors.white38),
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14)),
        subtitle: Text(subtitle, style: const TextStyle(fontSize: 11, color: Colors.white54)),
        value: value,
        activeThumbColor: activeColor,
        onChanged: onChanged,
      ),
    );
  }

  Widget _buildIoTSensorStatusStrip() {
    final sensor = _attachedSensor;
    final bool hasSensor = sensor != null;
    final bool isPirActive = hasSensor && sensor.pirMotion;
    final double distance = hasSensor ? sensor.distanceCm : 0.0;
    final bool door1Open = hasSensor && sensor.door1Open;
    final bool door2Open = hasSensor && sensor.door2Open;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0xFF14181F),
        border: Border(
          top: BorderSide(color: Colors.white.withValues(alpha: 0.08)),
          bottom: BorderSide(color: Colors.white.withValues(alpha: 0.08)),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  Container(
                    width: 7,
                    height: 7,
                    decoration: BoxDecoration(
                      color: hasSensor ? AppTheme.liveGreen : Colors.white24,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 6),
                  Text(
                    hasSensor
                        ? 'SENTRY NODE: ${sensor.name.toUpperCase()} (${sensor.ipAddress})'
                        : 'SENTRY SENSOR NODE OFFLINE',
                    style: TextStyle(
                      color: Colors.white.withValues(alpha: 0.6),
                      fontSize: 10,
                      fontWeight: FontWeight.w600,
                      letterSpacing: 0.5,
                    ),
                  ),
                ],
              ),
              Text(
                hasSensor && sensor.lastHeartbeat != null ? 'SYNCED' : 'STANDBY',
                style: const TextStyle(
                  color: AppTheme.liveGreen,
                  fontSize: 9,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              children: [
                // 1. PIR Motion Badge (active highlight)
                AnimatedContainer(
                  duration: const Duration(milliseconds: 300),
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                  decoration: BoxDecoration(
                    color: isPirActive
                        ? AppTheme.emergencyRed.withValues(alpha: 0.25)
                        : Colors.white.withValues(alpha: 0.05),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                      color: isPirActive ? AppTheme.emergencyRed : Colors.white12,
                      width: isPirActive ? 1.5 : 1.0,
                    ),
                  ),
                  child: Row(
                    children: [
                      Text(
                        '🚶',
                        style: TextStyle(
                          fontSize: 13,
                          color: isPirActive ? AppTheme.emergencyRed : Colors.white54,
                        ),
                      ),
                      const SizedBox(width: 5),
                      Text(
                        isPirActive ? 'PIR: MOTION DETECTED' : 'PIR: CLEAR',
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.bold,
                          color: isPirActive ? AppTheme.emergencyRed : Colors.white70,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),

                // 2. Ultrasonic Distance Gauge (cm gauge)
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
                  decoration: BoxDecoration(
                    color: (distance > 0 && distance < 60)
                        ? Colors.amber.withValues(alpha: 0.2)
                        : Colors.white.withValues(alpha: 0.05),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(
                      color: (distance > 0 && distance < 60) ? Colors.amber : Colors.white12,
                    ),
                  ),
                  child: Row(
                    children: [
                      const Text('📏', style: TextStyle(fontSize: 13)),
                      const SizedBox(width: 5),
                      Text(
                        '${distance.toStringAsFixed(1)} cm',
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.bold,
                          color: (distance > 0 && distance < 60) ? Colors.amber : Colors.white,
                        ),
                      ),
                      const SizedBox(width: 4),
                      Text(
                        (distance > 0 && distance < 60) ? '(PROXIMITY)' : '(RANGE)',
                        style: TextStyle(
                          fontSize: 9,
                          color: (distance > 0 && distance < 60) ? Colors.amber : Colors.white38,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),

                // 3. Door 1 Pill Tag
                _buildDoorPill(label: 'Door 1', isOpen: door1Open),
                const SizedBox(width: 8),

                // 4. Door 2 Pill Tag
                _buildDoorPill(label: 'Door 2', isOpen: door2Open),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildDoorPill({required String label, required bool isOpen}) {
    final Color badgeColor = isOpen ? const Color(0xFFFF5252) : const Color(0xFF00E676);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: badgeColor.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: badgeColor.withValues(alpha: 0.7)),
      ),
      child: Row(
        children: [
          const Text('🚪', style: TextStyle(fontSize: 13)),
          const SizedBox(width: 5),
          Text(
            '$label: ${isOpen ? 'OPEN' : 'CLOSED'}',
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.bold,
              color: badgeColor,
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return BiometricGate(
      promptReason: 'Authenticate to view secure camera ${_currentCamera.name}',
      child: Scaffold(
        backgroundColor: Colors.black,
        appBar: AppBar(
          backgroundColor: Colors.black,
          title: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(_currentCamera.name, style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
              Text('${_currentCamera.location} • WebRTC <300ms + 2-Way Audio',
                  style: const TextStyle(fontSize: 11, color: AppTheme.liveGreen)),
            ],
          ),
          actions: [
            IconButton(
              icon: const Icon(Icons.auto_awesome, color: AppTheme.cyberBlue),
              tooltip: 'Quick AI Controls',
              onPressed: _showAiControlsModal,
            ),
            IconButton(
              icon: Icon(_isMuted ? Icons.volume_off : Icons.volume_up),
              onPressed: () => setState(() => _isMuted = !_isMuted),
            ),
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: _initAndConnect,
            ),
          ],
        ),
        body: SafeArea(
          child: Column(
            children: [
              if (_cameras.isNotEmpty)
                Container(
                  height: 40,
                  margin: const EdgeInsets.symmetric(vertical: 8),
                  child: ListView.builder(
                    scrollDirection: Axis.horizontal,
                    itemCount: _cameras.length,
                    padding: const EdgeInsets.symmetric(horizontal: 12),
                    itemBuilder: (context, index) {
                      final cam = _cameras[index];
                      final isSelected = cam.id == _currentCamera.id;
                      return Padding(
                        padding: const EdgeInsets.only(right: 8),
                        child: ChoiceChip(
                          label: Text(cam.name,
                              style: TextStyle(
                                  fontSize: 12,
                                  color: isSelected ? Colors.black : Colors.white)),
                          selected: isSelected,
                          selectedColor: AppTheme.cyberBlue,
                          backgroundColor: AppTheme.cardSurface,
                          onSelected: (val) {
                            if (val) _switchCamera(cam);
                          },
                        ),
                      );
                    },
                  ),
                ),
              // 1. Live WebRTC Video Viewport
              Expanded(
                flex: 5,
                child: Stack(
                  children: [
                    Center(
                      child: _isConnecting
                          ? const Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                CircularProgressIndicator(color: AppTheme.cyberBlue),
                                SizedBox(height: 12),
                                Text('Connecting WebRTC Ultra-Low Latency Feed...',
                                    style: TextStyle(color: Colors.white60, fontSize: 12)),
                              ],
                            )
                          : _error != null
                              ? Column(
                                  mainAxisAlignment: MainAxisAlignment.center,
                                  children: [
                                    const Icon(Icons.error_outline,
                                        color: AppTheme.emergencyRed, size: 48),
                                    const SizedBox(height: 8),
                                    Text('WebRTC Failed: $_error',
                                        style: const TextStyle(
                                            color: Colors.white70, fontSize: 12),
                                        textAlign: TextAlign.center),
                                    const SizedBox(height: 12),
                                    ElevatedButton(
                                        onPressed: _initAndConnect,
                                        child: const Text('Retry')),
                                  ],
                                )
                              : RTCVideoView(
                                  _webrtcService.renderer,
                                  objectFit: RTCVideoViewObjectFit.RTCVideoViewObjectFitCover,
                                ),
                    ),
                    Positioned(
                      top: 12,
                      left: 12,
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                        decoration: BoxDecoration(
                          color: Colors.black.withValues(alpha: 0.6),
                          borderRadius: BorderRadius.circular(4),
                          border: Border.all(color: AppTheme.liveGreen),
                        ),
                        child: const Row(
                          children: [
                            Icon(Icons.fiber_manual_record,
                                color: AppTheme.liveGreen, size: 10),
                            SizedBox(width: 4),
                            Text('LIVE (P2P / RELAY)',
                                style: TextStyle(
                                    color: Colors.white,
                                    fontSize: 10,
                                    fontWeight: FontWeight.bold)),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              // IoT Sensor Status Strip below the video stream
              _buildIoTSensorStatusStrip(),

              // 2. Control Bar (Push-to-Talk 2-Way Audio, Quick AI Controls & Privacy Zones)
              Container(
                padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 16),
                color: AppTheme.cardSurface,
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    TalkbackButton(webrtcService: _webrtcService),
                    OutlinedButton.icon(
                      icon: const Icon(Icons.auto_awesome, color: AppTheme.cyberBlue, size: 16),
                      label: const Text('AI Controls',
                          style: TextStyle(color: AppTheme.cyberBlue, fontSize: 12)),
                      style: OutlinedButton.styleFrom(
                        side: BorderSide(color: AppTheme.cyberBlue.withValues(alpha: 0.6)),
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                      ),
                      onPressed: _showAiControlsModal,
                    ),
                    IconButton(
                      icon: const Icon(Icons.security, color: AppTheme.cyberBlue),
                      tooltip: 'Privacy Masking Active',
                      onPressed: () {
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(
                              content: Text(
                                  'Privacy Masking is enforced directly on Edge N100 hardware.')),
                        );
                      },
                    ),
                  ],
                ),
              ),

              // 3. 24-Hour Interactive Timeline Scrubber
              Expanded(
                flex: 4,
                child: Padding(
                  padding: const EdgeInsets.all(12.0),
                  child: TimelineScrubberWidget(
                    initialTime: DateTime.now(),
                    recordingSegments: _recordingSegments,
                    eventPins: _eventPins,
                    onSeek: (selectedTime) {
                      debugPrint('Seeked timeline to: $selectedTime');
                    },
                    onEventSelected: (event) {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                            builder: (context) => ClipPlayerScreen(event: event)),
                      );
                    },
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
