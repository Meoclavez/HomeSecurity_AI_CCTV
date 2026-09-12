import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

import '../core/theme/app_theme.dart';
import '../models/camera_feed.dart';
import '../services/api_service.dart';
import 'login_screen.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({Key? key}) : super(key: key);

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final _storage = const FlutterSecureStorage();
  final ApiService _apiService = ApiService();
  String _serverUrl = '';

  List<CameraFeed> _cameras = [];
  List<Esp32Sensor> _sensors = [];
  bool _isLoadingSensors = false;

  @override
  void initState() {
    super.initState();
    _loadSettings();
    _loadCameras();
    _loadSensors();
  }

  Future<void> _loadSettings() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _serverUrl = prefs.getString('server_url') ?? 'Unknown';
    });
  }

  Future<void> _loadCameras() async {
    try {
      final cameras = await _apiService.getCameras();
      if (mounted) {
        setState(() {
          _cameras = cameras;
        });
      }
    } catch (_) {}
  }

  Future<void> _loadSensors() async {
    setState(() => _isLoadingSensors = true);
    try {
      final sensors = await _apiService.getSensors();
      if (mounted) {
        setState(() {
          _sensors = sensors;
          _isLoadingSensors = false;
        });
      }
    } catch (e) {
      if (mounted) setState(() => _isLoadingSensors = false);
    }
  }

  Future<void> _onToggleSensorFeature(Esp32Sensor sensor, String key, bool value) async {
    final updatedToggles = Map<String, bool>.from(sensor.toggles);
    updatedToggles[key] = value;

    final updatedSensor = sensor.copyWith(toggles: updatedToggles);
    setState(() {
      final index = _sensors.indexWhere((s) => s.id == sensor.id);
      if (index != -1) {
        _sensors[index] = updatedSensor;
      }
    });

    await _apiService.updateSensorToggles(sensor.id, updatedToggles);
  }

  void _openCameraAiConfigModal() {
    if (_cameras.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No cameras loaded yet. Please check edge server connection.')),
      );
      return;
    }

    String selectedCameraId = _cameras.first.id;
    CameraFeatures currentFeatures = _cameras.first.features ?? const CameraFeatures();

    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      isScrollControlled: true,
      builder: (ctx) {
        return StatefulBuilder(
          builder: (BuildContext sheetCtx, StateSetter setModalState) {
            final activeCam = _cameras.firstWhere(
              (c) => c.id == selectedCameraId,
              orElse: () => _cameras.first,
            );

            return Container(
              height: MediaQuery.of(context).size.height * 0.85,
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
                child: Column(
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
                          child: const Icon(Icons.psychology, color: AppTheme.cyberBlue, size: 24),
                        ),
                        const SizedBox(width: 12),
                        const Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Camera AI Inference Features',
                                style: TextStyle(
                                  fontSize: 17,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.white,
                                ),
                              ),
                              Text(
                                'Edge N100 + Hailo-8 Hardware Acceleration',
                                style: TextStyle(fontSize: 12, color: Colors.white60),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 16),
                    // Camera Selector Chip Row
                    const Text(
                      'TARGET CAMERA',
                      style: TextStyle(fontSize: 11, fontWeight: FontWeight.bold, color: Colors.white54),
                    ),
                    const SizedBox(height: 8),
                    SingleChildScrollView(
                      scrollDirection: Axis.horizontal,
                      child: Row(
                        children: _cameras.map((cam) {
                          final isSelected = cam.id == selectedCameraId;
                          return Padding(
                            padding: const EdgeInsets.only(right: 8),
                            child: ChoiceChip(
                              label: Text(cam.name),
                              selected: isSelected,
                              selectedColor: AppTheme.cyberBlue,
                              labelStyle: TextStyle(
                                color: isSelected ? Colors.black : Colors.white,
                                fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                              ),
                              backgroundColor: AppTheme.cardSurface,
                              onSelected: (val) async {
                                if (val) {
                                  setModalState(() => selectedCameraId = cam.id);
                                  final features = await _apiService.getCameraFeatures(cam.id);
                                  setModalState(() => currentFeatures = features);
                                }
                              },
                            ),
                          );
                        }).toList(),
                      ),
                    ),
                    const SizedBox(height: 12),
                    const Divider(color: Colors.white12),
                    // Feature toggles list
                    Expanded(
                      child: ListView(
                        children: [
                          _buildFeatureSwitch(
                            title: 'Fall Detection',
                            subtitle: 'Kinematic trajectory & rapid floor impact analysis',
                            icon: Icons.personal_injury,
                            color: AppTheme.emergencyRed,
                            value: currentFeatures.fallDetectionEnabled,
                            onChanged: (val) async {
                              final updated = currentFeatures.copyWith(fallDetectionEnabled: val);
                              setModalState(() => currentFeatures = updated);
                              await _apiService.updateCameraFeatures(activeCam.id, updated);
                            },
                          ),
                          _buildFeatureSwitch(
                            title: 'Skeletal Tracking',
                            subtitle: '17-point real-time pose estimation and posture tracking',
                            icon: Icons.accessibility_new,
                            color: AppTheme.cyberBlue,
                            value: currentFeatures.skeletalTrackingEnabled,
                            onChanged: (val) async {
                              final updated = currentFeatures.copyWith(skeletalTrackingEnabled: val);
                              setModalState(() => currentFeatures = updated);
                              await _apiService.updateCameraFeatures(activeCam.id, updated);
                            },
                          ),
                          _buildFeatureSwitch(
                            title: 'Object Detection',
                            subtitle: 'YOLOv8 real-time person, vehicle, and animal bounding',
                            icon: Icons.category,
                            color: AppTheme.liveGreen,
                            value: currentFeatures.objectDetectionEnabled,
                            onChanged: (val) async {
                              final updated = currentFeatures.copyWith(objectDetectionEnabled: val);
                              setModalState(() => currentFeatures = updated);
                              await _apiService.updateCameraFeatures(activeCam.id, updated);
                            },
                          ),
                          _buildFeatureSwitch(
                            title: 'Package Detection',
                            subtitle: 'Porch drop-off and parcel theft prevention bounding',
                            icon: Icons.inventory_2,
                            color: Colors.orangeAccent,
                            value: currentFeatures.packageDetectionEnabled,
                            onChanged: (val) async {
                              final updated = currentFeatures.copyWith(packageDetectionEnabled: val);
                              setModalState(() => currentFeatures = updated);
                              await _apiService.updateCameraFeatures(activeCam.id, updated);
                            },
                          ),
                          _buildFeatureSwitch(
                            title: 'Animal Detection',
                            subtitle: 'Pet tracking & wildlife intrusion alerts',
                            icon: Icons.pets,
                            color: Colors.amber,
                            value: currentFeatures.animalDetectionEnabled,
                            onChanged: (val) async {
                              final updated = currentFeatures.copyWith(animalDetectionEnabled: val);
                              setModalState(() => currentFeatures = updated);
                              await _apiService.updateCameraFeatures(activeCam.id, updated);
                            },
                          ),
                          _buildFeatureSwitch(
                            title: 'Vehicle Detection',
                            subtitle: 'Driveway monitoring and license plate tracking',
                            icon: Icons.directions_car,
                            color: Colors.cyanAccent,
                            value: currentFeatures.vehicleDetectionEnabled,
                            onChanged: (val) async {
                              final updated = currentFeatures.copyWith(vehicleDetectionEnabled: val);
                              setModalState(() => currentFeatures = updated);
                              await _apiService.updateCameraFeatures(activeCam.id, updated);
                            },
                          ),
                          _buildFeatureSwitch(
                            title: 'Door Monitoring',
                            subtitle: 'Cross-reference IoT door contact state with vision bounding',
                            icon: Icons.door_front_door,
                            color: Colors.tealAccent,
                            value: currentFeatures.doorMonitoring,
                            onChanged: (val) async {
                              final updated = currentFeatures.copyWith(doorMonitoring: val);
                              setModalState(() => currentFeatures = updated);
                              await _apiService.updateCameraFeatures(activeCam.id, updated);
                            },
                          ),
                          _buildFeatureSwitch(
                            title: '24/7 DVR Recording',
                            subtitle: 'Continuous rolling hardware ring buffer on NVMe SSD',
                            icon: Icons.fiber_manual_record,
                            color: Colors.redAccent,
                            value: currentFeatures.dvrRecording247,
                            onChanged: (val) async {
                              final updated = currentFeatures.copyWith(dvrRecording247: val);
                              setModalState(() => currentFeatures = updated);
                              await _apiService.updateCameraFeatures(activeCam.id, updated);
                            },
                          ),
                        ],
                      ),
                    ),
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
                        onPressed: () => Navigator.pop(sheetCtx),
                        child: const Text('Close'),
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

  Widget _buildFeatureSwitch({
    required String title,
    required String subtitle,
    required IconData icon,
    required Color color,
    required bool value,
    required ValueChanged<bool> onChanged,
  }) {
    return Container(
      margin: const EdgeInsets.symmetric(vertical: 4),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.03),
        borderRadius: BorderRadius.circular(8),
      ),
      child: SwitchListTile(
        secondary: Icon(icon, color: value ? color : Colors.white38),
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14)),
        subtitle: Text(subtitle, style: const TextStyle(fontSize: 11, color: Colors.white54)),
        value: value,
        activeThumbColor: color,
        onChanged: onChanged,
      ),
    );
  }

  Future<void> _logout() async {
    await _storage.delete(key: 'access_token');
    await _storage.delete(key: 'refresh_token');
    if (mounted) {
      Navigator.pushAndRemoveUntil(
        context,
        MaterialPageRoute(builder: (_) => const LoginScreen()),
        (route) => false,
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.darkBackground,
      appBar: AppBar(
        title: const Text('Settings'),
        actions: [
          IconButton(
            icon: const Icon(Icons.logout, color: AppTheme.emergencyRed),
            onPressed: _logout,
            tooltip: 'Logout',
          )
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16.0),
        children: [
          // Section 1: Server Connection
          _buildSectionHeader('Server Connection'),
          ListTile(
            title: const Text('Current Server URL'),
            subtitle: Text(_serverUrl),
            trailing: const Icon(Icons.edit, size: 20),
            onTap: () {},
          ),
          const Divider(color: AppTheme.borderHighlight),

          // Section 2: AI Vision & Hardware Sensors
          _buildSectionHeader('AI Vision & Hardware Sensors'),
          ListTile(
            leading: Container(
              padding: const EdgeInsets.all(8),
              decoration: BoxDecoration(
                color: AppTheme.cyberBlue.withValues(alpha: 0.15),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Icon(Icons.psychology, color: AppTheme.cyberBlue, size: 20),
            ),
            title: const Text('Camera AI Vision Engines', style: TextStyle(fontWeight: FontWeight.w600)),
            subtitle: const Text('Configure Fall Detection, Skeletal & YOLOv8 models per camera'),
            trailing: const Icon(Icons.arrow_forward_ios, size: 16, color: Colors.white54),
            onTap: _openCameraAiConfigModal,
          ),
          const SizedBox(height: 12),
          // Sentry Hardware Manager Header
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4.0, vertical: 4.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'SENTRY HARDWARE MANAGER (ESP32 NODES)',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    color: Colors.white60,
                    letterSpacing: 0.5,
                  ),
                ),
                IconButton(
                  icon: const Icon(Icons.refresh, size: 18, color: AppTheme.cyberBlue),
                  onPressed: _loadSensors,
                  tooltip: 'Rescan ESP32 Sensor Nodes',
                ),
              ],
            ),
          ),
          if (_isLoadingSensors)
            const Padding(
              padding: EdgeInsets.symmetric(vertical: 8),
              child: Center(child: CircularProgressIndicator(color: AppTheme.cyberBlue)),
            )
          else if (_sensors.isEmpty)
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: AppTheme.cardSurface,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: AppTheme.borderHighlight),
              ),
              child: const Text(
                'No ESP32 sensor hardware detected on LAN. Ensure sensors are powered on and connected to Wi-Fi.',
                style: TextStyle(color: Colors.white54, fontSize: 12),
              ),
            )
          else
            ..._sensors.map((sensor) => _buildSensorNodeCard(sensor)).toList(),

          const Divider(color: AppTheme.borderHighlight),

          // Section 3: Hardware & Storage
          _buildSectionHeader('Hardware & Storage'),
          ListTile(
            title: const Text('Hardware Status'),
            subtitle: const Text('Hailo-8 / Intel QuickSync'),
            trailing: const Icon(Icons.memory, size: 20),
            onTap: () {},
          ),
          ListTile(
            title: const Text('Storage Policy'),
            subtitle: const Text('Retention, disk usage, cleanup'),
            trailing: const Icon(Icons.storage, size: 20),
            onTap: () {},
          ),
          const Divider(color: AppTheme.borderHighlight),

          // Section 4: Cameras & Notifications
          _buildSectionHeader('Cameras & Notifications'),
          ListTile(
            title: const Text('Manage Cameras'),
            subtitle: const Text('Add, remove, or edit RTSP streams'),
            trailing: const Icon(Icons.videocam, size: 20),
            onTap: () {},
          ),
          ListTile(
            title: const Text('Notifications'),
            subtitle: const Text('Push alerts, sound settings'),
            trailing: const Icon(Icons.notifications, size: 20),
            onTap: () {},
          ),
          const Divider(color: AppTheme.borderHighlight),

          // Section 5: Security & Account
          _buildSectionHeader('Security & Account'),
          ListTile(
            title: const Text('Change Password'),
            trailing: const Icon(Icons.lock, size: 20),
            onTap: () {},
          ),
          ListTile(
            title: const Text('Biometric Login'),
            trailing: Switch(
              value: true,
              onChanged: (val) {},
              activeThumbColor: AppTheme.cyberBlue,
            ),
          ),
          const Divider(color: AppTheme.borderHighlight),

          // Section 6: About
          _buildSectionHeader('About'),
          const ListTile(
            title: Text('Version'),
            subtitle: Text('1.2.0 (Build 42)'),
          ),
        ],
      ),
    );
  }

  Widget _buildSensorNodeCard(Esp32Sensor sensor) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: AppTheme.cardSurface,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: AppTheme.borderHighlight),
      ),
      child: Theme(
        data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          initiallyExpanded: true,
          leading: Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.05),
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Icon(Icons.developer_board, color: AppTheme.liveGreen, size: 22),
          ),
          title: Text(
            sensor.name,
            style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
          ),
          subtitle: Text(
            'IP: ${sensor.ipAddress} • Camera: ${sensor.cameraId ?? "Unassigned"}',
            style: const TextStyle(color: Colors.white54, fontSize: 11),
          ),
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Real-time telemetry preview badges
                  Row(
                    children: [
                      _buildMiniBadge(
                        label: sensor.pirMotion ? '🚶 PIR ACTIVE' : '🚶 PIR Clear',
                        color: sensor.pirMotion ? AppTheme.emergencyRed : Colors.white54,
                        bgColor: sensor.pirMotion
                            ? AppTheme.emergencyRed.withValues(alpha: 0.2)
                            : Colors.white.withValues(alpha: 0.05),
                      ),
                      const SizedBox(width: 6),
                      _buildMiniBadge(
                        label: '📏 ${sensor.distanceCm.toStringAsFixed(1)} cm',
                        color: (sensor.distanceCm > 0 && sensor.distanceCm < 60)
                            ? Colors.amber
                            : Colors.white70,
                        bgColor: Colors.white.withValues(alpha: 0.05),
                      ),
                      const SizedBox(width: 6),
                      _buildMiniBadge(
                        label: '🚪 D1: ${sensor.door1Open ? "OPEN" : "CLOSED"}',
                        color: sensor.door1Open ? AppTheme.emergencyRed : AppTheme.liveGreen,
                        bgColor: (sensor.door1Open ? AppTheme.emergencyRed : AppTheme.liveGreen)
                            .withValues(alpha: 0.15),
                      ),
                      const SizedBox(width: 6),
                      _buildMiniBadge(
                        label: '🚪 D2: ${sensor.door2Open ? "OPEN" : "CLOSED"}',
                        color: sensor.door2Open ? AppTheme.emergencyRed : AppTheme.liveGreen,
                        bgColor: (sensor.door2Open ? AppTheme.emergencyRed : AppTheme.liveGreen)
                            .withValues(alpha: 0.15),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    'PER-SENSOR HARDWARE TOGGLES',
                    style: TextStyle(
                      fontSize: 10,
                      fontWeight: FontWeight.bold,
                      color: Colors.white38,
                      letterSpacing: 0.5,
                    ),
                  ),
                  const SizedBox(height: 6),

                  // 1. PIR Motion Sensor Toggle
                  _buildPerSensorSwitch(
                    title: 'PIR Motion Sensor',
                    subtitle: 'Passive infrared human presence trigger',
                    icon: Icons.directions_walk,
                    value: sensor.toggles['pir'] ?? true,
                    activeColor: AppTheme.emergencyRed,
                    onChanged: (val) => _onToggleSensorFeature(sensor, 'pir', val),
                  ),

                  // 2. Ultrasonic Distance Sensor Toggle
                  _buildPerSensorSwitch(
                    title: 'Ultrasonic Distance',
                    subtitle: 'HC-SR04 sonar proximity and range gauge',
                    icon: Icons.straighten,
                    value: sensor.toggles['ultrasonic'] ?? true,
                    activeColor: Colors.amber,
                    onChanged: (val) => _onToggleSensorFeature(sensor, 'ultrasonic', val),
                  ),

                  // 3. Door 1 Reed Sensor Toggle
                  _buildPerSensorSwitch(
                    title: 'Door 1 Reed Switch',
                    subtitle: 'Primary entry magnetic portal contact',
                    icon: Icons.sensor_door,
                    value: sensor.toggles['door1'] ?? true,
                    activeColor: AppTheme.cyberBlue,
                    onChanged: (val) => _onToggleSensorFeature(sensor, 'door1', val),
                  ),

                  // 4. Door 2 Reed Sensor Toggle
                  _buildPerSensorSwitch(
                    title: 'Door 2 Reed Switch',
                    subtitle: 'Secondary entry magnetic portal contact',
                    icon: Icons.sensor_door,
                    value: sensor.toggles['door2'] ?? true,
                    activeColor: AppTheme.cyberBlue,
                    onChanged: (val) => _onToggleSensorFeature(sensor, 'door2', val),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMiniBadge({
    required String label,
    required Color color,
    required Color bgColor,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Text(
        label,
        style: TextStyle(
          color: color,
          fontSize: 10,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }

  Widget _buildPerSensorSwitch({
    required String title,
    required String subtitle,
    required IconData icon,
    required bool value,
    required Color activeColor,
    required ValueChanged<bool> onChanged,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2.0),
      child: SwitchListTile(
        dense: true,
        contentPadding: EdgeInsets.zero,
        secondary: Icon(icon, color: value ? activeColor : Colors.white38, size: 20),
        title: Text(title, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
        subtitle: Text(subtitle, style: const TextStyle(fontSize: 10, color: Colors.white54)),
        value: value,
        activeThumbColor: activeColor,
        onChanged: onChanged,
      ),
    );
  }

  Widget _buildSectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 4.0),
      child: Text(
        title,
        style: const TextStyle(
          color: AppTheme.cyberBlue,
          fontWeight: FontWeight.bold,
          fontSize: 14,
        ),
      ),
    );
  }
}

