import 'package:flutter/material.dart';
import '../core/theme/app_theme.dart';
import '../models/zone_model.dart';
import '../widgets/zone_canvas_painter.dart';

class ZoneEditorScreen extends StatefulWidget {
  const ZoneEditorScreen({Key? key}) : super(key: key);

  @override
  State<ZoneEditorScreen> createState() => _ZoneEditorScreenState();
}

class _ZoneEditorScreenState extends State<ZoneEditorScreen> {
  String _selectedCamera = 'Camera 01 - Main Gate';
  ZoneType _selectedTool = ZoneType.intrusion;
  TripwireDirection _tripwireDir = TripwireDirection.bidirectional;
  MaskMode _maskMode = MaskMode.blackout;

  final List<ZoneConfig> _savedZones = [];
  ZoneConfig? _activeDraftZone;
  int? _selectedVertexIndex;
  int? _hoveredVertexIndex;
  Offset? _activeMousePosition;

  @override
  void initState() {
    super.initState();
    _loadSampleZones();
  }

  void _loadSampleZones() {
    _savedZones.addAll([
      ZoneConfig(
        id: 'zone_1',
        cameraId: 'cam_01',
        name: 'Driveway Intrusion Polygon',
        zoneType: ZoneType.intrusion,
        polygonPoints: [
          Point2D(x: 0.15, y: 0.35),
          Point2D(x: 0.65, y: 0.30),
          Point2D(x: 0.85, y: 0.75),
          Point2D(x: 0.20, y: 0.80),
        ],
      ),
      ZoneConfig(
        id: 'zone_2',
        cameraId: 'cam_01',
        name: 'Front Gate Tripwire',
        zoneType: ZoneType.tripwire,
        lineStart: Point2D(x: 0.1, y: 0.2),
        lineEnd: Point2D(x: 0.9, y: 0.2),
        direction: TripwireDirection.aToB,
      ),
      ZoneConfig(
        id: 'zone_3',
        cameraId: 'cam_01',
        name: 'Neighbor Window Mask',
        zoneType: ZoneType.privacyMask,
        polygonPoints: [
          Point2D(x: 0.70, y: 0.05),
          Point2D(x: 0.95, y: 0.05),
          Point2D(x: 0.95, y: 0.25),
          Point2D(x: 0.70, y: 0.25),
        ],
      ),
    ]);
  }

  void _startNewZone() {
    setState(() {
      _activeDraftZone = ZoneConfig(
        id: 'draft_${DateTime.now().millisecondsSinceEpoch}',
        cameraId: 'cam_01',
        name: 'New ${_selectedTool.name.toUpperCase()} Zone',
        zoneType: _selectedTool,
        direction: _tripwireDir,
        maskMode: _maskMode,
      );
      _selectedVertexIndex = null;
    });
  }

  void _handleCanvasTap(Offset localPos, Size canvasSize) {
    if (_activeDraftZone == null) return;

    final normalized = _toNormalized(localPos, canvasSize);

    setState(() {
      if (_activeDraftZone!.zoneType == ZoneType.tripwire) {
        if (_activeDraftZone!.lineStart == null) {
          _activeDraftZone!.lineStart = normalized;
        } else if (_activeDraftZone!.lineEnd == null) {
          _activeDraftZone!.lineEnd = normalized;
        }
      } else {
        if (_activeDraftZone!.polygonPoints.length >= 3) {
          final firstPt = _toCanvasOffset(_activeDraftZone!.polygonPoints.first, canvasSize);
          if ((firstPt - localPos).distance < 24) {
            _saveDraftZone();
            return;
          }
        }
        _activeDraftZone!.polygonPoints.add(normalized);
      }
    });
  }

  void _handlePanUpdate(DragUpdateDetails details, Size canvasSize) {
    if (_activeDraftZone == null || _selectedVertexIndex == null) return;

    final normalized = _toNormalized(details.localPosition, canvasSize);
    setState(() {
      if (_activeDraftZone!.zoneType == ZoneType.tripwire) {
        if (_selectedVertexIndex == 0) {
          _activeDraftZone!.lineStart = normalized;
        } else if (_selectedVertexIndex == 1) {
          _activeDraftZone!.lineEnd = normalized;
        }
      } else {
        if (_selectedVertexIndex! < _activeDraftZone!.polygonPoints.length) {
          _activeDraftZone!.polygonPoints[_selectedVertexIndex!] = normalized;
        }
      }
    });
  }

  void _saveDraftZone() {
    if (_activeDraftZone == null) return;
    setState(() {
      _savedZones.add(_activeDraftZone!);
      _activeDraftZone = null;
      _selectedVertexIndex = null;
    });
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Zone saved to Edge AI NPU Pipeline'), backgroundColor: AppTheme.liveGreen),
    );
  }

  Point2D _toNormalized(Offset local, Size size) {
    return Point2D(
      x: (local.dx / size.width).clamp(0.0, 1.0),
      y: (local.dy / size.height).clamp(0.0, 1.0),
    );
  }

  Offset _toCanvasOffset(Point2D pt, Size size) {
    return Offset(pt.x * size.width, pt.y * size.height);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.darkBackground,
      appBar: AppBar(
        title: const Text('Interactive Zone & Privacy Mask Canvas', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
        actions: [
          IconButton(
            tooltip: 'Sync Zones with Backend',
            icon: const Icon(Icons.cloud_upload_outlined, color: AppTheme.cyberBlue),
            onPressed: () {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Synchronized 3 zones with HailoRT edge service.')),
              );
            },
          ),
        ],
      ),
      body: Row(
        children: [
          Container(
            width: 320,
            decoration: const BoxDecoration(
              color: AppTheme.cardSurface,
              border: Border(right: BorderSide(color: AppTheme.borderHighlight, width: 1)),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Padding(
                  padding: const EdgeInsets.all(12),
                  child: DropdownButtonFormField<String>(
                    value: _selectedCamera,
                    decoration: InputDecoration(
                      labelText: 'Select Camera Stream',
                      filled: true,
                      fillColor: AppTheme.darkBackground,
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(8)),
                      contentPadding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                    ),
                    items: ['Camera 01 - Main Gate', 'Camera 02 - Backyard Patio', 'Camera 03 - Warehouse Bay']
                        .map((c) => DropdownMenuItem(value: c, child: Text(c, style: const TextStyle(fontSize: 12))))
                        .toList(),
                    onChanged: (val) => setState(() => _selectedCamera = val!),
                  ),
                ),
                const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                  child: Text('DRAWING TOOL', style: TextStyle(color: Colors.white54, fontSize: 11, fontWeight: FontWeight.bold, letterSpacing: 1)),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 12),
                  child: Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: [
                      _buildToolChip(ZoneType.intrusion, 'Intrusion', Icons.security_rounded, AppTheme.emergencyRed),
                      _buildToolChip(ZoneType.tripwire, 'Tripwire', Icons.timeline_rounded, AppTheme.warningOrange),
                      _buildToolChip(ZoneType.privacyMask, 'Privacy Mask', Icons.blur_on_rounded, Colors.grey),
                      _buildToolChip(ZoneType.door, 'Door ROI', Icons.door_front_door_outlined, AppTheme.cyberBlue),
                      _buildToolChip(ZoneType.package, 'Package Zone', Icons.inventory_2_outlined, AppTheme.liveGreen),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 12),
                  child: Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _activeDraftZone == null ? _startNewZone : null,
                          icon: const Icon(Icons.add, size: 16),
                          label: const Text('New Zone', style: TextStyle(fontSize: 12)),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: AppTheme.cyberBlue,
                            foregroundColor: Colors.black,
                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                          ),
                        ),
                      ),
                      if (_activeDraftZone != null) ...[
                        const SizedBox(width: 8),
                        IconButton(
                          tooltip: 'Done / Save Zone',
                          icon: const Icon(Icons.check_circle_rounded, color: AppTheme.liveGreen),
                          onPressed: _saveDraftZone,
                        ),
                        IconButton(
                          tooltip: 'Cancel Draft',
                          icon: const Icon(Icons.cancel_rounded, color: AppTheme.emergencyRed),
                          onPressed: () => setState(() => _activeDraftZone = null),
                        ),
                      ],
                    ],
                  ),
                ),
                const Divider(height: 24, color: AppTheme.borderHighlight),
                const Padding(
                  padding: EdgeInsets.symmetric(horizontal: 16, vertical: 4),
                  child: Text('CONFIGURED ZONES (3)', style: TextStyle(color: Colors.white54, fontSize: 11, fontWeight: FontWeight.bold, letterSpacing: 1)),
                ),
                Expanded(
                  child: ListView.builder(
                    itemCount: _savedZones.length,
                    itemBuilder: (context, index) {
                      final zone = _savedZones[index];
                      return Container(
                        margin: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                        padding: const EdgeInsets.all(10),
                        decoration: BoxDecoration(
                          color: AppTheme.darkBackground,
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: AppTheme.borderHighlight),
                        ),
                        child: Row(
                          children: [
                            Switch(
                              value: zone.enabled,
                              activeColor: AppTheme.liveGreen,
                              onChanged: (val) => setState(() => zone.enabled = val),
                            ),
                            const SizedBox(width: 8),
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(zone.name, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 12)),
                                  Text('${zone.zoneType.name.toUpperCase()} • ${zone.polygonPoints.length} pts',
                                      style: const TextStyle(color: Colors.white54, fontSize: 10)),
                                ],
                              ),
                            ),
                            IconButton(
                              icon: const Icon(Icons.delete_outline, size: 18, color: Colors.white38),
                              onPressed: () => setState(() => _savedZones.removeAt(index)),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: LayoutBuilder(
              builder: (context, constraints) {
                final canvasSize = Size(constraints.maxWidth, constraints.maxHeight);

                return MouseRegion(
                  onHover: (event) {
                    setState(() {
                      _activeMousePosition = event.localPosition;
                    });
                  },
                  child: GestureDetector(
                    onTapDown: (details) => _handleCanvasTap(details.localPosition, canvasSize),
                    onPanDown: (details) {
                      if (_activeDraftZone == null) return;
                      final local = details.localPosition;
                      if (_activeDraftZone!.zoneType == ZoneType.tripwire) {
                        if (_activeDraftZone!.lineStart != null && (_toCanvasOffset(_activeDraftZone!.lineStart!, canvasSize) - local).distance < 20) {
                          _selectedVertexIndex = 0;
                        } else if (_activeDraftZone!.lineEnd != null && (_toCanvasOffset(_activeDraftZone!.lineEnd!, canvasSize) - local).distance < 20) {
                          _selectedVertexIndex = 1;
                        }
                      } else {
                        for (int i = 0; i < _activeDraftZone!.polygonPoints.length; i++) {
                          if ((_toCanvasOffset(_activeDraftZone!.polygonPoints[i], canvasSize) - local).distance < 20) {
                            _selectedVertexIndex = i;
                            break;
                          }
                        }
                      }
                    },
                    onPanUpdate: (details) => _handlePanUpdate(details, canvasSize),
                    onPanEnd: (_) => setState(() => _selectedVertexIndex = null),
                    child: CustomPaint(
                      size: canvasSize,
                      painter: ZoneCanvasPainter(
                        snapshotImage: null,
                        existingZones: _savedZones,
                        activeDraftZone: _activeDraftZone,
                        selectedVertexIndex: _selectedVertexIndex,
                        hoveredVertexIndex: _hoveredVertexIndex,
                        activeMousePosition: _activeMousePosition,
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildToolChip(ZoneType type, String label, IconData icon, Color color) {
    final isSelected = _selectedTool == type;
    return ChoiceChip(
      label: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: isSelected ? Colors.black : color),
          const SizedBox(width: 6),
          Text(label, style: TextStyle(fontSize: 11, color: isSelected ? Colors.black : Colors.white)),
        ],
      ),
      selected: isSelected,
      selectedColor: color,
      backgroundColor: AppTheme.darkBackground,
      onSelected: (val) {
        if (val) {
          setState(() {
            _selectedTool = type;
            if (_activeDraftZone != null) {
              _activeDraftZone!.zoneType = type;
            }
          });
        }
      },
    );
  }
}
