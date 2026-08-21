enum ZoneType { intrusion, tripwire, privacyMask, door, package }
enum MaskMode { blackout, blur, mosaic, color }
enum TripwireDirection { aToB, bToA, bidirectional }

class Point2D {
  double x;
  double y;
  Point2D({required this.x, required this.y});

  Map<String, dynamic> toJson() => {'x': x, 'y': y};
  factory Point2D.fromJson(Map<String, dynamic> json) => Point2D(
        x: (json['x'] as num).toDouble(),
        y: (json['y'] as num).toDouble(),
      );

  Point2D clone() => Point2D(x: x, y: y);
}

class ZoneConfig {
  String id;
  String cameraId;
  String name;
  ZoneType zoneType;
  bool enabled;
  List<Point2D> polygonPoints;
  Point2D? lineStart;
  Point2D? lineEnd;
  TripwireDirection direction;
  MaskMode maskMode;
  double dwellTimeSeconds;
  List<String> allowedClasses;

  ZoneConfig({
    required this.id,
    required this.cameraId,
    required this.name,
    required this.zoneType,
    this.enabled = true,
    List<Point2D>? polygonPoints,
    this.lineStart,
    this.lineEnd,
    this.direction = TripwireDirection.bidirectional,
    this.maskMode = MaskMode.blackout,
    this.dwellTimeSeconds = 0.0,
    List<String>? allowedClasses,
  })  : polygonPoints = polygonPoints ?? [],
        allowedClasses = allowedClasses ?? ['person', 'car', 'package'];
}
