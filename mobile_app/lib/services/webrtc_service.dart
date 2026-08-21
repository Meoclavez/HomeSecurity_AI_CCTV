import 'dart:async';
import 'dart:convert';
import 'package:flutter/foundation.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:http/http.dart' as http;
import '../core/constants/api_constants.dart';
import 'api_service.dart';

class WebRtcService {
  RTCPeerConnection? _peerConnection;
  final RTCVideoRenderer renderer = RTCVideoRenderer();
  MediaStream? _localAudioStream;
  MediaStreamTrack? _localAudioTrack;
  RTCRtpSender? _audioSender;

  bool isConnected = false;
  bool isTalkbackTransmitting = false;
  String? currentCameraId;
  String currentBaseUrl = ApiConstants.defaultBaseUrl;
  Timer? _reconnectTimer;
  int _reconnectAttempts = 0;
  static const int _maxReconnectAttempts = 5;

  Future<void> initialize() async {
    if (renderer.textureId == null) {
      await renderer.initialize();
    }
  }

  Future<Map<String, dynamic>> _fetchDynamicIceServers(String baseUrl) async {
    try {
      final response = await http
          .get(Uri.parse('$baseUrl/api/v1/webrtc/ice-servers'))
          .timeout(const Duration(seconds: 3));

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return {
          'iceServers': data['iceServers'],
          'sdpSemantics': 'unified-plan',
        };
      }
    } catch (e) {
      debugPrint('Fallback to default STUN servers: $e');
    }
    return ApiConstants.rtcIceServers;
  }

  Future<void> connect(String cameraId, {String? baseUrl, bool enableBackchannel = true}) async {
    currentCameraId = cameraId;
    currentBaseUrl = baseUrl ?? ApiService().baseUrl;
    _reconnectTimer?.cancel();

    await disconnect();

    final iceConfig = await _fetchDynamicIceServers(currentBaseUrl);
    _peerConnection = await createPeerConnection(iceConfig, ApiConstants.rtcMediaConstraints);

    _peerConnection!.onIceConnectionState = (RTCIceConnectionState state) {
      debugPrint('WebRTC ICE State for $cameraId: $state');
      if (state == RTCIceConnectionState.RTCIceConnectionStateDisconnected ||
          state == RTCIceConnectionState.RTCIceConnectionStateFailed) {
        _handleConnectionFailure();
      } else if (state == RTCIceConnectionState.RTCIceConnectionStateConnected) {
        _reconnectAttempts = 0;
        isConnected = true;
      }
    };

    _peerConnection!.onTrack = (RTCTrackEvent event) {
      if (event.streams.isNotEmpty && event.track.kind == 'video') {
        renderer.srcObject = event.streams[0];
      }
    };

    await _peerConnection!.addTransceiver(
      kind: RTCRtpMediaType.RTCRtpMediaTypeVideo,
      init: RTCRtpTransceiverInit(direction: TransceiverDirection.RecvOnly),
    );

    if (enableBackchannel) {
      final Map<String, dynamic> mediaConstraints = {
        'audio': {
          'echoCancellation': true,
          'noiseSuppression': true,
          'autoGainControl': true,
          'channelCount': 1,
          'sampleRate': 48000,
        },
        'video': false,
      };

      try {
        _localAudioStream = await navigator.mediaDevices.getUserMedia(mediaConstraints);
        _localAudioTrack = _localAudioStream!.getAudioTracks().first;
        _localAudioTrack!.enabled = false; // Start muted until PTT pressed
        _audioSender = await _peerConnection!.addTrack(_localAudioTrack!, _localAudioStream!);
      } catch (e) {
        debugPrint('Microphone init notice: $e');
        await _peerConnection!.addTransceiver(
          kind: RTCRtpMediaType.RTCRtpMediaTypeAudio,
          init: RTCRtpTransceiverInit(direction: TransceiverDirection.RecvOnly),
        );
      }
    }

    RTCSessionDescription offer = await _peerConnection!.createOffer(ApiConstants.rtcMediaConstraints);
    await _peerConnection!.setLocalDescription(offer);

    final response = await http.post(
      Uri.parse('$currentBaseUrl${ApiConstants.webrtcOfferEndpoint}'),
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'camera_id': cameraId,
        'sdp': offer.sdp,
        'type': 'offer',
      }),
    ).timeout(const Duration(seconds: 6));

    if (response.statusCode == 200) {
      final Map<String, dynamic> data = jsonDecode(response.body);
      final String answerSdp = data['sdp'] ?? '';
      await _peerConnection!.setRemoteDescription(RTCSessionDescription(answerSdp, 'answer'));
      isConnected = true;
    } else {
      throw Exception('WebRTC signaling rejected: HTTP ${response.statusCode}');
    }
  }

  void setTalkbackActive(bool active) {
    if (_localAudioTrack != null) {
      _localAudioTrack!.enabled = active;
      isTalkbackTransmitting = active;
    }
  }

  void _handleConnectionFailure() {
    isConnected = false;
    if (_reconnectAttempts >= _maxReconnectAttempts) {
      debugPrint('WebRTC maximum reconnection attempts reached for $currentCameraId');
      return;
    }

    final backoffSeconds = (1 << _reconnectAttempts);
    _reconnectAttempts++;
    debugPrint('Reconnecting WebRTC in ${backoffSeconds}s (attempt $_reconnectAttempts)');

    _reconnectTimer = Timer(Duration(seconds: backoffSeconds), () {
      if (currentCameraId != null) {
        connect(currentCameraId!, baseUrl: currentBaseUrl);
      }
    });
  }

  Future<void> disconnect() async {
    _reconnectTimer?.cancel();
    isConnected = false;
    isTalkbackTransmitting = false;

    if (_localAudioTrack != null) {
      await _localAudioTrack!.stop();
      _localAudioTrack = null;
    }
    if (_localAudioStream != null) {
      await _localAudioStream!.dispose();
      _localAudioStream = null;
    }
    if (renderer.srcObject != null) {
      for (var track in renderer.srcObject!.getTracks()) {
        await track.stop();
      }
      renderer.srcObject = null;
    }
    if (_peerConnection != null) {
      try {
        final transceivers = await _peerConnection!.transceivers;
        for (var t in transceivers) {
          await t.stop();
        }
        await _peerConnection!.close();
        await _peerConnection!.dispose();
      } catch (e) {
        debugPrint('PeerConnection disposal notice: $e');
      }
      _peerConnection = null;
    }
  }

  Future<void> dispose() async {
    await disconnect();
    await renderer.dispose();
  }
}
