import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:firebase_messaging/firebase_messaging.dart';

import 'core/theme/app_theme.dart';
import 'screens/app_shell.dart';
import 'services/notification_service.dart';
import 'services/api_service.dart';
import 'core/error_recovery.dart';

final GlobalKey<NavigatorState> navigatorKey = GlobalKey<NavigatorState>();

/// Top-level background message handler for FCM
@pragma('vm:entry-point')
Future<void> _firebaseMessagingBackgroundHandler(RemoteMessage message) async {
  await Firebase.initializeApp();
  debugPrint('Handling background message: ${message.messageId}');
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // 1. Initialize API Client and persistent base URL settings
  await ApiService().init();
  ConnectionMonitor().setBaseUrl(ApiService().baseUrl);
  ConnectionMonitor().startMonitoring();

  // 2. Initialize Firebase Core & Background Messaging
  try {
    await Firebase.initializeApp();
    FirebaseMessaging.onBackgroundMessage(_firebaseMessagingBackgroundHandler);
  } catch (e) {
    debugPrint('Firebase init notice (local development mode): $e');
  }

  // 3. Initialize Notification Channels & Native Audio
  await NotificationService().initialize(navigatorKey);

  runApp(const EdgeAiCctvApp());
}

class EdgeAiCctvApp extends StatelessWidget {
  const EdgeAiCctvApp({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Edge AI CCTV Surveillance',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.darkTheme,
      navigatorKey: navigatorKey,
      home: const AppShell(),
    );
  }
}
