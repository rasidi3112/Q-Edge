import 'package:flutter/material.dart';

import 'app_state.dart';
import 'data/api_client.dart';
import 'features/rounds/rounds_screen.dart';
import 'features/settings/settings_screen.dart';
import 'features/status/status_screen.dart';

/// Server base URL, set with `--dart-define=QEDGE_API_URL=...`.
const apiUrl = String.fromEnvironment(
  'QEDGE_API_URL',
  defaultValue: 'http://localhost:8000',
);

void main() {
  runApp(QEdgeApp(state: AppState(HttpApiClient(Uri.parse(apiUrl)))));
}

class QEdgeApp extends StatelessWidget {
  const QEdgeApp({super.key, required this.state});

  final AppState state;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Q-Edge',
      theme: ThemeData(colorSchemeSeed: Colors.indigo, useMaterial3: true),
      home: HomeScreen(state: state),
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, required this.state});

  final AppState state;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  int _tab = 0;

  @override
  Widget build(BuildContext context) {
    final pages = [
      StatusScreen(api: widget.state.api),
      RoundsScreen(state: widget.state),
      SettingsScreen(state: widget.state),
    ];
    return Scaffold(
      appBar: AppBar(title: const Text('Q-Edge')),
      body: IndexedStack(index: _tab, children: pages),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _tab,
        onDestinationSelected: (index) => setState(() => _tab = index),
        destinations: const [
          NavigationDestination(icon: Icon(Icons.dns), label: 'Server'),
          NavigationDestination(icon: Icon(Icons.sync), label: 'Rounds'),
          NavigationDestination(icon: Icon(Icons.settings), label: 'Settings'),
        ],
      ),
    );
  }
}
