import 'package:flutter/material.dart';

import '../../app_state.dart';
import '../common.dart';

/// Server address (fixed at build time) and the bearer token.
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key, required this.state});

  final AppState state;

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  late final TextEditingController _token = TextEditingController(
    text: widget.state.token ?? '',
  );

  @override
  void dispose() {
    _token.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        Field('server', widget.state.api.baseUrl.toString()),
        const Text(
          'Set at build time: flutter run --dart-define=QEDGE_API_URL=http://host:8000',
        ),
        const Section('Bearer token'),
        TextField(
          controller: _token,
          obscureText: true,
          decoration: const InputDecoration(
            labelText: 'Client or admin token',
            border: OutlineInputBorder(),
          ),
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            FilledButton(
              onPressed: () {
                widget.state.setToken(_token.text);
                ScaffoldMessenger.of(
                  context,
                ).showSnackBar(const SnackBar(content: Text('Token set')));
              },
              child: const Text('Use token'),
            ),
            const SizedBox(width: 8),
            TextButton(
              onPressed: () {
                _token.clear();
                widget.state.setToken(null);
              },
              child: const Text('Clear'),
            ),
          ],
        ),
        const Section('What this app does'),
        const Text(
          'This app is a read-only dashboard. It shows server health, key '
          'fingerprints and round state over HTTP and server-sent events. It '
          'performs no cryptography and no training on any platform, including '
          'the web build: clients log in with ML-DSA and send encrypted updates '
          'from Python (qedge.fl.remote). The token is kept in memory only and '
          'is sent over plain HTTP unless the server URL uses HTTPS.',
        ),
      ],
    );
  }
}
