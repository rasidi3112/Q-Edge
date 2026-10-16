import 'package:flutter/material.dart';

import '../../data/api_client.dart';
import '../../data/models.dart';
import '../common.dart';

/// Server health and key fingerprints from the public endpoints.
class StatusScreen extends StatefulWidget {
  const StatusScreen({super.key, required this.api});

  final ApiClient api;

  @override
  State<StatusScreen> createState() => _StatusScreenState();
}

class _StatusScreenState extends State<StatusScreen> {
  late Future<(Health, ServerKeys)> _status = _load();

  Future<(Health, ServerKeys)> _load() async =>
      (await widget.api.health(), await widget.api.serverKeys());

  void _retry() {
    setState(() {
      _status = _load();
    });
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<(Health, ServerKeys)>(
      future: _status,
      builder: (context, snapshot) {
        if (snapshot.connectionState != ConnectionState.done) {
          return const Center(child: CircularProgressIndicator());
        }
        if (snapshot.hasError) {
          return Unreachable(
            baseUrl: widget.api.baseUrl,
            error: snapshot.error!,
            onRetry: _retry,
          );
        }
        final (health, keys) = snapshot.data!;
        return RefreshIndicator(
          onRefresh: () async => _retry(),
          child: ListView(
            padding: const EdgeInsets.all(16),
            children: [
              Section('Server ${widget.api.baseUrl}'),
              Field('status', health.status),
              Field('version', health.version),
              for (final entry in health.services.entries)
                Field(entry.key, entry.value),
              const Section('Server keys'),
              Field('KEM', keys.kemAlgorithm),
              Field('KEM SHA-256', keys.kemFingerprint, monospace: true),
              Field('signature', keys.signatureAlgorithm),
              Field('signing SHA-256', keys.signingFingerprint,
                  monospace: true),
              const SizedBox(height: 12),
              const Text(
                'Compare these fingerprints with `qedge keys show` on the server '
                'before trusting a client configuration.',
              ),
            ],
          ),
        );
      },
    );
  }
}
