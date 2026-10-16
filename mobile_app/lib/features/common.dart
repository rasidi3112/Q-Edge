import 'package:flutter/material.dart';

import '../data/api_client.dart';

/// A heading inside a list.
class Section extends StatelessWidget {
  const Section(this.title, {super.key});

  final String title;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.only(top: 16, bottom: 8),
        child: Text(title, style: Theme.of(context).textTheme.titleMedium),
      );
}

/// A label and its value.
class Field extends StatelessWidget {
  const Field(this.label, this.value, {super.key, this.monospace = false});

  final String label;
  final String value;
  final bool monospace;

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 4),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            SizedBox(width: 130, child: Text(label)),
            Expanded(
              child: SelectableText(
                value,
                style:
                    monospace ? const TextStyle(fontFamily: 'monospace') : null,
              ),
            ),
          ],
        ),
      );
}

/// Shown instead of data when a request fails; no made-up values.
class Unreachable extends StatelessWidget {
  const Unreachable({
    super.key,
    required this.baseUrl,
    required this.error,
    required this.onRetry,
  });

  final Uri baseUrl;
  final Object error;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) => Center(
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Icon(Icons.cloud_off, size: 48),
              const SizedBox(height: 12),
              Text(
                error is ApiException &&
                        (error as ApiException).statusCode != null
                    ? 'Server refused the request'
                    : 'Server unreachable',
                style: Theme.of(context).textTheme.titleLarge,
              ),
              const SizedBox(height: 8),
              Text('$baseUrl\n$error', textAlign: TextAlign.center),
              const SizedBox(height: 16),
              FilledButton(onPressed: onRetry, child: const Text('Retry')),
            ],
          ),
        ),
      );
}
