import 'package:flutter/material.dart';

import '../../app_state.dart';
import '../../data/models.dart';
import '../common.dart';

/// One round, updated live from `/rounds/{id}/events` until it finishes.
class RoundDetailScreen extends StatefulWidget {
  const RoundDetailScreen({
    super.key,
    required this.state,
    required this.initial,
  });

  final AppState state;
  final RoundInfo initial;

  @override
  State<RoundDetailScreen> createState() => _RoundDetailScreenState();
}

class _RoundDetailScreenState extends State<RoundDetailScreen> {
  late final Stream<RoundInfo>? _events = widget.initial.isFinished
      ? null
      : widget.state.api.roundEvents(
          widget.initial.roundId,
          widget.state.token ?? '',
        );

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Round ${widget.initial.roundId}')),
      body: StreamBuilder<RoundInfo>(
        stream: _events,
        initialData: widget.initial,
        builder: (context, snapshot) {
          final round = snapshot.data ?? widget.initial;
          final live = _events != null &&
              snapshot.connectionState == ConnectionState.active &&
              !round.isFinished;
          return ListView(
            padding: const EdgeInsets.all(16),
            children: [
              Field('state', round.state),
              Field('updates', '${round.submissions} of ${round.minClients}'),
              if (round.deadline != null)
                Field(
                  'deadline',
                  DateTime.fromMillisecondsSinceEpoch(
                    (round.deadline! * 1000).round(),
                    isUtc: true,
                  ).toIso8601String(),
                ),
              if (round.modelSha256 != null)
                Field('model SHA-256', round.modelSha256!, monospace: true),
              if (round.error != null) Field('error', round.error!),
              const SizedBox(height: 12),
              if (snapshot.hasError)
                Text('Live updates stopped: ${snapshot.error}')
              else
                Text(live ? 'Live: waiting for events' : 'Not live'),
            ],
          );
        },
      ),
    );
  }
}
