import 'package:flutter/material.dart';

import '../../app_state.dart';
import '../../data/models.dart';
import '../common.dart';
import 'round_detail_screen.dart';

/// Rounds visible to the pasted token.
class RoundsScreen extends StatefulWidget {
  const RoundsScreen({super.key, required this.state});

  final AppState state;

  @override
  State<RoundsScreen> createState() => _RoundsScreenState();
}

class _RoundsScreenState extends State<RoundsScreen> {
  Future<List<RoundInfo>>? _rounds;
  String? _loadedFor;

  void _reload() {
    final token = widget.state.token;
    setState(() {
      _loadedFor = token;
      _rounds = token == null ? null : widget.state.api.rounds(token);
    });
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: widget.state,
      builder: (context, _) {
        final token = widget.state.token;
        if (token == null) {
          return const Center(
            child: Padding(
              padding: EdgeInsets.all(24),
              child: Text(
                'Paste a client or admin token in Settings to list rounds.',
                textAlign: TextAlign.center,
              ),
            ),
          );
        }
        if (token != _loadedFor) {
          _loadedFor = token;
          _rounds = widget.state.api.rounds(token);
        }
        return FutureBuilder<List<RoundInfo>>(
          future: _rounds,
          builder: (context, snapshot) {
            if (snapshot.connectionState != ConnectionState.done) {
              return const Center(child: CircularProgressIndicator());
            }
            if (snapshot.hasError) {
              return Unreachable(
                baseUrl: widget.state.api.baseUrl,
                error: snapshot.error!,
                onRetry: _reload,
              );
            }
            final rounds = snapshot.data!;
            return RefreshIndicator(
              onRefresh: () async => _reload(),
              child: rounds.isEmpty
                  ? ListView(
                      children: const [
                        Padding(
                          padding: EdgeInsets.all(24),
                          child: Text('No rounds yet.'),
                        ),
                      ],
                    )
                  : ListView(
                      children: [
                        for (final round in rounds)
                          ListTile(
                            title: Text('Round ${round.roundId}'),
                            subtitle: Text(
                              '${round.state}: '
                              '${round.submissions}/${round.minClients} updates',
                            ),
                            trailing: const Icon(Icons.chevron_right),
                            onTap: () => Navigator.of(context).push(
                              MaterialPageRoute<void>(
                                builder: (_) => RoundDetailScreen(
                                  state: widget.state,
                                  initial: round,
                                ),
                              ),
                            ),
                          ),
                      ],
                    ),
            );
          },
        );
      },
    );
  }
}
