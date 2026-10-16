import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:q_edge_mobile/app_state.dart';
import 'package:q_edge_mobile/data/api_client.dart';
import 'package:q_edge_mobile/data/models.dart';
import 'package:q_edge_mobile/main.dart';

import 'fake_api.dart';

Future<(FakeApiClient, AppState)> pumpApp(
  WidgetTester tester, {
  Object? failure,
}) async {
  final api = FakeApiClient()..failure = failure;
  final state = AppState(api);
  await tester.pumpWidget(QEdgeApp(state: state));
  await tester.pumpAndSettle();
  return (api, state);
}

void main() {
  testWidgets('server tab shows health and key fingerprints', (tester) async {
    await pumpApp(tester);

    expect(find.text('healthy'), findsOneWidget);
    expect(find.text('X25519-ML-KEM-768'), findsOneWidget);
    expect(find.text('aa11'), findsOneWidget);
    expect(find.text('bb22'), findsOneWidget);
  });

  testWidgets('an unreachable server shows an error, not data', (
    tester,
  ) async {
    final (api, _) = await pumpApp(
      tester,
      failure: const ApiException('cannot reach http://test:8000'),
    );

    expect(find.text('Server unreachable'), findsOneWidget);
    expect(find.text('healthy'), findsNothing);

    api.failure = null;
    await tester.tap(find.text('Retry'));
    await tester.pumpAndSettle();
    expect(find.text('healthy'), findsOneWidget);
    expect(api.healthCalls, 2);
  });

  testWidgets('rounds need a token and then come from the API', (
    tester,
  ) async {
    final (api, state) = await pumpApp(tester);
    await tester.tap(find.text('Rounds'));
    await tester.pumpAndSettle();
    expect(
        find.textContaining('Paste a client or admin token'), findsOneWidget);

    state.setToken('  secret-token  ');
    await tester.pumpAndSettle();

    expect(api.tokens, ['secret-token']);
    expect(find.text('Round 7'), findsOneWidget);
    expect(find.text('open: 1/3 updates'), findsOneWidget);
  });

  testWidgets('a refused token is reported as a refusal', (tester) async {
    final (api, state) = await pumpApp(tester);
    await tester.tap(find.text('Rounds'));
    api.failure = const ApiException('invalid token', statusCode: 401);
    state.setToken('bad');
    await tester.pumpAndSettle();

    expect(find.text('Server refused the request'), findsOneWidget);
  });

  testWidgets('round detail follows server-sent events until closed', (
    tester,
  ) async {
    final (api, state) = await pumpApp(tester);
    state.setToken('t');
    await tester.tap(find.text('Rounds'));
    await tester.pumpAndSettle();
    await tester.tap(find.text('Round 7'));
    await tester.pumpAndSettle();
    expect(find.text('1 of 3'), findsOneWidget);

    api.events.add(
      const RoundInfo(
          roundId: 7, state: 'aggregating', minClients: 3, submissions: 3),
    );
    await tester.pumpAndSettle();
    expect(find.text('3 of 3'), findsOneWidget);
    expect(find.text('Live: waiting for events'), findsOneWidget);

    api.events.add(
      const RoundInfo(
        roundId: 7,
        state: 'closed',
        minClients: 3,
        submissions: 3,
        modelSha256: 'cafe',
      ),
    );
    await api.events.close();
    await tester.pumpAndSettle();
    expect(find.text('closed'), findsOneWidget);
    expect(find.text('cafe'), findsOneWidget);
    expect(find.text('Not live'), findsOneWidget);
  });

  testWidgets('settings set and clear the token', (tester) async {
    final (_, state) = await pumpApp(tester);
    await tester.tap(find.text('Settings'));
    await tester.pumpAndSettle();

    await tester.enterText(find.byType(TextField), 'abc');
    await tester.tap(find.text('Use token'));
    await tester.pump();
    expect(state.token, 'abc');

    await tester.tap(find.text('Clear'));
    await tester.pump();
    expect(state.token, isNull);
    expect(find.textContaining('performs no cryptography'), findsOneWidget);
  });
}
