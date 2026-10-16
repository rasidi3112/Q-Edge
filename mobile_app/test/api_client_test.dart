import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:q_edge_mobile/data/api_client.dart';

void main() {
  final base = Uri.parse('http://server:8000');

  test('health and rounds are parsed and rounds send the bearer token',
      () async {
    final seen = <String, String?>{};
    final client = HttpApiClient(
      base,
      client: MockClient((request) async {
        seen[request.url.path] = request.headers['Authorization'];
        if (request.url.path == '/health') {
          return http.Response(
            jsonEncode({
              'status': 'healthy',
              'version': '0.2.0',
              'services': {'database': 'ok'}
            }),
            200,
          );
        }
        return http.Response(
          jsonEncode([
            {
              'round_id': 1,
              'state': 'closed',
              'min_clients': 2,
              'submissions': 2,
              'deadline': null,
              'model_sha256': 'ab',
              'error': null
            },
          ]),
          200,
        );
      }),
    );

    expect((await client.health()).services['database'], 'ok');
    final rounds = await client.rounds('tok');
    expect(rounds.single.isFinished, isTrue);
    expect(seen['/rounds'], 'Bearer tok');
    expect(seen['/health'], isNull);
  });

  test('HTTP errors and connection failures become ApiException', () async {
    final refusing = HttpApiClient(
      base,
      client: MockClient((_) async => http.Response('{"detail":"no"}', 401)),
    );
    await expectLater(
      refusing.rounds('x'),
      throwsA(
          isA<ApiException>().having((e) => e.statusCode, 'statusCode', 401)),
    );

    final offline = HttpApiClient(
      base,
      client: MockClient((_) async => throw http.ClientException('refused')),
    );
    await expectLater(
      offline.health(),
      throwsA(isA<ApiException>()
          .having((e) => e.statusCode, 'statusCode', isNull)),
    );
  });

  test('server-sent events yield rounds and skip comments', () async {
    const body = ': keep-alive\n\n'
        'event: round\n'
        'data: {"round_id": 3, "state": "open", "min_clients": 2, "submissions": 0}\n\n'
        'event: other\ndata: {}\n\n'
        'event: round\n'
        'data: {"round_id": 3, "state": "closed", "min_clients": 2, "submissions": 2}\n\n';

    final states = await parseRoundEvents(
      Stream.fromIterable([body.substring(0, 40), body.substring(40)]),
    ).map((r) => r.state).toList();

    expect(states, ['open', 'closed']);
  });
}
