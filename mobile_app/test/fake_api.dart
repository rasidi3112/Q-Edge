import 'dart:async';

import 'package:q_edge_mobile/data/api_client.dart';
import 'package:q_edge_mobile/data/models.dart';

const healthFixture = Health(
  status: 'healthy',
  version: '0.2.0',
  services: {'database': 'ok'},
);
const keysFixture = ServerKeys(
  kemAlgorithm: 'X25519-ML-KEM-768',
  kemFingerprint: 'aa11',
  signatureAlgorithm: 'ML-DSA-65',
  signingFingerprint: 'bb22',
);
const openRound = RoundInfo(
  roundId: 7,
  state: 'open',
  minClients: 3,
  submissions: 1,
);

/// In-memory [ApiClient] whose answers the tests set; no timers, no I/O.
class FakeApiClient implements ApiClient {
  Object? failure;
  List<RoundInfo> roundList = [openRound];
  final events = StreamController<RoundInfo>();
  final tokens = <String>[];
  int healthCalls = 0;

  @override
  Uri get baseUrl => Uri.parse('http://test:8000');

  @override
  Future<Health> health() async {
    healthCalls++;
    if (failure != null) throw failure!;
    return healthFixture;
  }

  @override
  Future<ServerKeys> serverKeys() async => keysFixture;

  @override
  Future<List<RoundInfo>> rounds(String token) async {
    tokens.add(token);
    if (failure != null) throw failure!;
    return roundList;
  }

  @override
  Stream<RoundInfo> roundEvents(int roundId, String token) => events.stream;

  @override
  void close() => events.close();
}
