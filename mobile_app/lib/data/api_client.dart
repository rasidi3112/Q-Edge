/// HTTP and server-sent-events access to the Q-Edge server.
library;

import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import 'models.dart';

/// A request failed: the server was unreachable or answered with an error.
class ApiException implements Exception {
  const ApiException(this.message, {this.statusCode});

  final String message;
  final int? statusCode;

  @override
  String toString() =>
      statusCode == null ? message : 'HTTP $statusCode: $message';
}

/// The calls the app makes. Widget tests inject a fake implementation.
abstract class ApiClient {
  /// Base URL the client talks to, for display.
  Uri get baseUrl;

  Future<Health> health();

  Future<ServerKeys> serverKeys();

  /// Rounds visible to the bearer [token] (a client or admin token).
  Future<List<RoundInfo>> rounds(String token);

  /// State changes of one round until it closes or fails.
  Stream<RoundInfo> roundEvents(int roundId, String token);

  void close();
}

/// [ApiClient] over `package:http`.
///
/// The app does no cryptography: it reads public endpoints and, with a
/// pasted bearer token, round state. Tokens come from the challenge-response
/// login of a Python client (`qedge.fl.remote.RemoteClient.login`).
class HttpApiClient implements ApiClient {
  HttpApiClient(this.baseUrl, {http.Client? client, Duration? timeout})
      : _client = client ?? http.Client(),
        _timeout = timeout ?? const Duration(seconds: 5);

  @override
  final Uri baseUrl;
  final http.Client _client;
  final Duration _timeout;

  Uri _url(String path) => baseUrl.resolve(path);

  Map<String, String> _auth(String token) => {
        'Authorization': 'Bearer $token',
      };

  Future<dynamic> _getJson(String path, {Map<String, String>? headers}) async {
    final http.Response response;
    try {
      response =
          await _client.get(_url(path), headers: headers).timeout(_timeout);
    } on TimeoutException {
      throw ApiException('no answer from $baseUrl within $_timeout');
    } on http.ClientException catch (error) {
      throw ApiException('cannot reach $baseUrl: ${error.message}');
    }
    if (response.statusCode != 200) {
      throw ApiException(response.body, statusCode: response.statusCode);
    }
    return jsonDecode(response.body);
  }

  @override
  Future<Health> health() async =>
      Health.fromJson(await _getJson('/health') as Map<String, dynamic>);

  @override
  Future<ServerKeys> serverKeys() async => ServerKeys.fromJson(
        await _getJson('/crypto/keys') as Map<String, dynamic>,
      );

  @override
  Future<List<RoundInfo>> rounds(String token) async {
    final body = await _getJson('/rounds', headers: _auth(token)) as List;
    return [
      for (final item in body) RoundInfo.fromJson(item as Map<String, dynamic>),
    ];
  }

  @override
  Stream<RoundInfo> roundEvents(int roundId, String token) async* {
    final request = http.Request('GET', _url('/rounds/$roundId/events'))
      ..headers.addAll({..._auth(token), 'Accept': 'text/event-stream'});
    final http.StreamedResponse response;
    try {
      response = await _client.send(request).timeout(_timeout);
    } on TimeoutException {
      throw ApiException('no answer from $baseUrl within $_timeout');
    } on http.ClientException catch (error) {
      throw ApiException('cannot reach $baseUrl: ${error.message}');
    }
    if (response.statusCode != 200) {
      throw ApiException(
        await response.stream.bytesToString(),
        statusCode: response.statusCode,
      );
    }
    yield* parseRoundEvents(response.stream.transform(utf8.decoder));
  }

  @override
  void close() => _client.close();
}

/// Parse `event: round` / `data: {...}` server-sent events into rounds.
///
/// Comment lines (`: keep-alive`) and other event types are skipped.
Stream<RoundInfo> parseRoundEvents(Stream<String> chunks) async* {
  var event = 'message';
  final data = StringBuffer();
  await for (final line in chunks.transform(const LineSplitter())) {
    if (line.isEmpty) {
      if (event == 'round' && data.isNotEmpty) {
        yield RoundInfo.fromJson(
          jsonDecode(data.toString()) as Map<String, dynamic>,
        );
      }
      event = 'message';
      data.clear();
    } else if (line.startsWith('event:')) {
      event = line.substring(6).trim();
    } else if (line.startsWith('data:')) {
      data.write(line.substring(5).trim());
    }
  }
}
