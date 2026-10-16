/// JSON models for the Q-Edge server API (`src/qedge/server/app.py`).
library;

/// `GET /health`.
class Health {
  const Health({
    required this.status,
    required this.version,
    required this.services,
  });

  factory Health.fromJson(Map<String, dynamic> json) => Health(
        status: json['status'] as String,
        version: json['version'] as String,
        services: Map<String, String>.from(json['services'] as Map),
      );

  final String status;
  final String version;
  final Map<String, String> services;
}

/// `GET /crypto/keys`: algorithm names and SHA-256 fingerprints.
class ServerKeys {
  const ServerKeys({
    required this.kemAlgorithm,
    required this.kemFingerprint,
    required this.signatureAlgorithm,
    required this.signingFingerprint,
  });

  factory ServerKeys.fromJson(Map<String, dynamic> json) => ServerKeys(
        kemAlgorithm: json['kem_algorithm'] as String,
        kemFingerprint: json['kem_fingerprint'] as String,
        signatureAlgorithm: json['signature_algorithm'] as String,
        signingFingerprint: json['signing_fingerprint'] as String,
      );

  final String kemAlgorithm;
  final String kemFingerprint;
  final String signatureAlgorithm;
  final String signingFingerprint;
}

/// One round from `GET /rounds` or a `round` event of `/rounds/{id}/events`.
class RoundInfo {
  const RoundInfo({
    required this.roundId,
    required this.state,
    required this.minClients,
    required this.submissions,
    this.deadline,
    this.modelSha256,
    this.error,
  });

  factory RoundInfo.fromJson(Map<String, dynamic> json) => RoundInfo(
        roundId: json['round_id'] as int,
        state: json['state'] as String,
        minClients: json['min_clients'] as int,
        submissions: json['submissions'] as int,
        deadline: (json['deadline'] as num?)?.toDouble(),
        modelSha256: json['model_sha256'] as String?,
        error: json['error'] as String?,
      );

  final int roundId;
  final String state;
  final int minClients;
  final int submissions;
  final double? deadline;
  final String? modelSha256;
  final String? error;

  bool get isFinished => state == 'closed' || state == 'failed';
}
