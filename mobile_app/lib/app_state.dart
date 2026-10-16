/// Shared state: the API client and the bearer token the user pasted.
library;

import 'package:flutter/foundation.dart';

import 'data/api_client.dart';

/// Holds what several screens need. Screens listen with `ListenableBuilder`;
/// this is the app's only state-management mechanism (docs/decisions/0011).
class AppState extends ChangeNotifier {
  AppState(this.api);

  final ApiClient api;
  String? _token;

  /// Bearer token for `/rounds`; kept in memory only, never written to disk.
  String? get token => _token;

  void setToken(String? value) {
    final trimmed = value?.trim();
    _token = (trimmed == null || trimmed.isEmpty) ? null : trimmed;
    notifyListeners();
  }
}
