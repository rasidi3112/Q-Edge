# Q-Edge dashboard

A read-only Flutter dashboard for a Q-Edge server. It shows:

- **Server**: `/health` and the KEM and ML-DSA key fingerprints from
  `/crypto/keys`. If the server cannot be reached or refuses a request, the
  screen says so; it never shows made-up data.
- **Rounds**: `/rounds` with a pasted bearer token, and per round the live
  state from the server-sent events of `/rounds/{id}/events`.
- **Settings**: the server URL (fixed at build time) and the token.

The app performs no cryptography and no training on any platform, the web
build included. Clients log in with ML-DSA and send encrypted updates from
Python (`qedge.fl.remote.RemoteClient`). On-device post-quantum crypto is not
implemented. The token is kept in memory only.

## Run

```bash
flutter pub get
flutter run -d chrome --web-port 3000 --dart-define=QEDGE_API_URL=http://localhost:8000
```

The server allows browser requests from `CORS_ORIGINS`
(`http://localhost:3000` by default), hence `--web-port 3000`. A token comes
from a Python client's `login()` (`RemoteClient.headers`).

## Layout

```
lib/
├── main.dart                     # app shell and navigation
├── app_state.dart                # ChangeNotifier: API client and token
├── data/
│   ├── api_client.dart           # ApiClient interface, HttpApiClient, SSE parser
│   └── models.dart               # Health, ServerKeys, RoundInfo
└── features/
    ├── common.dart               # shared widgets, Unreachable error view
    ├── rounds/                   # rounds list and live round detail
    ├── settings/                 # token entry
    └── status/                   # health and key fingerprints
```

State is a single `ChangeNotifier` read with `ListenableBuilder`
([ADR 0011](../docs/decisions/0011-flutter-state-with-changenotifier.md)).

## Test

```bash
flutter analyze
flutter test
```

Widget tests inject `test/fake_api.dart`, which answers from memory without
timers; `test/api_client_test.dart` checks the HTTP client against
`package:http/testing.dart`'s `MockClient`.
