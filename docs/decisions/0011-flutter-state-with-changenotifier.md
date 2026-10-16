# 0011. Flutter dashboard state with one ChangeNotifier

## Context

The old app kept all state in one 1 076-line `StatefulWidget`, filled it
with `Random`, `sin`/`cos` curves and `Future.delayed` "handshake" logs, and
called `setState` from timers without checking `mounted` (finding I7). The
new app only reads server state: health, key fingerprints, rounds and round
events. Its shared state is small: the API client and a bearer token.

## Decision

- One `AppState extends ChangeNotifier` holds the `ApiClient` and the
  token; screens rebuild with `ListenableBuilder`. No state-management
  package (provider, riverpod, bloc) is added.
- Requests are `Future`s rendered with `FutureBuilder`; round events are a
  `Stream` rendered with `StreamBuilder`. These builders drop results that
  arrive after their widget is gone, so no screen calls `setState` after an
  `await`.
- `ApiClient` is an interface. Production uses `HttpApiClient`
  (`package:http`), tests inject a fake. The base URL comes from
  `--dart-define=QEDGE_API_URL`.
- Errors are shown as errors (`Unreachable`), never replaced by data.

## Consequences

- One dependency (`http`). Moving to a state package later would touch
  only `AppState` and the builders.
- The token is not persisted; it has to be pasted again after a restart.
- Polling is replaced by server-sent events only on the round detail
  screen; the lists refresh on pull.
