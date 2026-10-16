# 0008. One FL implementation, two runners

## Context

Experiments need many short runs (strategies × α × seeds) and must finish in
seconds on a laptop. Flower's simulation engine starts Ray, builds an app
bundle and an isolated environment per run (about 20 s overhead for 30
rounds of logistic regression) and is not installable everywhere Ray is not.
The project should still run with the current Flower API (`ClientApp`,
`ServerApp`, `flwr run`) so that the code can move to real nodes. The old
code had a Flower client with a different training loop and an aggregator
the server did not use, so results depended on which path ran.

## Decision

- Training (`qedge.fl.training`), aggregation (`qedge.fl.strategies`),
  partitioning and models are plain functions and classes with no Flower
  dependency.
- `qedge.fl.simulation.run_simulation` is the in-process runner used by
  `qedge simulate` and `experiments/`.
- `qedge.fl.flower_app` is thin glue: the client handler calls the same
  `local_train` on the same Dirichlet shard with the same per-round seed, and
  `QEdgeStrategy` only replaces Flower FedAvg's `aggregate_train` with
  `ServerStrategy.step`. A slow test checks that both runners end with the
  same test loss.
- FedProx is a client loss term, not a server rule. SCAFFOLD is not
  implemented. Clipping and noise exist, but no ε is reported because there
  is no accountant.

## Consequences

- Numbers in `results/` come from the light runner. They also hold for
  `flwr run` as long as the equivalence test passes.
- The light runner has no client sampling, dropouts or stragglers; every
  client trains every round.
- `flwr run experiments/flower` needs `flwr[simulation]` (the `flower-sim`
  extra) and a local SuperLink entry with `num-supernodes` equal to the
  `clients` run-config value.
