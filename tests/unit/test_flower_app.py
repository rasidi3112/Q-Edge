import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sklearn")
pytest.importorskip("flwr")

from flwr.app import MetricRecord, RecordDict

from qedge.fl.flower_app import (
    QEdgeStrategy,
    arrays_to_record,
    client_train,
    config_from_flower,
    record_to_arrays,
)
from qedge.fl.models import build_model, get_weights, set_weights
from qedge.fl.partition import dirichlet_partition, load_dataset
from qedge.fl.simulation import SimulationConfig, run_simulation
from qedge.fl.strategies import ServerStrategy
from qedge.fl.training import local_train


def test_array_records_round_trip():
    weights = {"w": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.ones(2, np.float32)}

    back = record_to_arrays(arrays_to_record(weights))

    assert back.keys() == weights.keys()
    for name in weights:
        np.testing.assert_array_equal(back[name], weights[name])


def test_run_config_uses_flower_key_names_and_rejects_unknown_keys():
    config = config_from_flower({"local-epochs": 2, "server-lr": 0.1, "clients": 3})
    assert (config.local_epochs, config.server_lr, config.clients) == (2, 0.1, 3)
    with pytest.raises(ValueError, match="Unknown run config keys"):
        config_from_flower({"num-rounds": 3})


def test_client_train_matches_the_simulator_client_step():
    config = SimulationConfig(clients=4, seed=2)
    data = load_dataset(config.dataset, seed=2)
    shard = dirichlet_partition(data.y_train, 4, config.alpha, seed=2)[1]
    model = build_model("logreg", data.n_features, data.n_classes, seed=2)
    start = get_weights(model)

    reply = client_train(RecordDict({"arrays": arrays_to_record(start)}), config, 1, 3)

    set_weights(model, start)
    loss = local_train(
        model,
        data.x_train[shard],
        data.y_train[shard],
        config.train_config(),
        seed=2 * 1_000_003 + 3 * 1_009 + 1,
    )
    expected = get_weights(model)
    received = record_to_arrays(reply["arrays"])
    for name in expected:
        np.testing.assert_array_equal(received[name], expected[name])
    assert reply["metrics"]["num-examples"] == len(shard)
    assert reply["metrics"]["train_loss"] == pytest.approx(loss)


def test_strategy_aggregates_with_the_server_strategy_and_keeps_its_state():
    start = {"w": np.zeros(3, np.float32)}
    updates = [({"w": np.full(3, 1.0, np.float32)}, 1), ({"w": np.full(3, 3.0, np.float32)}, 3)]
    contents = [
        RecordDict({"arrays": arrays_to_record(w), "metrics": MetricRecord({"num-examples": n})})
        for w, n in updates
    ]
    strategy = QEdgeStrategy(ServerStrategy("fedadam", lr=0.1))
    reference = ServerStrategy("fedadam", lr=0.1)

    with pytest.raises(RuntimeError, match="configure_train"):
        strategy.aggregate_contents(contents)
    strategy._current = start
    expected = start
    for _ in range(2):
        result = record_to_arrays(strategy.aggregate_contents(contents))
        expected = reference.step(expected, updates)
        np.testing.assert_allclose(result["w"], expected["w"])


@pytest.mark.slow
def test_flower_simulation_matches_the_light_loop():
    pytest.importorskip("ray")
    import flwr.simulation

    from qedge.fl import flower_app

    seen = {}
    original = flower_app.evaluate

    def recording_evaluate(model, x, y):
        accuracy, loss = original(model, x, y)
        seen["loss"] = loss
        return accuracy, loss

    flower_app.evaluate = recording_evaluate
    try:
        flwr.simulation.run_simulation(
            server_app=flower_app.server_app,
            client_app=flower_app.client_app,
            num_supernodes=10,
            backend_config={"client_resources": {"num_cpus": 1, "num_gpus": 0.0}},
        )
    finally:
        flower_app.evaluate = original

    expected = run_simulation(SimulationConfig())["history"][-1]["test_loss"]
    assert seen["loss"] == pytest.approx(expected, rel=1e-5)
