import dataclasses

import numpy as np
import pytest

from qedge.crypto.backends import get_kem, get_signer
from qedge.fl.secagg import (
    MaskedInput,
    Quantizer,
    SecAggError,
    SecAggServer,
    SecureAggregator,
    flatten,
)
from qedge.fl.strategies import fedavg

IDS = ["a", "b", "c", "d"]


def _aggregator(backend):
    return SecureAggregator(
        IDS,
        kem=get_kem("X25519-ML-KEM-768", backend),
        signer=get_signer("ML-DSA-65", backend),
    )


def _updates(seed=0):
    rng = np.random.default_rng(seed)
    return {
        cid: (
            {
                "w": rng.normal(size=(4, 3)).astype(np.float32),
                "b": rng.normal(size=3).astype(np.float32),
            },
            int(rng.integers(5, 60)),
        )
        for cid in IDS
    }


def _run_until_inputs(agg, round_id, updates):
    """Run the protocol by hand and return the server and the masked inputs."""
    server = SecAggServer(round_id, agg.signing_keys, signer=agg._signer, quantizer=agg.quantizer)
    broadcast = server.collect_advertisements([c.advertise(round_id) for c in agg.clients.values()])
    sent = [m for c in agg.clients.values() for m in c.share(broadcast, agg.signing_keys)]
    inbox = server.route(sent)
    names = ["w", "b"]
    inputs = []
    for cid, client in agg.clients.items():
        client.receive(inbox[cid], agg.signing_keys)
        weights, n = updates[cid]
        inputs.append(client.masked_input(weights, names, n))
    return server, inputs, names


def test_honest_sum_equals_fedavg_within_quantization_error(backend):
    agg = _aggregator(backend)
    updates = _updates()

    result = agg.aggregate(1, updates)

    expected = fedavg(list(updates.values()))
    step = 2.0**-agg.quantizer.frac_bits
    for name in expected:
        np.testing.assert_allclose(result[name], expected[name], atol=step)
        assert result[name].dtype == np.float32
    assert agg.last_cost.bytes_up_per_client > 0 and agg.last_cost.server_seconds > 0


def test_a_single_masked_input_reveals_nothing_about_its_update():
    agg = _aggregator("pure")
    updates = _updates()
    _, inputs, names = _run_until_inputs(agg, 2, updates)

    for item in inputs:
        weights, n = updates[item.client_id]
        truth = flatten(weights, names)
        decoded = agg.quantizer.decode(item.values[:-1], divisor=n)
        # Masks are uniform over the ring, so the decoded values are huge and uncorrelated.
        assert np.abs(decoded - truth).min() > 1e3
        assert abs(np.corrcoef(decoded, truth)[0, 1]) < 0.8
    # Any strict subset still carries uncancelled masks.
    partial = sum(i.values for i in inputs[:3]) & agg.quantizer.modulus_mask
    assert np.abs(agg.quantizer.decode(partial[:-1])).max() > 1e3


def test_the_same_update_is_masked_differently_each_round():
    agg = _aggregator("pure")
    updates = _updates()
    _, first, _ = _run_until_inputs(agg, 3, updates)
    _, second, _ = _run_until_inputs(agg, 4, updates)

    assert not np.array_equal(first[0].values, second[0].values)


def test_a_missing_client_fails_the_round():
    agg = _aggregator("pure")
    server, inputs, names = _run_until_inputs(agg, 5, _updates())

    with pytest.raises(SecAggError, match="dropouts are not supported"):
        server.aggregate(inputs[:-1], names, _updates()["a"][0])
    with pytest.raises(SecAggError, match="dropouts"):
        agg.aggregate(6, {k: v for k, v in _updates().items() if k != "a"})


def test_tampered_or_replayed_inputs_are_rejected():
    agg = _aggregator("pure")
    server, inputs, names = _run_until_inputs(agg, 7, _updates())
    like = _updates()["a"][0]

    tampered = dataclasses.replace(inputs[0], values=inputs[0].values ^ np.uint64(1))
    with pytest.raises(SecAggError, match="not validly signed"):
        server.aggregate([tampered, *inputs[1:]], names, like)

    replay = SecAggServer(8, agg.signing_keys, signer=agg._signer, quantizer=agg.quantizer)
    with pytest.raises(SecAggError, match="not validly signed"):
        replay.aggregate(inputs, names, like)
    with pytest.raises(SecAggError, match="duplicate"):
        server.aggregate([inputs[0], *inputs], names, like)


def test_clients_reject_a_kem_key_substituted_by_the_server():
    agg = _aggregator("pure")
    clients = list(agg.clients.values())
    ads = [c.advertise(9) for c in clients]
    forged_key, _ = get_kem("X25519-ML-KEM-768", "pure").generate_keypair()
    ads[1] = dataclasses.replace(ads[1], kem_public_key=forged_key)

    with pytest.raises(SecAggError, match="advertisement of 'b'"):
        clients[0].share(ads, agg.signing_keys)


def test_clients_reject_misaddressed_ciphertexts():
    agg = _aggregator("pure")
    clients = agg.clients
    ads = [c.advertise(10) for c in clients.values()]
    for_b = [m for m in clients["a"].share(ads, agg.signing_keys) if m.recipient == "b"]
    clients["c"].share(ads, agg.signing_keys)

    with pytest.raises(SecAggError, match="invalid pair ciphertext"):
        clients["c"].receive(for_b, agg.signing_keys)


def test_masking_needs_every_pair_secret():
    agg = _aggregator("pure")
    client = agg.clients["b"]
    ads = [c.advertise(11) for c in agg.clients.values()]
    client.share(ads, agg.signing_keys)

    with pytest.raises(SecAggError, match="no pair secret"):
        client.masked_input(_updates()["b"][0], ["w", "b"], 3)


def test_quantizer_round_trips_and_detects_overflow():
    q = Quantizer(bits=32, frac_bits=12, clip=4.0)
    values = np.array([-3.5, 0.0, 1.25, 10.0])

    np.testing.assert_allclose(q.decode(q.encode(values)), [-3.5, 0.0, 1.25, 4.0])
    np.testing.assert_allclose(
        q.decode(q.encode(values, weight=3), divisor=3), [-3.5, 0.0, 1.25, 4.0]
    )
    q.check_capacity(1000)
    with pytest.raises(ValueError, match="overflow"):
        q.check_capacity(200_000)
    with pytest.raises(ValueError):
        Quantizer(bits=8)


@pytest.mark.parametrize("bits", [16, 32, 64])
def test_decoding_keeps_small_negative_values_exact(bits):
    q = Quantizer(bits=bits, frac_bits=8, clip=10.0)
    values = np.array([-1 / 256, -9.0, 9.0, 0.5])

    np.testing.assert_array_equal(q.decode(q.encode(values)), values)


def test_masked_input_size_is_eight_bytes_per_value():
    assert MaskedInput("a", np.zeros(5, np.uint64), b"").nbytes == 40
