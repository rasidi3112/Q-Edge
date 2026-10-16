import numpy as np
import pytest

from qedge.fl.attacks import ATTACKS, Attack


def test_attackers_are_a_seeded_fixed_fraction():
    attack = Attack("label_flip", 0.3)

    assert len(attack.attackers(10, seed=0)) == 3
    assert attack.attackers(10, seed=0) == attack.attackers(10, seed=0)
    assert Attack("label_flip", 0.0).attackers(10, seed=0) == frozenset()


@pytest.mark.parametrize("name", ["label_flip", "scaling", "fake_samples"])
def test_poisoning_attacks_flip_labels(name):
    np.testing.assert_array_equal(Attack(name, 0.1).labels(np.array([0, 1, 2]), 3), [2, 1, 0])


def test_random_noise_keeps_labels_and_replaces_the_delta():
    attack = Attack("random_noise", 0.1, scale=2.0)
    delta = {"w": np.zeros(5000, np.float32)}

    sent = attack.delta(delta, 10, np.random.default_rng(0))

    np.testing.assert_array_equal(attack.labels(np.array([0, 1]), 2), [0, 1])
    assert np.std(sent["w"]) == pytest.approx(2.0, rel=0.05)
    assert sent["w"].dtype == np.float32


def test_scaling_boosts_the_delta_by_the_client_count_by_default():
    sent = Attack("scaling", 0.1).delta({"w": np.ones(3)}, 10, np.random.default_rng(0))

    np.testing.assert_allclose(sent["w"], 10.0)


def test_fake_samples_inflates_only_the_claimed_count():
    attack = Attack("fake_samples", 0.1)

    assert attack.claimed_samples(40, 10) == 4000
    assert Attack("label_flip", 0.1).claimed_samples(40, 10) == 40
    np.testing.assert_array_equal(attack.delta({"w": np.ones(2)}, 10, None)["w"], [1.0, 1.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "backdoor", "fraction": 0.1},
        {"name": "scaling", "fraction": 1.0},
        {"name": "label_flip", "fraction": 0.1, "scale": 2.0},
    ],
)
def test_invalid_attacks_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Attack(**kwargs)


def test_attack_names_are_listed():
    assert set(ATTACKS) == {"label_flip", "scaling", "random_noise", "fake_samples"}
