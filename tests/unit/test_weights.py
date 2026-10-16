import pickle

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from qedge.fl.weights import (
    MAGIC,
    ModelSpec,
    TensorSpec,
    WeightFormatError,
    decode_weights,
    encode_weights,
)

SPEC = ModelSpec(
    (
        TensorSpec("encoder.weight", (3, 2), "float32"),
        TensorSpec("encoder.bias", (2,), "float32"),
        TensorSpec("head.scale", (), "float64"),
    )
)


def _arrays():
    rng = np.random.default_rng(0)
    return {
        "encoder.weight": rng.normal(size=(3, 2)).astype(np.float32),
        "encoder.bias": rng.normal(size=2).astype(np.float32),
        "head.scale": np.float64(0.5) * np.ones(()),
    }


def test_roundtrip_preserves_names_shapes_dtypes_and_values():
    arrays = _arrays()

    decoded = decode_weights(SPEC, encode_weights(SPEC, arrays))

    assert list(decoded) == [t.name for t in SPEC.tensors]
    for name, array in arrays.items():
        assert decoded[name].dtype == array.dtype
        np.testing.assert_array_equal(decoded[name], array)
        assert decoded[name].flags.writeable


def test_payload_size_is_header_plus_raw_tensors():
    assert len(encode_weights(SPEC, _arrays())) == 4 + 32 + 6 * 4 + 2 * 4 + 8 == SPEC.nbytes


@pytest.mark.parametrize(
    ("change", "match"),
    [
        (lambda a: a.pop("encoder.bias"), "names"),
        (lambda a: a.update(extra=np.zeros(1, np.float32)), "names"),
        (lambda a: a.update({"encoder.bias": np.zeros(3, np.float32)}), "shape"),
        (lambda a: a.update({"encoder.bias": np.zeros(2, np.float64)}), "dtype"),
        (lambda a: a.update({"encoder.bias": np.array([0.0, np.nan], np.float32)}), "NaN"),
    ],
)
def test_encode_rejects_arrays_that_do_not_match_the_spec(change, match):
    arrays = _arrays()
    change(arrays)

    with pytest.raises(WeightFormatError, match=match):
        encode_weights(SPEC, arrays)


def test_decode_rejects_payloads_for_another_spec_or_with_bad_framing():
    payload = encode_weights(SPEC, _arrays())
    other = ModelSpec((TensorSpec("w", (3, 2), "float32"), *SPEC.tensors[1:]))

    with pytest.raises(WeightFormatError, match="different model spec"):
        decode_weights(other, payload)
    with pytest.raises(WeightFormatError, match="bytes"):
        decode_weights(SPEC, payload[:-1])
    with pytest.raises(WeightFormatError, match="magic"):
        decode_weights(SPEC, b"XXXX" + payload[4:])


def test_decode_rejects_non_finite_values():
    payload = bytearray(encode_weights(SPEC, _arrays()))
    payload[36:40] = np.array([np.inf], np.float32).tobytes()

    with pytest.raises(WeightFormatError, match="NaN or infinity"):
        decode_weights(SPEC, bytes(payload))


def test_decode_never_unpickles():
    payload = pickle.dumps({"encoder.weight": np.zeros((3, 2), np.float32)})

    with pytest.raises(WeightFormatError):
        decode_weights(SPEC, payload)


def test_spec_validation():
    with pytest.raises(ValueError, match="unique"):
        ModelSpec((TensorSpec("w", (1,), "float32"), TensorSpec("w", (2,), "float32")))
    with pytest.raises(ValueError, match="dtype"):
        TensorSpec("w", (1,), "int64")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="negative"):
        TensorSpec("w", (-1,), "float32")


def test_spec_from_arrays_matches_explicit_spec():
    assert ModelSpec.from_arrays(_arrays()) == SPEC


@given(st.binary(max_size=200))
def test_arbitrary_bytes_only_raise_weight_format_error(data):
    with pytest.raises(WeightFormatError):
        decode_weights(SPEC, data)


@given(st.binary(min_size=SPEC.nbytes - 36, max_size=SPEC.nbytes - 36))
def test_well_framed_random_tensor_bytes_decode_or_raise_weight_format_error(body):
    payload = MAGIC + SPEC.digest() + body
    try:
        decoded = decode_weights(SPEC, payload)
    except WeightFormatError:
        return
    assert all(np.all(np.isfinite(array)) for array in decoded.values())
