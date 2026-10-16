import math

import numpy as np
import pytest

from qedge.fl.weights import (
    ModelSpec,
    TensorSpec,
    UpdateHeader,
    WeightFormatError,
    decode_update,
    encode_update,
)
from qedge.server.db import BlobStore

SPEC = ModelSpec((TensorSpec("w", (2,), "float32"),))
ARRAYS = {"w": np.array([1.0, 2.0], np.float32)}


def test_update_roundtrip_keeps_header_and_weights():
    header = UpdateHeader(round_id=7, n_samples=120, local_loss=0.25)

    decoded_header, arrays = decode_update(SPEC, encode_update(header, SPEC, ARRAYS))

    assert decoded_header == header
    np.testing.assert_array_equal(arrays["w"], ARRAYS["w"])


def test_update_rejects_bad_headers():
    with pytest.raises(WeightFormatError, match="not finite"):
        encode_update(UpdateHeader(1, 1, math.nan), SPEC, ARRAYS)
    with pytest.raises(WeightFormatError, match="out of range"):
        encode_update(UpdateHeader(1, 2**32, 0.0), SPEC, ARRAYS)
    with pytest.raises(WeightFormatError, match="QEU1"):
        decode_update(SPEC, b"QEW1" + bytes(64))


def test_blob_store_is_content_addressed_and_detects_corruption(tmp_path):
    blobs = BlobStore(tmp_path)
    digest = blobs.put(b"weights")

    assert blobs.put(b"weights") == digest
    assert blobs.get(digest) == b"weights"
    (tmp_path / digest).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="corrupted"):
        blobs.get(digest)
