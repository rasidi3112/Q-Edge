import pytest

from qedge.crypto.backends import available_backends

LIBOQS_AVAILABLE = "liboqs" in available_backends()
requires_liboqs = pytest.mark.skipif(not LIBOQS_AVAILABLE, reason="liboqs not installed")


# The pure backend is required (CI installs q-edge[pqc-pure]); liboqs runs where present.
@pytest.fixture(params=["pure", pytest.param("liboqs", marks=requires_liboqs)])
def backend(request):
    return request.param
