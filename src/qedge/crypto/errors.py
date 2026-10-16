"""Exceptions raised by `qedge.crypto`."""

from __future__ import annotations


class CryptoError(Exception):
    """Base class for errors raised by `qedge.crypto`."""


class BackendUnavailableError(CryptoError):
    """The requested backend or algorithm cannot be used in this environment."""


class InvalidKeyError(CryptoError, ValueError):
    """A key has the wrong length or encoding for its algorithm."""


class InvalidCiphertextError(CryptoError, ValueError):
    """A KEM ciphertext has the wrong length or encoding."""


class KeyStoreError(CryptoError):
    """Key files are missing, unreadable, too permissive or inconsistent."""
