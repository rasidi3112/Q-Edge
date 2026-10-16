"""Command-line entry point: ``qedge keys init`` and ``qedge keys show``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from qedge.crypto.backends import BACKENDS, get_kem, get_signer
from qedge.crypto.errors import CryptoError
from qedge.crypto.keys import (
    DEFAULT_KEM,
    DEFAULT_SIGNATURE,
    SERVER_KEMS,
    ServerKeys,
    fingerprint,
    generate_server_keys,
    load_server_keys,
)
from qedge.crypto.sign import SIGNATURE_ALGORITHMS


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qedge")
    commands = parser.add_subparsers(dest="command", required=True)
    keys = commands.add_parser("keys", help="manage server key files")
    actions = keys.add_subparsers(dest="action", required=True)
    backend_default = os.environ.get("QEDGE_CRYPTO_BACKEND", "liboqs")

    init = actions.add_parser("init", help="create server KEM and ML-DSA keys")
    init.add_argument("--dir", type=Path, required=True, help="key directory")
    init.add_argument("--backend", choices=BACKENDS, default=backend_default)
    init.add_argument("--kem", choices=SERVER_KEMS, default=DEFAULT_KEM)
    init.add_argument("--signature", choices=SIGNATURE_ALGORITHMS, default=DEFAULT_SIGNATURE)
    init.add_argument("--force", action="store_true", help="replace existing key files")

    show = actions.add_parser("show", help="check keys and print fingerprints")
    show.add_argument("--dir", type=Path, required=True, help="key directory")
    show.add_argument("--backend", choices=BACKENDS, default=backend_default)
    return parser


def _describe(keys: ServerKeys, directory: Path) -> str:
    rows = [
        ("key directory", str(directory)),
        ("kem", keys.kem_algorithm),
        ("kem sha256", fingerprint(keys.kem_public_key)),
        ("signature", keys.signature_algorithm),
        ("signing sha256", fingerprint(keys.signing_public_key)),
    ]
    return "\n".join(f"{label + ':':16}{value}" for label, value in rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and return the process exit code (0 on success, 1 on error)."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = _parser().parse_args(argv)
    try:
        if args.action == "init":
            keys = generate_server_keys(
                args.dir,
                kem=get_kem(args.kem, args.backend),
                signer=get_signer(args.signature, args.backend),
                overwrite=args.force,
            )
        else:
            keys = load_server_keys(args.dir, backend=args.backend)
    except CryptoError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(_describe(keys, args.dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
