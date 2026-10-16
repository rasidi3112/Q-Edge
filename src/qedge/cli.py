"""Command-line entry point: ``qedge keys``, ``clients``, ``simulate`` and ``demo``.

Server and FL modules are imported inside their commands so that each
command needs only its own extra (``server`` or ``fl``).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from qedge.crypto.backends import BACKENDS, get_kem, get_signer
from qedge.crypto.errors import CryptoError, KeyStoreError
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
from qedge.protocol import SCOPES

FL_DATASETS = ("breast_cancer", "digits")
FL_MODELS = ("logreg", "mlp", "classical-head", "hybrid-vqc")
HEAD_MODELS = ("classical-head", "hybrid-vqc")
ANSATZE = ("strongly_entangling", "hardware_efficient")
ZNE_METHODS = ("richardson", "poly", "exponential")
ATTACKS = ("label_flip", "scaling", "random_noise", "fake_samples")
DEFENSES = ("median", "trimmed_mean", "krum", "multi_krum", "norm_clip", "bounded")
FL_STRATEGIES = ("fedavg", "fedprox", "fedadam", "fedyogi")


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

    clients = commands.add_parser("clients", help="manage client identities")
    client_actions = clients.add_subparsers(dest="action", required=True)
    keygen = client_actions.add_parser("keygen", help="create a client ML-DSA key pair")
    keygen.add_argument("--out", type=Path, required=True, help="directory for client.pub/.key")
    keygen.add_argument("--backend", choices=BACKENDS, default=backend_default)
    keygen.add_argument("--signature", choices=SIGNATURE_ALGORITHMS, default=DEFAULT_SIGNATURE)
    keygen.add_argument("--force", action="store_true", help="replace existing key files")

    add = client_actions.add_parser("add", help="enroll a client id with its public key")
    add.add_argument("--id", required=True, dest="client_id")
    add.add_argument("--public-key", type=Path, required=True, help="raw ML-DSA public key file")
    add.add_argument("--scope", choices=SCOPES, default="client")
    add.add_argument("--backend", choices=BACKENDS, default=backend_default)
    add.add_argument("--signature", choices=SIGNATURE_ALGORITHMS, default=DEFAULT_SIGNATURE)
    add.add_argument(
        "--database-url",
        default=os.environ.get("QEDGE_DATABASE_URL"),
        help="defaults to qedge.db in $QEDGE_DATA_DIR (qedge-data)",
    )

    sim = commands.add_parser("simulate", help="run an in-process FL simulation")
    sim.add_argument("--dataset", choices=FL_DATASETS, default="breast_cancer")
    sim.add_argument("--model", choices=FL_MODELS, default="logreg")
    sim.add_argument("--clients", type=int, default=10)
    sim.add_argument("--rounds", type=int, default=30)
    sim.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration")
    sim.add_argument("--strategy", choices=FL_STRATEGIES, default="fedavg")
    sim.add_argument("--seed", type=int, default=0)
    sim.add_argument("--local-epochs", type=int, default=1)
    sim.add_argument("--batch-size", type=int, default=16)
    sim.add_argument("--lr", type=float, default=0.1, help="client SGD learning rate")
    sim.add_argument("--mu", type=float, default=0.0, help="FedProx mu (0: 0.01 for fedprox)")
    sim.add_argument("--server-lr", type=float, default=0.05, help="FedAdam/FedYogi step")
    sim.add_argument("--compression", type=float, help="top-k ratio for update deltas")
    sim.add_argument("--dp-clip", type=float, help="L2 clip for update deltas")
    sim.add_argument("--dp-noise", type=float, default=0.0, help="Gaussian noise multiplier")
    sim.add_argument("--out", type=Path, default=Path("results/runs"), help="parent directory")
    sim.add_argument("--attack", choices=ATTACKS, help="malicious-client behaviour")
    sim.add_argument("--attack-fraction", type=float, default=0.0)
    sim.add_argument("--attack-scale", type=float, help="see qedge.fl.attacks")
    sim.add_argument("--defense", choices=DEFENSES, help="robust rule replacing FedAvg")
    sim.add_argument("--trim", type=float, dest="beta", help="trimmed_mean beta")
    sim.add_argument("--krum-f", type=int, dest="f", help="Byzantine clients Krum tolerates")
    sim.add_argument("--sample-cap", type=float, dest="cap", help="bounded: cap x median")
    sim.add_argument(
        "--secure-aggregation", action="store_true", help="pairwise-masked FedAvg (qedge.fl.secagg)"
    )
    sim.add_argument("--secagg-backend", choices=BACKENDS, default="pure")
    head = sim.add_argument_group("hybrid-vqc and classical-head options")
    head.add_argument("--qubits", type=int, dest="n_qubits")
    head.add_argument("--layers", type=int, dest="n_layers")
    head.add_argument("--ansatz", choices=ANSATZE)
    head.add_argument("--reupload", action="store_true", default=None, help="data re-uploading")
    head.add_argument("--noise", type=float, help="gate error probability (default.mixed)")
    head.add_argument("--noise-channel", choices=("depolarizing", "amplitude_damping"))
    head.add_argument("--readout", type=float, help="readout bit-flip probability")
    head.add_argument("--shots", type=int, help="finite shots (parameter-shift gradients)")
    head.add_argument("--zne", choices=ZNE_METHODS, help="zero-noise extrapolation")
    head.add_argument("--zne-scales", type=float, nargs="+", help="odd integers, e.g. 1 3 5")
    head.add_argument("--zne-order", type=int, help="order for --zne poly")
    head.add_argument("--encoder-hidden", type=int)

    demo = commands.add_parser("demo", help="run each component once and print the results")
    demo.add_argument(
        "--backend", choices=BACKENDS, help="crypto backend (default: liboqs if installed)"
    )
    return parser


HEAD_OPTIONS = (
    "n_qubits",
    "n_layers",
    "ansatz",
    "reupload",
    "noise",
    "noise_channel",
    "readout",
    "shots",
    "zne",
    "zne_scales",
    "zne_order",
    "encoder_hidden",
)


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
    if args.command == "clients":
        return _clients(args)
    if args.command == "simulate":
        return _simulate(args)
    if args.command == "demo":
        from qedge.demo import run_demo

        return run_demo(sys.stdout, args.backend)
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


def _clients(args: argparse.Namespace) -> int:
    from qedge.server.auth import ClientConflictError, enroll_client
    from qedge.server.db import Database, default_database_url

    if getattr(args, "database_url", "") is None:
        data_dir = Path(os.environ.get("QEDGE_DATA_DIR", "qedge-data"))
        args.database_url = default_database_url(data_dir)
    try:
        signer = get_signer(args.signature, args.backend)
        if args.action == "keygen":
            args.out.mkdir(parents=True, exist_ok=True, mode=0o700)
            public_path, secret_path = args.out / "client.pub", args.out / "client.key"
            if not args.force and (public_path.exists() or secret_path.exists()):
                raise KeyStoreError(f"{args.out} already holds client keys; pass --force")
            public_key, secret_key = signer.generate_keypair()
            secret_path.unlink(missing_ok=True)
            fd = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "wb") as handle:
                handle.write(secret_key)
            public_path.write_bytes(public_key)
            print(f"{'public key:':16}{public_path}\n{'fingerprint:':16}{fingerprint(public_key)}")
            return 0
        public_key = args.public_key.read_bytes()
        if args.database_url.startswith("sqlite:///"):
            Path(args.database_url.removeprefix("sqlite:///")).parent.mkdir(
                parents=True, exist_ok=True
            )
        database = Database(args.database_url)
        database.upgrade()
        with database.session() as session:
            created = enroll_client(session, args.client_id, public_key, args.scope, signer=signer)
        database.dispose()
    except (CryptoError, ClientConflictError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    verb = "enrolled" if created else "already enrolled"
    print(f"{verb} {args.client_id} ({args.scope}), key {fingerprint(public_key)}")
    return 0


def _simulate(args: argparse.Namespace) -> int:
    from qedge.fl.simulation import SimulationConfig, run_simulation

    options = {k: getattr(args, k) for k in HEAD_OPTIONS if getattr(args, k) is not None}
    if options and args.model not in HEAD_MODELS:
        print(f"error: {sorted(options)} only apply to {HEAD_MODELS}", file=sys.stderr)
        return 1
    try:
        config = SimulationConfig(
            dataset=args.dataset,
            model=args.model,
            clients=args.clients,
            rounds=args.rounds,
            alpha=args.alpha,
            strategy=args.strategy,
            seed=args.seed,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            mu=args.mu,
            server_lr=args.server_lr,
            compression=args.compression,
            dp_clip=args.dp_clip,
            dp_noise=args.dp_noise,
            model_options=options,
            attack=args.attack,
            attack_fraction=args.attack_fraction,
            attack_scale=args.attack_scale,
            defense=args.defense,
            secure_aggregation=args.secure_aggregation,
            secagg_backend=args.secagg_backend,
            defense_options={
                k: getattr(args, k) for k in ("beta", "f", "cap") if getattr(args, k) is not None
            },
        )
        summary = run_simulation(config, args.out)
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    rows = [
        ("run directory", summary["run_dir"]),
        ("parameters", summary["parameters"]),
        *([("head params", summary["model"]["head_parameters"])] if summary["model"] else []),
        ("final accuracy", f"{summary['final_accuracy']:.4f}"),
        ("best accuracy", f"{summary['best_accuracy']:.4f}"),
        ("bytes up", summary["bytes_up"]),
        ("bytes down", summary["bytes_down"]),
        ("seconds", f"{summary['seconds']:.2f}"),
    ]
    print("\n".join(f"{label + ':':16}{value}" for label, value in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
