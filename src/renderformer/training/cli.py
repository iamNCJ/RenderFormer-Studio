"""Public dispatcher for renderer and material-autoencoder training runners."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Sequence
from types import ModuleType


_RUNNER_MODULES = {
    "rf1": "renderformer.training.train_rf1",
    "rf2": "renderformer.training.train_rf2",
    "material": "renderformer.training.train_material",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the lightweight top-level parser without importing ML packages."""

    parser = argparse.ArgumentParser(
        prog="renderformer train",
        description=(
            "Train an RF1/RF2 renderer or the material autoencoder. Arguments "
            "after the runner name are forwarded unchanged to that runner."
        ),
        epilog=(
            "Examples:\n"
            "  renderformer train rf1 --config_file configs/model/v1/200m_full_attention.yaml\n"
            "  renderformer train rf2 --config_file "
            "configs/model/v2/200m_sliding_sink_summary.yaml\n"
            "  renderformer train material --model-config "
            "configs/model/material/autoencoder.yaml --help\n"
            "  python -m torch.distributed.run --standalone "
            "--nproc-per-node 8 --module renderformer.training.cli rf2 "
            "--config_file CONFIG.yaml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="version",
        metavar="{rf1,rf2,material}",
    )
    subparsers.add_parser(
        "rf1",
        add_help=False,
        help="Run the RF1 renderer training loop.",
    )
    subparsers.add_parser(
        "rf2",
        add_help=False,
        help="Run the RF2 multi-resolution training loop.",
    )
    subparsers.add_parser(
        "material",
        add_help=False,
        help="Run checkpoint-compatible material-autoencoder training.",
    )
    return parser


def _load_runner(version: str) -> Callable[[], object]:
    module: ModuleType = importlib.import_module(_RUNNER_MODULES[version])
    runner = getattr(module, "main")
    if not callable(runner):
        raise TypeError(f"{_RUNNER_MODULES[version]}.main is not callable")
    return runner


def main(argv: Sequence[str] | None = None) -> int:
    """Select a version and invoke its runner in the current process.

    The training runners historically read their Tyro arguments from
    ``sys.argv``. The dispatcher removes only the ``rf1``/``rf2`` selector,
    forwards every remaining token unchanged, and restores ``sys.argv`` when
    the runner returns or raises.
    """

    args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not args:
        parser.print_help(sys.stderr)
        return 2
    if args[0] in {"-h", "--help"}:
        parser.print_help()
        return 0

    version = args[0]
    if version not in _RUNNER_MODULES:
        parser.error(
            f"argument version: invalid choice: {version!r} "
            "(choose from 'rf1', 'rf2', 'material')"
        )

    runner = _load_runner(version)
    original_argv = sys.argv
    sys.argv = [f"{parser.prog} {version}", *args[1:]]
    try:
        result = runner()
    finally:
        sys.argv = original_argv
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
