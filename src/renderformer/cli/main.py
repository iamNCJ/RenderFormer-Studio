"""Unified, lazy command dispatcher for ``python -m renderformer``."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable, Sequence
from types import ModuleType

from renderformer import __version__


_COMMAND_MODULES = {
    "infer": "renderformer.cli.infer",
    "train": "renderformer.training.cli",
    "data": "renderformer.data.cli.main",
}

_COMMAND_HELP = {
    "infer": "Render an RF1 or RF2 H5 scene.",
    "train": "Train an RF1/RF2 renderer or the material autoencoder.",
    "data": "Generate, export, postprocess, or validate scene data.",
}


def build_parser() -> argparse.ArgumentParser:
    """Build top-level help without importing a command implementation."""

    parser = argparse.ArgumentParser(
        prog="renderformer",
        description="RenderFormer data, training, and inference tools.",
        epilog=(
            "Run 'renderformer COMMAND --help' for command-specific options.\n"
            "The equivalent source-checkout form is "
            "'python -m renderformer COMMAND ...'."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    commands = parser.add_subparsers(dest="command", metavar="{infer,train,data}")
    for name in ("infer", "train", "data"):
        commands.add_parser(name, add_help=False, help=_COMMAND_HELP[name])
    return parser


def _load_command(name: str) -> Callable[[Sequence[str] | None], object]:
    module_name = _COMMAND_MODULES[name]
    module: ModuleType = importlib.import_module(module_name)
    command = getattr(module, "main")
    if not callable(command):
        raise TypeError(f"{module_name}.main is not callable")
    return command


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one subcommand while forwarding its arguments unchanged."""

    args = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    if not args:
        parser.print_help(sys.stderr)
        return 2
    if args[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    if args[0] == "--version":
        parser.parse_args(args)

    command_name = args[0]
    if command_name not in _COMMAND_MODULES:
        parser.error(
            f"argument command: invalid choice: {command_name!r} "
            f"(choose from {', '.join(repr(name) for name in _COMMAND_MODULES)})"
        )
    result = _load_command(command_name)(args[1:])
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
