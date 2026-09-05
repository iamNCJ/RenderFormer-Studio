#!/usr/bin/env python3
"""Compare the structure and values of two HDF5 files."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def _describe(obj: h5py.Dataset | h5py.Group | None) -> str:
    if obj is None:
        return "MISSING"
    if isinstance(obj, h5py.Dataset):
        return f"{obj.shape} {obj.dtype}"
    return "Group"


def _compare_datasets(first: h5py.Dataset, second: h5py.Dataset) -> str:
    first_data = np.asarray(first)
    second_data = np.asarray(second)
    if np.array_equal(first_data, second_data):
        return "MATCH"
    if not (
        np.issubdtype(first_data.dtype, np.number)
        and np.issubdtype(second_data.dtype, np.number)
    ):
        return "DIFF (content)"
    difference = np.abs(first_data.astype(np.float64) - second_data.astype(np.float64))
    max_difference = float(difference.max(initial=0.0))
    if np.allclose(first_data, second_data):
        return f"CLOSE (max diff: {max_difference:.2e})"
    return f"DIFF (max diff: {max_difference:.2e})"


def compare_h5(first_path: str | Path, second_path: str | Path) -> int:
    """Print a comparison table and return the number of mismatches."""
    first_path = Path(first_path)
    second_path = Path(second_path)
    print(f"Comparing:\n1: {first_path}\n2: {second_path}\n", flush=True)

    mismatches = 0
    with h5py.File(first_path, "r") as first, h5py.File(second_path, "r") as second:
        first_keys: set[str] = set()
        second_keys: set[str] = set()
        first.visit(first_keys.add)
        second.visit(second_keys.add)

        print(f"{'Key':<50} {'File 1 (shape/type)':<30} {'File 2 (shape/type)':<30} Status")
        print("-" * 120)
        for key in sorted(first_keys | second_keys):
            first_obj = first.get(key)
            second_obj = second.get(key)
            first_info = _describe(first_obj)
            second_info = _describe(second_obj)
            if first_obj is None:
                status = "MISSING in 1"
            elif second_obj is None:
                status = "MISSING in 2"
            elif first_info != second_info:
                status = "MISMATCH (shape/type)"
            elif isinstance(first_obj, h5py.Dataset) and isinstance(second_obj, h5py.Dataset):
                status = _compare_datasets(first_obj, second_obj)
            else:
                status = "MATCH (group)"
            if not status.startswith(("MATCH", "CLOSE")):
                mismatches += 1
            print(f"{key:<50} {first_info:<30} {second_info:<30} {status}")

        print("\nRoot attributes:")
        for name in sorted(set(first.attrs) | set(second.attrs)):
            first_value = first.attrs.get(name, "MISSING")
            second_value = second.attrs.get(name, "MISSING")
            if not np.array_equal(first_value, second_value):
                mismatches += 1
                print(f"{name}:\n  1: {first_value}\n  2: {second_value}")
    return mismatches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args(argv)
    return int(compare_h5(args.first, args.second) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
