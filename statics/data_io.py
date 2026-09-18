"""Generic HDF5 persistence for static data products (spectrum, wavefunction,
potential), mirroring :mod:`dynamics.trajectory`'s save/load contract."""

import json
from pathlib import Path

import h5py
import numpy as np


def save_arrays(path, format_name, format_version, arrays, metadata):
    """Atomically save named float arrays and JSON metadata to HDF5."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(temporary, "w") as handle:
        handle.attrs["format"] = format_name
        handle.attrs["format_version"] = format_version
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        for name, values in arrays.items():
            handle.create_dataset(name, data=np.asarray(values, dtype=float))
    temporary.replace(path)


def load_arrays(path, format_name, format_version):
    """Load ``(arrays, metadata)`` from an HDF5 file saved by :func:`save_arrays`."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("format") != format_name:
            raise ValueError(f"{path} is not a {format_name!r} file")
        if handle.attrs.get("format_version") != format_version:
            raise ValueError(f"Unsupported data format in {path}")
        metadata = json.loads(handle.attrs["metadata_json"])
        arrays = {name: handle[name][...] for name in handle.keys()}
    return arrays, metadata
