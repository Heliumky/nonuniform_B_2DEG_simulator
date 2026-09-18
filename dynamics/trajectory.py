"""HDF5 persistence for driven coefficient trajectories."""

import json
from pathlib import Path

import h5py
import numpy as np


FORMAT_VERSION = 1


def save_trajectory(path, states, times, metadata):
    """Atomically save coefficient snapshots, times, and JSON metadata to HDF5."""
    path = Path(path)
    states = np.asarray(states, dtype=np.complex128)
    times = np.asarray(times, dtype=float)
    if states.ndim != 2:
        raise ValueError("states must have shape (number_of_snapshots, basis_size)")
    if times.ndim != 1 or times.size != states.shape[0]:
        raise ValueError("times must be one-dimensional with one value per state snapshot")
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("times must be finite and strictly increasing")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    chunk_rows = min(128, states.shape[0])
    with h5py.File(temporary, "w") as handle:
        handle.attrs["format"] = "nonuniform_B_2DEG_trajectory"
        handle.attrs["format_version"] = FORMAT_VERSION
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        handle.create_dataset("times_au", data=times)
        handle.create_dataset(
            "states_sho_coefficients", data=states, chunks=(chunk_rows, states.shape[1]),
            compression="gzip", compression_opts=4, shuffle=True,
        )
    temporary.replace(path)


def load_trajectory(path):
    """Load ``(states, times, metadata)`` from a trajectory HDF5 file."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("format") != "nonuniform_B_2DEG_trajectory":
            raise ValueError(f"{path} is not a nonuniform-B 2DEG trajectory file")
        if handle.attrs.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported trajectory format in {path}")
        metadata = json.loads(handle.attrs["metadata_json"])
        states = handle["states_sho_coefficients"][...]
        times = handle["times_au"][...]
    if states.ndim != 2 or times.ndim != 1 or states.shape[0] != times.size:
        raise ValueError(f"Invalid state/time dataset shapes in {path}")
    return states, times, metadata
