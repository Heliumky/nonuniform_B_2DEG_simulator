"""Stage 2a: derive and persist probability density from a saved trajectory."""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

if __package__:
    from . import config
    from .propagation import configured_run_metadata
    from .trajectory import load_trajectory
else:  # Support direct execution from the dynamics directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dynamics import config
    from dynamics.propagation import configured_run_metadata
    from dynamics.trajectory import load_trajectory

from src.basis import sho_basis


DENSITY_FORMAT = "nonuniform_B_2DEG_density"
DENSITY_FORMAT_VERSION = 1


def configured_density_grid():
    """Return the common wavefunction-reconstruction grid in atomic units."""
    return np.linspace(-np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE,
                       np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE,
                       config.SPATIAL_GRID_POINTS)


def configured_density_metadata():
    """Return provenance metadata for a configured density product."""
    x = configured_density_grid()
    return {
        "trajectory_metadata": configured_run_metadata(),
        "spatial_grid_points": config.SPATIAL_GRID_POINTS,
        "x_min_au": float(x[0]),
        "x_max_au": float(x[-1]),
    }


def density_path_for_trajectory(trajectory_path, spatial_grid_points):
    """Return the default density filename derived from ``trajectory_path``."""
    trajectory_path = Path(trajectory_path)
    tag = trajectory_path.stem.removeprefix("trajectory_")
    return trajectory_path.with_name(f"density_{tag}_Nx{spatial_grid_points}.h5")


def resolve_trajectory_path(path):
    """Resolve a bare trajectory filename from the standard data directory."""
    path = Path(path)
    if path.exists() or path.parent != Path("."):
        return path
    standard_path = config.DATA_DIRECTORY / path
    return standard_path if standard_path.exists() else path


def density_grid(metadata, spatial_grid_points):
    """Return a reconstruction grid using the trajectory's length scale."""
    try:
        length = metadata["magnetic_length_scale"]
    except KeyError as error:
        raise ValueError(f"Trajectory metadata is missing {error.args[0]!r}") from error
    if spatial_grid_points < 2:
        raise ValueError("spatial_grid_points must be at least 2")
    return np.linspace(-np.sqrt(2) * length, np.sqrt(2) * length, spatial_grid_points)


def density_metadata(trajectory_metadata, x):
    """Return provenance metadata for a density product from any trajectory."""
    return {
        "trajectory_metadata": trajectory_metadata,
        "spatial_grid_points": int(x.size),
        "x_min_au": float(x[0]),
        "x_max_au": float(x[-1]),
    }


def _save_density(path, x, times, density, metadata):
    """Atomically save density snapshots and provenance to HDF5."""
    path = Path(path)
    x = np.asarray(x, dtype=float)
    times = np.asarray(times, dtype=float)
    density = np.asarray(density, dtype=float)
    if density.shape != (times.size, x.size):
        raise ValueError("density must have shape (len(times), len(x))")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(temporary, "w") as handle:
        handle.attrs["format"] = DENSITY_FORMAT
        handle.attrs["format_version"] = DENSITY_FORMAT_VERSION
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        handle.create_dataset("x_au", data=x)
        handle.create_dataset("times_au", data=times)
        handle.create_dataset("density_au", data=density,
                              chunks=(min(128, times.size), x.size),
                              compression="gzip", compression_opts=4, shuffle=True)
    temporary.replace(path)


def _load_density(path):
    """Load (x, times, density, metadata) from a density HDF5 file."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("format") != DENSITY_FORMAT:
            raise ValueError(f"{path} is not a nonuniform-B 2DEG density file")
        if handle.attrs.get("format_version") != DENSITY_FORMAT_VERSION:
            raise ValueError(f"Unsupported density-data format in {path}")
        metadata = json.loads(handle.attrs["metadata_json"])
        x = handle["x_au"][...]
        times = handle["times_au"][...]
        density = handle["density_au"][...]
    if density.shape != (times.size, x.size):
        raise ValueError(f"Invalid density dataset shape in {path}")
    return x, times, density, metadata


def load_configured_density():
    """Load configured density data only; never reconstruct it implicitly."""
    path = config.DENSITY_DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No density data at {path}. Run python density.py from dynamics/ "
            "or python -m dynamics.density from the repository root first."
        )
    x, times, density, metadata = _load_density(path)
    if metadata != configured_density_metadata():
        raise ValueError(
            f"Density metadata in {path} does not match dynamics/config.py. "
            "Run python density.py --force from dynamics/ or "
            "python -m dynamics.density --force from the repository root."
        )
    return x, times, density


def load_density_file(path):
    """Load any valid density HDF5 file without requiring active config."""
    return _load_density(path)


def calculate_and_save_density(trajectory_path, output_path=None, *, spatial_grid_points,
                               force=False):
    """Reconstruct and save density for an arbitrary trajectory HDF5 file."""
    trajectory_path = resolve_trajectory_path(trajectory_path)
    states, times, trajectory_metadata = load_trajectory(trajectory_path)
    try:
        basis_size = int(trajectory_metadata["basis_size"])
        if states.shape[1] != basis_size:
            raise ValueError("trajectory basis_size metadata disagrees with coefficient data")
        x = density_grid(trajectory_metadata, spatial_grid_points)
        metadata = density_metadata(trajectory_metadata, x)
        mass = trajectory_metadata["effective_mass"]
        omega = trajectory_metadata["cyclotron_frequency"]
        hbar = trajectory_metadata["hbar"]
    except KeyError as error:
        raise ValueError(f"Trajectory metadata is missing {error.args[0]!r}") from error

    output_path = (density_path_for_trajectory(trajectory_path, spatial_grid_points)
                   if output_path is None else Path(output_path))
    if output_path.exists() and not force:
        _, _, _, existing_metadata = _load_density(output_path)
        if existing_metadata != metadata:
            raise ValueError(
                f"Existing {output_path} was derived from different data or grid. "
                "Choose --output or use --force to replace it."
            )
        x, saved_times, density, _ = _load_density(output_path)
        return x, saved_times, density, output_path, False

    basis = sho_basis(x, basis_size, mass, omega, hbar)
    density = np.abs(states @ basis) ** 2
    _save_density(output_path, x, times, density, metadata)
    return x, times, density, output_path, True


def calculate_and_save_configured_density(force=False):
    """Reconstruct density from trajectory data and save it once."""
    if not force and config.DENSITY_DATA_FILE.exists():
        return (*load_configured_density(), False)
    x, times, density, _, calculated = calculate_and_save_density(
        config.TRAJECTORY_FILE, config.DENSITY_DATA_FILE,
        spatial_grid_points=config.SPATIAL_GRID_POINTS, force=force,
    )
    return x, times, density, calculated


def main(trajectory=None, output=None, spatial_grid_points=None, force=False):
    """Create density data from the configured or an explicitly chosen trajectory."""
    if trajectory is None and output is None and spatial_grid_points is None:
        x, times, density, calculated = calculate_and_save_configured_density(force=force)
        destination = config.DENSITY_DATA_FILE
    else:
        trajectory = config.TRAJECTORY_FILE if trajectory is None else Path(trajectory)
        spatial_grid_points = (config.SPATIAL_GRID_POINTS if spatial_grid_points is None
                               else spatial_grid_points)
        x, times, density, destination, calculated = calculate_and_save_density(
            trajectory, output, spatial_grid_points=spatial_grid_points, force=force)
    action = "Saved" if calculated else "Reused"
    print(f"{action} {destination}")
    print(f"density array: {density.shape}; x points: {x.size}; time points: {times.size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate density from a saved HDF5 trajectory.")
    parser.add_argument("trajectory", nargs="?", help="input trajectory HDF5 (default: configured trajectory)")
    parser.add_argument("--output", "-o", help="output density HDF5 (default: derived from trajectory name)")
    parser.add_argument("--nx", type=int, dest="spatial_grid_points",
                        help="number of x-grid points (default: config value)")
    parser.add_argument("--force", action="store_true", help="recompute and overwrite density data")
    try:
        arguments = parser.parse_args()
        main(arguments.trajectory, arguments.output, arguments.spatial_grid_points, arguments.force)
    except (FileNotFoundError, ValueError) as error:
        print(f"Cannot create density data: {error}", file=sys.stderr)
        raise SystemExit(1)
