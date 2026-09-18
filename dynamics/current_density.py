"""Stage 2: derive and persist the spatially resolved current density from a
saved trajectory.

This is the local line-current profile ``(J_x(x,t), J_y(x,t))``. Integrating
it over ``x`` (see :mod:`dynamics.current`) does *not* give an absolute
current for either component — see that module's docstring for exactly what
it does and does not mean.
"""

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
else:  # Support ``cd dynamics && python current_density.py``.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dynamics import config
    from dynamics.propagation import configured_run_metadata
    from dynamics.trajectory import load_trajectory

from src.basis import sho_basis_and_derivative


def reconstruct_wavefunction(states, x, *, mass, omega, hbar):
    """Reconstruct ``ψ`` and ``∂xψ`` from ``states`` of shape ``(nt,N)``.

    Returned complex arrays have shape ``(nt,len(x))``; all values use au.
    """
    basis, derivative = sho_basis_and_derivative(x, states.shape[-1], mass, omega, hbar)
    return states @ basis, states @ derivative


def line_current_density(psi, derivative, x, ky, *, hbar, charge, field, mass, length):
    """Return line current densities ``(Jx,Jy)`` in atomic units.

    ``Jx(x,t)`` is already the true current crossing position ``x`` (the
    ``y``-normalization exactly cancels for the confined coordinate: it
    equals ``L_y`` times the properly normalized 2D density, and integrating
    that over the periodic ``y`` direction reintroduces exactly one factor
    of ``L_y``), so it is not integrated further; ``Ix=∫Jx dx`` in
    :mod:`dynamics.current` is a diagnostic only, not a new physical
    quantity. ``Jy(x,t)`` is a genuine linear density in ``x`` with no such
    cancellation: ``Jy(x,t)/L_y`` is the properly normalized 2D density, so
    ``∫Jy(x,t)dx`` (computed in :mod:`dynamics.current` as ``iy_per_ly``) is
    the transverse current **per unit channel length L_y**, not an absolute
    current, because ``L_y`` never appears as an actual parameter in this
    single-``k_y``-channel model.
    """
    density = np.abs(psi) ** 2
    jx = -charge * hbar / mass * np.imag(np.conj(psi) * derivative)
    py = hbar * ky + charge * field * np.asarray(x)**2 / (2 * length)
    return jx, -charge * py * density / mass


def continuity_residual(psi, jx, times, x, charge):
    """Return ``∂t(-e|ψ|²)+∂xJx`` for ``(nt,nx)`` sampled arrays."""
    return (np.gradient(-charge * np.abs(psi)**2, times, axis=0, edge_order=2)
            + np.gradient(jx, x, axis=1, edge_order=2))


CURRENT_DENSITY_FORMAT = "nonuniform_B_2DEG_current_density"
CURRENT_DENSITY_FORMAT_VERSION = 1


def configured_current_density_grid():
    """Return the common reconstruction grid in atomic units."""
    return np.linspace(-np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE,
                       np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE,
                       config.SPATIAL_GRID_POINTS)


def configured_current_density_metadata():
    """Return metadata that identifies a current-density product unambiguously."""
    x = configured_current_density_grid()
    return {
        "trajectory_metadata": configured_run_metadata(),
        "spatial_grid_points": config.SPATIAL_GRID_POINTS,
        "x_min_au": float(x[0]),
        "x_max_au": float(x[-1]),
    }


def current_density_path_for_trajectory(trajectory_path, spatial_grid_points):
    """Return the default current-density filename derived from ``trajectory_path``."""
    trajectory_path = Path(trajectory_path)
    tag = trajectory_path.stem.removeprefix("trajectory_")
    return trajectory_path.with_name(f"current_density_{tag}_Nx{spatial_grid_points}.h5")


def resolve_trajectory_path(path):
    """Resolve a bare trajectory filename from the standard data directory.

    This keeps ``cd dynamics && python current_density.py trajectory_...h5``
    useful, while an explicitly relative or absolute path remains exactly as
    supplied.
    """
    path = Path(path)
    if path.exists() or path.parent != Path("."):
        return path
    standard_path = config.DATA_DIRECTORY / path
    return standard_path if standard_path.exists() else path


def current_density_grid(metadata, spatial_grid_points):
    """Return a reconstruction grid using the physical scale in trajectory metadata."""
    try:
        length = metadata["magnetic_length_scale"]
    except KeyError as error:
        raise ValueError(f"Trajectory metadata is missing {error.args[0]!r}") from error
    if spatial_grid_points < 2:
        raise ValueError("spatial_grid_points must be at least 2")
    return np.linspace(-np.sqrt(2) * length, np.sqrt(2) * length, spatial_grid_points)


def current_density_metadata(trajectory_metadata, x):
    """Return provenance metadata for a current-density product from any trajectory."""
    return {
        "trajectory_metadata": trajectory_metadata,
        "spatial_grid_points": int(x.size),
        "x_min_au": float(x[0]),
        "x_max_au": float(x[-1]),
    }


def _save_current_density(path, x, times, jx, jy, metadata):
    """Atomically save the current-density arrays and their provenance to HDF5."""
    path = Path(path)
    x = np.asarray(x, dtype=float)
    times = np.asarray(times, dtype=float)
    jx = np.asarray(jx, dtype=float)
    jy = np.asarray(jy, dtype=float)
    expected_shape = (times.size, x.size)
    if jx.shape != expected_shape or jy.shape != expected_shape:
        raise ValueError("jx and jy must both have shape (len(times), len(x))")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    chunks = (min(128, times.size), x.size)
    with h5py.File(temporary, "w") as handle:
        handle.attrs["format"] = CURRENT_DENSITY_FORMAT
        handle.attrs["format_version"] = CURRENT_DENSITY_FORMAT_VERSION
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        handle.create_dataset("x_au", data=x)
        handle.create_dataset("times_au", data=times)
        for name, values in (("jx_au", jx), ("jy_au", jy)):
            handle.create_dataset(name, data=values, chunks=chunks,
                                  compression="gzip", compression_opts=4, shuffle=True)
    temporary.replace(path)


def _load_current_density(path):
    """Load ``(x, times, jx, jy, metadata)`` from a current-density HDF5 file."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("format") != CURRENT_DENSITY_FORMAT:
            raise ValueError(f"{path} is not a nonuniform-B 2DEG current-density file")
        if handle.attrs.get("format_version") != CURRENT_DENSITY_FORMAT_VERSION:
            raise ValueError(f"Unsupported current-density format in {path}")
        metadata = json.loads(handle.attrs["metadata_json"])
        x = handle["x_au"][...]
        times = handle["times_au"][...]
        jx = handle["jx_au"][...]
        jy = handle["jy_au"][...]
    expected_shape = (times.size, x.size)
    if jx.shape != expected_shape or jy.shape != expected_shape:
        raise ValueError(f"Invalid current-density dataset shapes in {path}")
    return x, times, jx, jy, metadata


def load_configured_current_density():
    """Load current-density data for the exact configuration; never calculate it."""
    path = config.CURRENT_DENSITY_DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No current-density data at {path}. Run `python current_density.py` "
            "from dynamics/ or `python -m dynamics.current_density` from the "
            "repository root first."
        )
    x, times, jx, jy, metadata = _load_current_density(path)
    if metadata != configured_current_density_metadata():
        raise ValueError(
            f"Current-density metadata in {path} does not match dynamics/config.py. "
            "Run `python current_density.py --force` from dynamics/ or "
            "`python -m dynamics.current_density --force` from the repository root."
        )
    return x, times, jx, jy


def load_current_density_file(path):
    """Load any valid current-density HDF5 file without requiring the active config."""
    x, times, jx, jy, metadata = _load_current_density(path)
    return x, times, jx, jy, metadata


def calculate_and_save_current_density(trajectory_path, output_path=None, *, spatial_grid_points,
                                       force=False):
    """Derive current density from an arbitrary saved trajectory and persist it.

    Physics parameters are read from the trajectory's metadata, rather than
    from the active ``dynamics/config.py``.  Thus this can process an older or
    differently configured trajectory without rerunning dynamics.
    """
    trajectory_path = resolve_trajectory_path(trajectory_path)
    states, times, trajectory_metadata = load_trajectory(trajectory_path)
    try:
        basis_size = int(trajectory_metadata["basis_size"])
        if states.shape[1] != basis_size:
            raise ValueError("trajectory basis_size metadata disagrees with coefficient data")
        x = current_density_grid(trajectory_metadata, spatial_grid_points)
        metadata = current_density_metadata(trajectory_metadata, x)
        parameters = {
            name: trajectory_metadata[name]
            for name in ("effective_mass", "cyclotron_frequency", "hbar", "ky",
                         "elementary_charge", "magnetic_field", "magnetic_length_scale")
        }
    except KeyError as error:
        raise ValueError(f"Trajectory metadata is missing {error.args[0]!r}") from error

    output_path = (current_density_path_for_trajectory(trajectory_path, spatial_grid_points)
                   if output_path is None else Path(output_path))
    if output_path.exists() and not force:
        _, _, _, _, existing_metadata = _load_current_density(output_path)
        if existing_metadata != metadata:
            raise ValueError(
                f"Existing {output_path} was derived from different data or grid. "
                "Choose --output or use --force to replace it."
            )
        x, saved_times, jx, jy, _ = _load_current_density(output_path)
        return x, saved_times, jx, jy, output_path, False

    psi, derivative = reconstruct_wavefunction(
        states, x, mass=parameters["effective_mass"],
        omega=parameters["cyclotron_frequency"], hbar=parameters["hbar"],
    )
    jx, jy = line_current_density(
        psi, derivative, x, parameters["ky"], hbar=parameters["hbar"],
        charge=parameters["elementary_charge"], field=parameters["magnetic_field"],
        mass=parameters["effective_mass"], length=parameters["magnetic_length_scale"],
    )
    _save_current_density(output_path, x, times, jx, jy, metadata)
    return x, times, jx, jy, output_path, True


def calculate_and_save_configured_current_density(force=False):
    """Derive current-density arrays from the saved trajectory, then save them once."""
    if not force and config.CURRENT_DENSITY_DATA_FILE.exists():
        return (*load_configured_current_density(), False)

    x, times, jx, jy, _, calculated = calculate_and_save_current_density(
        config.TRAJECTORY_FILE, config.CURRENT_DENSITY_DATA_FILE,
        spatial_grid_points=config.SPATIAL_GRID_POINTS, force=force,
    )
    return x, times, jx, jy, calculated


def main(trajectory=None, output=None, spatial_grid_points=None, force=False):
    """Create current-density data from the configured or an explicitly chosen trajectory."""
    if trajectory is None and output is None and spatial_grid_points is None:
        x, times, jx, jy, calculated = calculate_and_save_configured_current_density(force=force)
        destination = config.CURRENT_DENSITY_DATA_FILE
    else:
        trajectory = config.TRAJECTORY_FILE if trajectory is None else Path(trajectory)
        spatial_grid_points = (config.SPATIAL_GRID_POINTS if spatial_grid_points is None
                               else spatial_grid_points)
        x, times, jx, jy, destination, calculated = calculate_and_save_current_density(
            trajectory, output, spatial_grid_points=spatial_grid_points, force=force)
    action = "Saved" if calculated else "Reused"
    print(f"{action} {destination}")
    print(f"current-density arrays: {jx.shape}; x points: {x.size}; time points: {times.size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate current density from a saved HDF5 trajectory.")
    parser.add_argument("trajectory", nargs="?", help="input trajectory HDF5 (default: configured trajectory)")
    parser.add_argument("--output", "-o", help="output current-density HDF5 (default: derived from trajectory name)")
    parser.add_argument("--nx", type=int, dest="spatial_grid_points",
                        help="number of x-grid points (default: config value)")
    parser.add_argument("--force", action="store_true", help="recompute and overwrite current-density data")
    try:
        arguments = parser.parse_args()
        main(arguments.trajectory, arguments.output, arguments.spatial_grid_points, arguments.force)
    except (FileNotFoundError, ValueError) as error:
        print(f"Cannot create current-density data: {error}", file=sys.stderr)
        raise SystemExit(1)
