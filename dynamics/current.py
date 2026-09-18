"""Stage 3: derive and persist the x-integrated current from a saved
current-density file.

``current_density.py`` gives the spatially resolved line current density
``(J_x(x,t), J_y(x,t))``. Neither integral over ``x`` computed here is an
*absolute* current in the sense of amperes for a specific device:

- ``J_x(x,t)`` is already the true current crossing position ``x`` (see
  :mod:`dynamics.current_density`): a well-defined, ``L_y``-independent
  quantity by itself. ``Ix(t)=∫J_x(x,t)dx`` therefore does **not** correspond
  to a new physical quantity — it sums together current values at different
  positions, which is not standard. It is kept only as the same diagnostic
  previously computed ad hoc by ``plot_current_density.py --compare``.

- ``J_y(x,t)`` is a genuine linear density in ``x``. Writing the full
  wavefunction as ``Ψ(x,y,t)=ψ(x,t)e^{ik_yy}/sqrt(L_y)``, the properly
  normalized 2D current density is ``J_y(x,t)/L_y``, so the true transverse
  current crossing a full-width line is ``(1/L_y)∫J_y(x,t)dx`` — **not**
  ``∫J_y(x,t)dx`` alone (dimensional check: ``J_y`` already has units of
  charge/time, so integrating it over ``x`` gives charge·length/time, not a
  current). This module saves ``∫J_y(x,t)dx`` as ``iy_per_ly_au`` — the
  transverse current **per unit channel length L_y** (equivalently, its
  value at ``L_y=1`` a.u.), not an absolute current. ``L_y`` is never a
  parameter of this single-``k_y``-channel model (see the exclusion noted at
  the top of the repository README), so there is no way to convert this to
  an absolute current here; that would require summing over the occupied
  ``k_y`` Fermi sea, where the ``L_y`` dependence cancels against the
  density of states.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

if __package__:
    from . import config
    from .current_density import configured_current_density_metadata, load_current_density_file
else:  # Support ``cd dynamics && python current.py``.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dynamics import config
    from dynamics.current_density import configured_current_density_metadata, load_current_density_file


def integrate_current(x, jx, jy):
    """Integrate line current densities over ``x``.

    ``jx`` and ``jy`` have shape ``(nt,nx)``; returns ``(ix,iy_per_ly)`` of
    shape ``(nt,)``. ``ix`` has no established physical meaning (diagnostic
    only). ``iy_per_ly`` is the transverse current per unit channel length
    ``L_y``, not an absolute current — see the module docstring.
    """
    ix = np.trapezoid(jx, x, axis=1)
    iy_per_ly = np.trapezoid(jy, x, axis=1)
    return ix, iy_per_ly


CURRENT_FORMAT = "nonuniform_B_2DEG_current"
CURRENT_FORMAT_VERSION = 1


def current_path_for_current_density(current_density_path):
    """Return the default current filename derived from ``current_density_path``."""
    current_density_path = Path(current_density_path)
    tag = current_density_path.stem.removeprefix("current_density_")
    return current_density_path.with_name(f"current_{tag}.h5")


def resolve_current_density_path(path):
    """Resolve a bare current-density filename from the standard data directory.

    This keeps ``cd dynamics && python current.py current_density_...h5``
    useful, while an explicitly relative or absolute path remains exactly as
    supplied.
    """
    path = Path(path)
    if path.exists() or path.parent != Path("."):
        return path
    standard_path = config.DATA_DIRECTORY / path
    return standard_path if standard_path.exists() else path


def current_metadata(current_density_metadata):
    """Return provenance metadata for a current product from any current-density file."""
    return {"current_density_metadata": current_density_metadata}


def configured_current_metadata():
    """Return metadata that identifies a current product unambiguously."""
    return current_metadata(configured_current_density_metadata())


def _save_current(path, times, ix, iy_per_ly, metadata):
    """Atomically save the current arrays and their provenance to HDF5."""
    path = Path(path)
    times = np.asarray(times, dtype=float)
    ix = np.asarray(ix, dtype=float)
    iy_per_ly = np.asarray(iy_per_ly, dtype=float)
    if ix.shape != times.shape or iy_per_ly.shape != times.shape:
        raise ValueError("ix and iy_per_ly must both have the same shape as times")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with h5py.File(temporary, "w") as handle:
        handle.attrs["format"] = CURRENT_FORMAT
        handle.attrs["format_version"] = CURRENT_FORMAT_VERSION
        handle.attrs["metadata_json"] = json.dumps(metadata, sort_keys=True)
        handle.create_dataset("times_au", data=times)
        handle.create_dataset("ix_au", data=ix)
        # Transverse current per unit channel length L_y, NOT an absolute
        # current -- see the module docstring. Deliberately not named
        # "iy_au" to avoid that misreading.
        handle.create_dataset("iy_per_ly_au", data=iy_per_ly)
    temporary.replace(path)


def _load_current(path):
    """Load ``(times, ix, iy_per_ly, metadata)`` from a current HDF5 file."""
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("format") != CURRENT_FORMAT:
            raise ValueError(f"{path} is not a nonuniform-B 2DEG current file")
        if handle.attrs.get("format_version") != CURRENT_FORMAT_VERSION:
            raise ValueError(f"Unsupported current format in {path}")
        metadata = json.loads(handle.attrs["metadata_json"])
        times = handle["times_au"][...]
        ix = handle["ix_au"][...]
        iy_per_ly = handle["iy_per_ly_au"][...]
    if ix.shape != times.shape or iy_per_ly.shape != times.shape:
        raise ValueError(f"Invalid current dataset shapes in {path}")
    return times, ix, iy_per_ly, metadata


def load_configured_current():
    """Load current data for the exact configuration; never calculate it."""
    path = config.CURRENT_DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No current data at {path}. Run `python current.py` from dynamics/ "
            "or `python -m dynamics.current` from the repository root first."
        )
    times, ix, iy_per_ly, metadata = _load_current(path)
    if metadata != configured_current_metadata():
        raise ValueError(
            f"Current metadata in {path} does not match dynamics/config.py. "
            "Run `python current.py --force` from dynamics/ or "
            "`python -m dynamics.current --force` from the repository root."
        )
    return times, ix, iy_per_ly


def load_current_file(path):
    """Load any valid current HDF5 file without requiring the active config."""
    times, ix, iy_per_ly, metadata = _load_current(path)
    return times, ix, iy_per_ly, metadata


def calculate_and_save_current(current_density_path, output_path=None, *, force=False):
    """Derive the current from an arbitrary saved current-density file.

    This never reconstructs the wavefunction or reruns propagation; it only
    reads the already-saved line current density and integrates it over
    ``x``, so it works directly on data produced by an existing
    ``current_density.py`` run. See the module docstring for what ``ix`` and
    ``iy_per_ly`` do (and do not) mean physically.
    """
    current_density_path = resolve_current_density_path(current_density_path)
    x, times, jx, jy, current_density_metadata = load_current_density_file(current_density_path)
    metadata = current_metadata(current_density_metadata)

    output_path = (current_path_for_current_density(current_density_path)
                   if output_path is None else Path(output_path))
    if output_path.exists() and not force:
        _, _, _, existing_metadata = _load_current(output_path)
        if existing_metadata != metadata:
            raise ValueError(
                f"Existing {output_path} was derived from different current-density data. "
                "Choose --output or use --force to replace it."
            )
        saved_times, ix, iy_per_ly, _ = _load_current(output_path)
        return saved_times, ix, iy_per_ly, output_path, False

    ix, iy_per_ly = integrate_current(x, jx, jy)
    _save_current(output_path, times, ix, iy_per_ly, metadata)
    return times, ix, iy_per_ly, output_path, True


def calculate_and_save_configured_current(force=False):
    """Derive the current from the saved current-density file, then save it once."""
    if not force and config.CURRENT_DATA_FILE.exists():
        return (*load_configured_current(), False)

    times, ix, iy_per_ly, _, calculated = calculate_and_save_current(
        config.CURRENT_DENSITY_DATA_FILE, config.CURRENT_DATA_FILE, force=force,
    )
    return times, ix, iy_per_ly, calculated


def main(current_density=None, output=None, force=False):
    """Create current data from the configured or an explicitly chosen current-density file."""
    if current_density is None and output is None:
        times, ix, iy_per_ly, calculated = calculate_and_save_configured_current(force=force)
        destination = config.CURRENT_DATA_FILE
    else:
        current_density = (config.CURRENT_DENSITY_DATA_FILE if current_density is None
                           else Path(current_density))
        times, ix, iy_per_ly, destination, calculated = calculate_and_save_current(
            current_density, output, force=force)
    action = "Saved" if calculated else "Reused"
    print(f"{action} {destination}")
    print(f"current arrays: {ix.shape}; time points: {times.size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Integrate a saved HDF5 current-density file over x (Ix diagnostic; Iy per unit Ly).")
    parser.add_argument("current_density", nargs="?",
                        help="input current-density HDF5 (default: configured current-density data)")
    parser.add_argument("--output", "-o", help="output current HDF5 (default: derived from current-density name)")
    parser.add_argument("--force", action="store_true", help="recompute and overwrite current data")
    try:
        arguments = parser.parse_args()
        main(arguments.current_density, arguments.output, arguments.force)
    except (FileNotFoundError, ValueError) as error:
        print(f"Cannot create current data: {error}", file=sys.stderr)
        raise SystemExit(1)
