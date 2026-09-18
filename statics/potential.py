"""Static effective-potential calculation and HDF5 persistence.

The potential formula itself lives in ``src.hamiltonian``.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

if __package__:
    from . import config
    from .data_io import load_arrays, save_arrays
else:  # Support ``cd statics && python potential.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from statics import config
    from statics.data_io import load_arrays, save_arrays

from src.hamiltonian import effective_potential

FORMAT = "nonuniform_B_2DEG_static_potential"
FORMAT_VERSION = 1


def compute_potential(x, ky_values, dc_potential):
    """Return ``V_eff(x)`` in meV for each ``ky`` in ``ky_values`` (au).

    Output has shape ``(len(ky_values), len(x))``.
    """
    return np.asarray([
        effective_potential(x, ky, dc_potential, hbar=config.HBAR,
            charge=config.ELEMENTARY_CHARGE, field=config.MAGNETIC_FIELD,
            mass=config.EFFECTIVE_MASS, length=config.MAGNETIC_LENGTH_SCALE)
        for ky in ky_values
    ]) * config.AU_TO_MEV


def configured_potential_metadata():
    """Return every setting needed to validate a cached potential file."""
    return {
        "ky_values_au": list(config.POTENTIAL_KY_AU),
        "dc_potential": config.POTENTIAL_V_DC,
        "x_points": config.POTENTIAL_X_POINTS,
        "hbar": config.HBAR,
        "elementary_charge": config.ELEMENTARY_CHARGE,
        "effective_mass": config.EFFECTIVE_MASS,
        "magnetic_field": config.MAGNETIC_FIELD,
        "magnetic_length_scale": config.MAGNETIC_LENGTH_SCALE,
        "cyclotron_frequency": config.CYCLOTRON_FREQUENCY,
    }


def _run_configured_potential():
    x = np.linspace(-np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE,
                    np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE,
                    config.POTENTIAL_X_POINTS)
    values = compute_potential(x, config.POTENTIAL_KY_AU, config.POTENTIAL_V_DC)
    return x, values


def load_configured_potential():
    """Load the saved potential data for the exact current configuration.

    Plotting callers use this function and deliberately never calculate the
    potential implicitly.
    """
    path = config.POTENTIAL_DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No potential data at {path}. Run `python potential.py` from "
            "statics/ or `python -m statics.potential` from the repository "
            "root first."
        )
    arrays, metadata = load_arrays(path, FORMAT, FORMAT_VERSION)
    if metadata != configured_potential_metadata():
        raise ValueError(
            f"Potential metadata in {path} does not match statics/config.py. "
            "Run `python potential.py --force` from statics/ or "
            "`python -m statics.potential --force` from the repository root."
        )
    return arrays["x_au"], arrays["potential_mev"]


def calculate_and_save_configured_potential(force=False):
    """Load a matching cached potential or calculate and persist it once."""
    if not force and config.POTENTIAL_DATA_FILE.exists():
        return (*load_configured_potential(), False)
    x, values = _run_configured_potential()
    save_arrays(config.POTENTIAL_DATA_FILE, FORMAT, FORMAT_VERSION,
        {"x_au": x, "potential_mev": values}, configured_potential_metadata())
    return x, values, True


def main(force=False):
    """Create or reuse the HDF5 potential data for :mod:`statics.config`."""
    x, values, calculated = calculate_and_save_configured_potential(force=force)
    action = "Saved" if calculated else "Reused"
    print(f"{action} {config.POTENTIAL_DATA_FILE}")
    print(f"x points: {x.size}; potential shape: {values.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the configured static effective potential.")
    parser.add_argument("--force", action="store_true", help="recompute and overwrite the saved potential data")
    try:
        main(force=parser.parse_args().force)
    except (FileNotFoundError, ValueError) as error:
        print(f"Cannot create potential data: {error}", file=sys.stderr)
        raise SystemExit(1)
