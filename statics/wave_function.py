"""Static eigenstate probability-density calculation and HDF5 persistence."""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

if __package__:
    from . import config
    from .data_io import load_arrays, save_arrays
else:  # Support ``cd statics && python wave_function.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from statics import config
    from statics.data_io import load_arrays, save_arrays

from src.basis import sho_basis
from src.hamiltonian import build_hamiltonian
from src.state_index import validate_state_index

FORMAT = "nonuniform_B_2DEG_static_wavefunction"
FORMAT_VERSION = 1


def eigenstate_probability(ky, dc_potential, state_index, basis_size, x):
    """Return static ``|ψ_n(x)|²``; ``n=0`` is the ground state."""
    state_index = validate_state_index(state_index, basis_size)
    matrix = build_hamiltonian(ky, dc_potential, basis_size, hbar=config.HBAR,
        charge=config.ELEMENTARY_CHARGE, field=config.MAGNETIC_FIELD,
        mass=config.EFFECTIVE_MASS, length=config.MAGNETIC_LENGTH_SCALE,
        omega=config.CYCLOTRON_FREQUENCY)
    vector = eigh(matrix)[1][:, state_index]
    return np.abs(vector @ sho_basis(x, basis_size, config.EFFECTIVE_MASS,
                                     config.CYCLOTRON_FREQUENCY, config.HBAR)) ** 2


def configured_wavefunction_metadata():
    """Return every setting needed to validate a cached wavefunction file."""
    return {
        "ky_nm": config.WAVEFUNCTION_KY_NM,
        "dc_potential": config.WAVEFUNCTION_V_DC,
        "state_index": config.WAVEFUNCTION_STATE_INDEX,
        "basis_size": config.WAVEFUNCTION_BASIS_SIZE,
        "x_points": config.WAVEFUNCTION_X_POINTS,
        "hbar": config.HBAR,
        "elementary_charge": config.ELEMENTARY_CHARGE,
        "effective_mass": config.EFFECTIVE_MASS,
        "magnetic_field": config.MAGNETIC_FIELD,
        "magnetic_length_scale": config.MAGNETIC_LENGTH_SCALE,
        "cyclotron_frequency": config.CYCLOTRON_FREQUENCY,
    }


def _run_configured_wavefunction():
    x = np.linspace(-np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE,
                    np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE,
                    config.WAVEFUNCTION_X_POINTS)
    probability = eigenstate_probability(config.WAVEFUNCTION_KY_NM * config.AU_TO_NM,
        config.WAVEFUNCTION_V_DC, config.WAVEFUNCTION_STATE_INDEX,
        config.WAVEFUNCTION_BASIS_SIZE, x)
    return x, probability


def load_configured_wavefunction():
    """Load the saved wavefunction data for the exact current configuration.

    Plotting callers use this function and deliberately never calculate the
    wavefunction implicitly.
    """
    path = config.WAVEFUNCTION_DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No wavefunction data at {path}. Run `python wave_function.py` from "
            "statics/ or `python -m statics.wave_function` from the repository "
            "root first."
        )
    arrays, metadata = load_arrays(path, FORMAT, FORMAT_VERSION)
    if metadata != configured_wavefunction_metadata():
        raise ValueError(
            f"Wavefunction metadata in {path} does not match statics/config.py. "
            "Run `python wave_function.py --force` from statics/ or "
            "`python -m statics.wave_function --force` from the repository root."
        )
    return arrays["x_au"], arrays["probability_au"]


def calculate_and_save_configured_wavefunction(force=False):
    """Load a matching cached wavefunction or calculate and persist it once."""
    if not force and config.WAVEFUNCTION_DATA_FILE.exists():
        return (*load_configured_wavefunction(), False)
    x, probability = _run_configured_wavefunction()
    save_arrays(config.WAVEFUNCTION_DATA_FILE, FORMAT, FORMAT_VERSION,
        {"x_au": x, "probability_au": probability}, configured_wavefunction_metadata())
    return x, probability, True


def main(force=False):
    """Create or reuse the HDF5 wavefunction data for :mod:`statics.config`."""
    x, probability, calculated = calculate_and_save_configured_wavefunction(force=force)
    action = "Saved" if calculated else "Reused"
    print(f"{action} {config.WAVEFUNCTION_DATA_FILE}")
    print(f"x points: {x.size}; probability shape: {probability.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the configured static wavefunction.")
    parser.add_argument("--force", action="store_true", help="recompute and overwrite the saved wavefunction data")
    try:
        main(force=parser.parse_args().force)
    except (FileNotFoundError, ValueError) as error:
        print(f"Cannot create wavefunction data: {error}", file=sys.stderr)
        raise SystemExit(1)
