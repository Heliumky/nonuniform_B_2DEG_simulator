"""Static band-spectrum calculation and HDF5 persistence."""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

if __package__:
    from . import config
    from .data_io import load_arrays, save_arrays
else:  # Support ``cd statics && python spectrum.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from statics import config
    from statics.data_io import load_arrays, save_arrays

from src.hamiltonian import build_hamiltonian
from src.lanczos import lanczos_eigenpairs

FORMAT = "nonuniform_B_2DEG_static_spectrum"
FORMAT_VERSION = 1


def compute_spectrum(ky_nm, dc_potential, basis_size, state_count, solver, lanczos_dimension):
    """Return energies for ``n=0, ..., state_count-1`` in meV."""
    if not 1 <= state_count < basis_size:
        raise ValueError("state_count must satisfy 1 <= state_count < basis_size")
    energies = []
    for value in ky_nm:
        matrix = build_hamiltonian(value * config.AU_TO_NM, dc_potential, basis_size,
            hbar=config.HBAR, charge=config.ELEMENTARY_CHARGE,
            field=config.MAGNETIC_FIELD, mass=config.EFFECTIVE_MASS,
            length=config.MAGNETIC_LENGTH_SCALE, omega=config.CYCLOTRON_FREQUENCY)
        values = (eigh(matrix, eigvals_only=True)[:state_count] if solver == "dense"
                  else lanczos_eigenpairs(matrix, state_count, lanczos_dimension)[0])
        energies.append(values)
    return np.asarray(energies) * config.AU_TO_MEV


def configured_spectrum_metadata():
    """Return every setting needed to validate a cached spectrum file."""
    return {
        "ky_range_nm": list(config.SPECTRUM_KY_RANGE_NM),
        "ky_points": config.SPECTRUM_KY_POINTS,
        "dc_potential": config.SPECTRUM_V_DC,
        "basis_size": config.SPECTRUM_BASIS_SIZE,
        "state_count": config.SPECTRUM_STATE_COUNT,
        "solver": config.SPECTRUM_SOLVER,
        "lanczos_dimension": config.LANCZOS_DIMENSION,
        "hbar": config.HBAR,
        "elementary_charge": config.ELEMENTARY_CHARGE,
        "effective_mass": config.EFFECTIVE_MASS,
        "magnetic_field": config.MAGNETIC_FIELD,
        "magnetic_length_scale": config.MAGNETIC_LENGTH_SCALE,
        "cyclotron_frequency": config.CYCLOTRON_FREQUENCY,
    }


def _run_configured_spectrum():
    ky = np.linspace(*config.SPECTRUM_KY_RANGE_NM, config.SPECTRUM_KY_POINTS)
    energies = compute_spectrum(ky, config.SPECTRUM_V_DC, config.SPECTRUM_BASIS_SIZE,
        config.SPECTRUM_STATE_COUNT, config.SPECTRUM_SOLVER, config.LANCZOS_DIMENSION)
    return ky, energies


def load_configured_spectrum():
    """Load the saved spectrum for the exact current configuration.

    Plotting callers use this function and deliberately never calculate the
    spectrum implicitly.
    """
    path = config.SPECTRUM_DATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No spectrum data at {path}. Run `python spectrum.py` from statics/ "
            "or `python -m statics.spectrum` from the repository root first."
        )
    arrays, metadata = load_arrays(path, FORMAT, FORMAT_VERSION)
    if metadata != configured_spectrum_metadata():
        raise ValueError(
            f"Spectrum metadata in {path} does not match statics/config.py. "
            "Run `python spectrum.py --force` from statics/ or "
            "`python -m statics.spectrum --force` from the repository root."
        )
    return arrays["ky_nm"], arrays["energies_mev"]


def calculate_and_save_configured_spectrum(force=False):
    """Load a matching cached spectrum or calculate and persist it once."""
    if not force and config.SPECTRUM_DATA_FILE.exists():
        return (*load_configured_spectrum(), False)
    ky, energies = _run_configured_spectrum()
    save_arrays(config.SPECTRUM_DATA_FILE, FORMAT, FORMAT_VERSION,
        {"ky_nm": ky, "energies_mev": energies}, configured_spectrum_metadata())
    return ky, energies, True


def main(force=False):
    """Create or reuse the HDF5 spectrum data for :mod:`statics.config`."""
    ky, energies, calculated = calculate_and_save_configured_spectrum(force=force)
    action = "Saved" if calculated else "Reused"
    print(f"{action} {config.SPECTRUM_DATA_FILE}")
    print(f"ky points: {ky.size}; energies: {energies.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate the configured static spectrum.")
    parser.add_argument("--force", action="store_true", help="recompute and overwrite the saved spectrum data")
    try:
        main(force=parser.parse_args().force)
    except (FileNotFoundError, ValueError) as error:
        print(f"Cannot create spectrum data: {error}", file=sys.stderr)
        raise SystemExit(1)
