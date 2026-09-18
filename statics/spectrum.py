"""Static band-spectrum calculation and figure."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from src.hamiltonian import build_hamiltonian
from src.lanczos import lanczos_eigenpairs
from . import config
from .plot_metadata import title_with_parameters


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


def main():
    """Compute and save the configured band spectrum."""
    config.SPECTRUM_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ky = np.linspace(*config.SPECTRUM_KY_RANGE_NM, config.SPECTRUM_KY_POINTS)
    energies = compute_spectrum(ky, config.SPECTRUM_V_DC,
        config.SPECTRUM_BASIS_SIZE, config.SPECTRUM_STATE_COUNT, config.SPECTRUM_SOLVER,
        config.LANCZOS_DIMENSION)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for state_index, band in enumerate(energies.T):
        label = r"$n=0$ (ground state)" if state_index == 0 else rf"$n={state_index}$"
        ax.plot(ky, band, lw=1.2, label=label)
    ax.set_xlabel(r"$k_y$ (nm$^{-1}$)")
    ax.set_ylabel("Energy (meV)")
    ax.set_ylim(-5, 10)
    ax.legend(title="Energy eigenstate", ncols=2, fontsize=8)
    solver_details = config.SPECTRUM_SOLVER
    if config.SPECTRUM_SOLVER == "lanczos":
        solver_details += rf", Krylov dimension $={config.LANCZOS_DIMENSION}$"
    ax.set_title(title_with_parameters(
        "Static band spectrum",
        rf"$V_{{\rm DC}}/(\hbar\omega_c)={config.SPECTRUM_V_DC / (config.HBAR * config.CYCLOTRON_FREQUENCY):g}$, "
        rf"$k_y\in[{ky[0]:g}, {ky[-1]:g}]\,\mathrm{{nm}}^{{-1}}$, "
        rf"$N_{{k_y}}={config.SPECTRUM_KY_POINTS}$" + "\n" +
        rf"$N_{{\rm basis}}={config.SPECTRUM_BASIS_SIZE}$, "
        rf"eigenstates $n=0,\ldots,{config.SPECTRUM_STATE_COUNT - 1}$, solver: {solver_details}"),
        fontsize=10, pad=14)
    fig.tight_layout()
    fig.savefig(config.SPECTRUM_OUTPUT, dpi=150)
    print(f"Saved {config.SPECTRUM_OUTPUT}")
    plt.show()


if __name__ == "__main__": main()
