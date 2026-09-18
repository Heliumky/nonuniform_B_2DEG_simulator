"""Static eigenstate probability-density calculation and figure."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from src.basis import sho_basis
from src.hamiltonian import build_hamiltonian
from src.state_index import state_index_label, validate_state_index
from . import config
from .plot_metadata import title_with_parameters


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


def main():
    """Render the configured static eigenstate probability density."""
    config.WAVEFUNCTION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE,
                    np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE,
                    config.WAVEFUNCTION_X_POINTS)
    probability = eigenstate_probability(config.WAVEFUNCTION_KY_NM * config.AU_TO_NM,
        config.WAVEFUNCTION_V_DC,
        config.WAVEFUNCTION_STATE_INDEX, config.WAVEFUNCTION_BASIS_SIZE, x)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.plot(x * config.AU_TO_NM, probability)
    ax.set_xlabel("x (nm)")
    ax.set_ylabel(r"$|\psi_n(x)|^2$")
    ax.set_title(title_with_parameters(
        "Static eigenstate probability density",
        f"eigenstate {state_index_label(config.WAVEFUNCTION_STATE_INDEX)}, "
        rf"$k_y={config.WAVEFUNCTION_KY_NM:g}\,\mathrm{{nm}}^{{-1}}$, "
        rf"$V_{{\rm DC}}/(\hbar\omega_c)={config.WAVEFUNCTION_V_DC / (config.HBAR * config.CYCLOTRON_FREQUENCY):g}$, "
        rf"$N_{{\rm basis}}={config.WAVEFUNCTION_BASIS_SIZE}$, $N_x={x.size}$, "
        rf"$x\in[{x[0] * config.AU_TO_NM:.3g}, {x[-1] * config.AU_TO_NM:.3g}]\,\mathrm{{nm}}$"),
        fontsize=10, pad=14)
    fig.tight_layout()
    fig.savefig(config.WAVEFUNCTION_OUTPUT, dpi=150)
    print(f"Saved {config.WAVEFUNCTION_OUTPUT}")
    plt.show()


if __name__ == "__main__": main()
