"""Render the static band spectrum from its saved HDF5 data.

This never recomputes the spectrum; run ``python spectrum.py`` first.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

if __package__:
    from . import config
    from .plot_metadata import title_with_parameters
    from .spectrum import load_configured_spectrum
else:  # Support ``cd statics && python plot_spectrum.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from statics import config
    from statics.plot_metadata import title_with_parameters
    from statics.spectrum import load_configured_spectrum


def main():
    """Plot the configured static band spectrum from saved data."""
    ky, energies = load_configured_spectrum()
    config.SPECTRUM_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"Cannot plot spectrum: {error}", file=sys.stderr)
        raise SystemExit(1)
