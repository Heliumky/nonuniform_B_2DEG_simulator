"""Render the static eigenstate probability density from its saved HDF5 data.

This never recomputes the wavefunction; run ``python wave_function.py`` first.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

if __package__:
    from . import config
    from .plot_metadata import title_with_parameters
    from .wave_function import load_configured_wavefunction
else:  # Support ``cd statics && python plot_wave_function.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from statics import config
    from statics.plot_metadata import title_with_parameters
    from statics.wave_function import load_configured_wavefunction

from src.state_index import state_index_label


def main():
    """Plot the configured static eigenstate probability density from saved data."""
    x, probability = load_configured_wavefunction()
    config.WAVEFUNCTION_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"Cannot plot wavefunction: {error}", file=sys.stderr)
        raise SystemExit(1)
