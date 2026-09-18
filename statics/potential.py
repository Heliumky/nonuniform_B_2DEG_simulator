"""Plot the static effective potential; its formula lives in ``src``."""

import matplotlib.pyplot as plt
import numpy as np

from src.hamiltonian import effective_potential
from . import config
from .plot_metadata import title_with_parameters


def main():
    """Render the user-selected effective-potential curves."""
    config.POTENTIAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE,
                    np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE,
                    config.POTENTIAL_X_POINTS)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for ky in config.POTENTIAL_KY_AU:
        values = effective_potential(
            x, ky, config.POTENTIAL_V_DC,
            hbar=config.HBAR, charge=config.ELEMENTARY_CHARGE,
            field=config.MAGNETIC_FIELD, mass=config.EFFECTIVE_MASS,
            length=config.MAGNETIC_LENGTH_SCALE)
        ax.plot(x * config.AU_TO_NM, values * config.AU_TO_MEV,
                label=rf"$k_y={ky / config.AU_TO_NM:.3g}\,\mathrm{{nm}}^{{-1}}$")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel(r"$V_\mathrm{eff}(x)$ (meV)")
    ax.set_ylim(0, 30)
    ax.legend(title="Channel momentum")
    ax.set_title(title_with_parameters(
        "Static effective potential",
        rf"$V_{{\rm DC}}/(\hbar\omega_c)={config.POTENTIAL_V_DC / (config.HBAR * config.CYCLOTRON_FREQUENCY):g}$, "
        rf"$x\in[{x[0] * config.AU_TO_NM:.3g}, {x[-1] * config.AU_TO_NM:.3g}]\,\mathrm{{nm}}$, "
        rf"$N_x={x.size}$, "
        r"$V_{\rm eff}(x)=(\hbar k_y+eB_0x^2/(2L))^2/(2m^*)+eV_{\rm DC}x/L$"),
        fontsize=10, pad=14)
    fig.tight_layout()
    fig.savefig(config.POTENTIAL_OUTPUT, dpi=150)
    print(f"Saved {config.POTENTIAL_OUTPUT}")
    plt.show()


if __name__ == "__main__":
    main()
