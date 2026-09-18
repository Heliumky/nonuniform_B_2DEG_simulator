"""Side-by-side ``t=0`` cosine and sine effective-potential figure."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.hamiltonian import effective_potential

from . import config
from .propagation import ac_factor


def effective_potentials_at_zero(x):
    """Return the cosine and sine effective potentials at ``t=0`` in meV."""
    kwargs = {
        "hbar": config.HBAR,
        "charge": config.ELEMENTARY_CHARGE,
        "field": config.MAGNETIC_FIELD,
        "mass": config.EFFECTIVE_MASS,
        "length": config.MAGNETIC_LENGTH_SCALE,
    }
    potentials = []
    for waveform in ("cos", "sin"):
        ac_potential = config.V_AC * ac_factor(
            0.0, config.AC_FREQUENCY, waveform
        )
        potentials.append(
            effective_potential(x, config.KY, config.V_DC, ac_potential, **kwargs)
            * config.AU_TO_MEV
        )
    return tuple(potentials)


def main():
    """Save the configured sine-versus-cosine effective potential at ``t=0``."""
    config.POTENTIAL_T0_COMPARISON_PNG.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE,
                    np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE,
                    config.SPATIAL_GRID_POINTS)
    cos_potential, sin_potential = effective_potentials_at_zero(x)
    x_nm = x * config.AU_TO_NM
    lower = min(np.min(cos_potential), np.min(sin_potential))
    upper = max(np.max(cos_potential), np.max(sin_potential))
    margin = 0.05 * (upper - lower)

    figure, axes = plt.subplots(1, 2, figsize=config.POTENTIAL_COMPARISON_FIGURE_SIZE,
                                sharex=True, sharey=True, layout="constrained")
    for axis, waveform, potential, color, phase in zip(
            axes, ("cos", "sin"), (cos_potential, sin_potential),
            ("#c43c39", "#007c78"), ("1", "0")):
        axis.plot(x_nm, potential, color=color, lw=2.2)
        axis.axvline(0.0, color="0.55", lw=0.8, ls="--", zorder=0)
        axis.grid(axis="y", color="0.88", lw=0.8)
        axis.set_title(rf"$f(t)=\{waveform}(\omega t)$,  $f(0)={phase}$",
                       color=color, fontweight="bold")
        axis.set_xlabel(r"$x$ (nm)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"$V_{\rm eff}(x,0)$ (meV)")
    axes[0].set_ylim(lower - margin, upper + margin)
    figure.suptitle(
        rf"Driven potential at $t=0$:  $k_y={config.KY / config.AU_TO_NM:.2f}\,\mathrm{{nm}}^{{-1}}$, "
        rf"$V_{{\rm DC}}={config.V_DC / (config.HBAR * config.CYCLOTRON_FREQUENCY):.2f}\hbar\omega_c$, "
        rf"$V_{{\rm AC}}={config.V_AC / (config.HBAR * config.CYCLOTRON_FREQUENCY):.2f}\hbar\omega_c$"
        + "\n"
        + rf"Common prepared state: static-Hamiltonian eigenstate $n={config.INITIAL_STATE_INDEX}$",
        fontsize=13,
    )
    figure.savefig(config.POTENTIAL_T0_COMPARISON_PNG, dpi=150)
    plt.close(figure)
    print(f"Saved {config.POTENTIAL_T0_COMPARISON_PNG}")


if __name__ == "__main__":
    main()
