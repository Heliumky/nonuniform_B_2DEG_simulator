"""One-period, side-by-side cosine and sine effective-potential animation."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path
import sys

if not __package__:  # Support direct execution from the dynamics directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hamiltonian import effective_potential

if __package__:
    from . import config
    from .propagation import ac_factor
else:  # Support ``cd dynamics && python potential_comparison.py``.
    from dynamics import config
    from dynamics.propagation import ac_factor


def effective_potentials(x, time):
    """Return cosine and sine effective potentials at ``time`` in meV."""
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
            time, config.AC_FREQUENCY, waveform
        )
        potentials.append(
            effective_potential(x, config.KY, config.V_DC, ac_potential, **kwargs)
            * config.AU_TO_MEV
        )
    return tuple(potentials)


def main():
    """Save the configured one-period sine-versus-cosine potential GIF."""
    config.POTENTIAL_COMPARISON_GIF.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(-np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE,
                    np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE,
                    config.SPATIAL_GRID_POINTS)
    x_nm = x * config.AU_TO_NM
    period = 2 * np.pi / config.AC_FREQUENCY
    times = np.linspace(0.0, config.POTENTIAL_COMPARISON_PERIODS * period,
                        config.POTENTIAL_COMPARISON_FRAMES)
    cos_potential, sin_potential = effective_potentials(x, times[0])
    extrema = [effective_potential(
        x, config.KY, config.V_DC, ac_potential,
        hbar=config.HBAR, charge=config.ELEMENTARY_CHARGE,
        field=config.MAGNETIC_FIELD, mass=config.EFFECTIVE_MASS,
        length=config.MAGNETIC_LENGTH_SCALE,
    ) * config.AU_TO_MEV for ac_potential in (-config.V_AC, config.V_AC)]
    lower = min(np.min(values) for values in extrema)
    upper = max(np.max(values) for values in extrema)
    margin = 0.05 * (upper - lower)

    figure, axes = plt.subplots(1, 2, figsize=config.POTENTIAL_COMPARISON_FIGURE_SIZE,
                                sharex=True, sharey=True)
    figure.subplots_adjust(left=0.08, right=0.985, bottom=0.13, top=0.76, wspace=0.03)
    lines = []
    for axis, waveform, potential, color, initial_factor in zip(
            axes, ("cos", "sin"), (cos_potential, sin_potential),
            ("#c43c39", "#007c78"), ("1", "0")):
        line, = axis.plot(x_nm, potential, color=color, lw=2.2)
        lines.append(line)
        axis.axvline(0.0, color="0.55", lw=0.8, ls="--", zorder=0)
        axis.grid(axis="y", color="0.88", lw=0.8)
        axis.set_title(rf"$f(t)=\{waveform}(\omega t)$;  $f(0)={initial_factor}$",
                       color=color, fontweight="bold")
        axis.set_xlabel(r"$x$ (nm)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"$V_{\rm eff}(x,t)$ (meV)")
    axes[0].set_ylim(lower - margin, upper + margin)
    figure.suptitle(
        rf"Effective potential:  $k_y={config.KY / config.AU_TO_NM:.2f}\,\mathrm{{nm}}^{{-1}}$, "
        rf"$V_{{\rm DC}}={config.V_DC / (config.HBAR * config.CYCLOTRON_FREQUENCY):.2f}\hbar\omega_c$, "
        rf"$V_{{\rm AC}}={config.V_AC / (config.HBAR * config.CYCLOTRON_FREQUENCY):.2f}\hbar\omega_c$",
        fontsize=12, y=0.975,
    )
    time_label = figure.text(0.5, 0.865, "", ha="center", va="center", fontsize=11,
                             bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.7"})

    def update(frame):
        cos_values, sin_values = effective_potentials(x, times[frame])
        lines[0].set_ydata(cos_values)
        lines[1].set_ydata(sin_values)
        time_label.set_text(rf"$t/T={times[frame] / period:.3f}$")
        return *lines, time_label

    movie = animation.FuncAnimation(figure, update, frames=times.size, blit=False)
    movie.save(config.POTENTIAL_COMPARISON_GIF, writer="pillow", fps=config.FRAMES_PER_SECOND)
    plt.close(figure)
    print(f"Saved {config.POTENTIAL_COMPARISON_GIF}")


if __name__ == "__main__":
    main()
