"""Render the static effective potential from its saved HDF5 data.

This never recomputes the potential; run ``python potential.py`` first.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

if __package__:
    from . import config
    from .plot_metadata import title_with_parameters
    from .potential import load_configured_potential
else:  # Support ``cd statics && python plot_potential.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from statics import config
    from statics.plot_metadata import title_with_parameters
    from statics.potential import load_configured_potential


def main():
    """Plot the configured static effective-potential curves from saved data."""
    x, values = load_configured_potential()
    config.POTENTIAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    for ky, curve in zip(config.POTENTIAL_KY_AU, values):
        ax.plot(x * config.AU_TO_NM, curve,
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
        r"$V_{\rm eff}(x)=(\hbar k_y+eB_0x^2/(2L))^2/(2m^\ast)+eV_{\rm DC}x/L$"),
        fontsize=10, pad=14)
    fig.tight_layout()
    fig.savefig(config.POTENTIAL_OUTPUT, dpi=150)
    print(f"Saved {config.POTENTIAL_OUTPUT}")
    plt.show()


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"Cannot plot potential: {error}", file=sys.stderr)
        raise SystemExit(1)
