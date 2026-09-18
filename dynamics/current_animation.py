"""Current-density animation for the configured driven calculation."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from . import config
from .current import line_current_density, reconstruct_wavefunction
from .propagation import run_dynamics


def main():
    """Run configured dynamics and save a labelled line-current animation."""
    config.CURRENT_GIF.parent.mkdir(parents=True, exist_ok=True)
    states, times = run_dynamics(ky=config.KY, dc_potential=config.V_DC,
        ac_amplitude=config.V_AC, ac_frequency=config.AC_FREQUENCY,
        basis_size=config.BASIS_SIZE, initial_state_index=config.INITIAL_STATE_INDEX,
        total_time=config.TOTAL_TIME, number_of_steps=config.NUMBER_OF_STEPS,
        initial_eigsh_ncv=config.INITIAL_EIGSH_NCV, ac_waveform=config.AC_WAVEFORM)
    x = np.linspace(-np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE,
                    np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE, config.SPATIAL_GRID_POINTS)
    psi, derivative = reconstruct_wavefunction(states, x, mass=config.EFFECTIVE_MASS,
        omega=config.CYCLOTRON_FREQUENCY, hbar=config.HBAR)
    jx, jy = line_current_density(psi, derivative, x, config.KY, hbar=config.HBAR,
        charge=config.ELEMENTARY_CHARGE, field=config.MAGNETIC_FIELD,
        mass=config.EFFECTIVE_MASS, length=config.MAGNETIC_LENGTH_SCALE)
    sampled_times = times[::config.FRAME_STRIDE]
    jx, jy = jx[::config.FRAME_STRIDE], jy[::config.FRAME_STRIDE]
    x_nm = x * config.AU_TO_NM

    figure, (jx_axis, jy_axis) = plt.subplots(
        2, 1, figsize=(8.2, 7.0), sharex=True, layout="constrained"
    )
    styles = ((jx_axis, jx, "#c43c39", r"$L_y j_x$ (a.u.)", r"Longitudinal current $J_x$"),
              (jy_axis, jy, "#007c78", r"$L_y j_y$ (a.u.)", r"Transverse current $J_y$"))
    lines = []

    for axis, values, color, label, title in styles:
        line, = axis.plot(x_nm, values[0], color=color, lw=2.2)
        lines.append(line)
        # A fixed full-trajectory scale makes amplitudes comparable frame to
        # frame and prevents axis/tick jitter in the GIF.
        magnitude = np.max(np.abs(values))
        limit = 1.12 * magnitude if magnitude else 1.0
        axis.set_ylim(-limit, limit)
        axis.axhline(0.0, color="0.35", lw=0.8, zorder=0)
        axis.axvline(0.0, color="0.55", lw=0.8, ls="--", zorder=0)
        axis.grid(axis="y", color="0.88", lw=0.8)
        axis.set_ylabel(label)
        axis.set_title(title, loc="left", color=color, fontsize=11, fontweight="bold")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        axis.spines[["top", "right"]].set_visible(False)
    jy_axis.set_xlabel(r"$x$ (nm)")
    time_text = jx_axis.text(0.98, 0.88, "", transform=jx_axis.transAxes,
                               ha="right", va="top", fontsize=10,
                               bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75"})
    figure.suptitle(
        rf"Driven channel currents  |  $k_y={config.KY / config.AU_TO_NM:.2f}\,\mathrm{{nm}}^{{-1}}$"
        rf",  $V_{{\rm DC}}={config.V_DC / (config.HBAR * config.CYCLOTRON_FREQUENCY):.2f}\hbar\omega_c$"
        rf",  $V_{{\rm AC}}={config.V_AC / (config.HBAR * config.CYCLOTRON_FREQUENCY):.2f}\hbar\omega_c$"
        rf",  initial $n={config.INITIAL_STATE_INDEX}$"
        + (" (ground state)" if config.INITIAL_STATE_INDEX == 0 else "")
        + "  (fixed current scale)",
        fontsize=12,
    )

    def update(frame):
        lines[0].set_ydata(jx[frame])
        lines[1].set_ydata(jy[frame])
        time_text.set_text(f"t / T = {sampled_times[frame] / config.TOTAL_TIME:.3f}")
        return *lines, time_text

    movie = animation.FuncAnimation(figure, update, frames=len(jx), interval=50, blit=False)
    movie.save(config.CURRENT_GIF, writer="pillow", fps=config.FRAMES_PER_SECOND)
    print(f"Saved {config.CURRENT_GIF}")


if __name__ == "__main__": main()
