"""Probability-density animation for the configured driven calculation."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from src.hamiltonian import effective_potential
from . import config
from .propagation import ac_factor, reconstruct_probability, run_dynamics


def main():
    """Run configured dynamics and save its probability animation."""
    config.PROBABILITY_GIF.parent.mkdir(parents=True, exist_ok=True)
    states, times = run_dynamics(ky=config.KY, dc_potential=config.V_DC,
        ac_amplitude=config.V_AC, ac_frequency=config.AC_FREQUENCY, basis_size=config.BASIS_SIZE,
        initial_state_index=config.INITIAL_STATE_INDEX, total_time=config.TOTAL_TIME,
        number_of_steps=config.NUMBER_OF_STEPS,
        initial_eigsh_ncv=config.INITIAL_EIGSH_NCV, ac_waveform=config.AC_WAVEFORM)
    x = np.linspace(-np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE,
                    np.sqrt(2) * config.MAGNETIC_LENGTH_SCALE, config.SPATIAL_GRID_POINTS)
    probability = reconstruct_probability(states, x, config.BASIS_SIZE)[::config.FRAME_STRIDE]
    sample_times = times[::config.FRAME_STRIDE]
    potentials = np.array([effective_potential(x, config.KY, config.V_DC,
        config.V_AC * ac_factor(t, config.AC_FREQUENCY, config.AC_WAVEFORM),
        hbar=config.HBAR, charge=config.ELEMENTARY_CHARGE, field=config.MAGNETIC_FIELD,
        mass=config.EFFECTIVE_MASS, length=config.MAGNETIC_LENGTH_SCALE) * config.AU_TO_MEV
        for t in sample_times])
    fig, (ax_v, ax_p) = plt.subplots(2, 1, figsize=config.FIGURE_SIZE, sharex=True)
    fig.suptitle(rf"Driven probability density | initial $n={config.INITIAL_STATE_INDEX}$"
                 + (" (ground state)" if config.INITIAL_STATE_INDEX == 0 else ""))
    vline, = ax_v.plot(x * config.AU_TO_NM, potentials[0]); pline, = ax_p.plot(x * config.AU_TO_NM, probability[0])
    ax_v.set_ylim(potentials.min()-.5, potentials.max()+.5); ax_p.set_ylim(0, probability.max()*1.1)
    def update(i): vline.set_ydata(potentials[i]); pline.set_ydata(probability[i]); return vline, pline
    movie = animation.FuncAnimation(fig, update, frames=len(sample_times), blit=True)
    movie.save(config.PROBABILITY_GIF, writer="pillow", fps=config.FRAMES_PER_SECOND)
    print(f"Saved {config.PROBABILITY_GIF}")


if __name__ == "__main__": main()
