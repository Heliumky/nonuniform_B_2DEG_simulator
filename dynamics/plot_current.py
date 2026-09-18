"""Public stage 4: plot a saved HDF5 current dataset without recalculation.

Plots ``Ix(t)`` (diagnostic only, no established physical meaning) and
``Iy(t)`` per unit channel length ``L_y`` (not an absolute current) — see
:mod:`dynamics.current` for exactly what these mean.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

if __package__:
    from . import config
    from .current import load_configured_current, load_current_file
else:  # Support ``cd dynamics && python plot_current.py``.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dynamics import config
    from dynamics.current import load_configured_current, load_current_file


def _trajectory_metadata(metadata):
    return metadata["current_density_metadata"]["trajectory_metadata"]


def _default_output_path(current_path):
    """Put a user-selected current plot in the normal figures directory."""
    return config.OUTPUT_DIRECTORY / f"{Path(current_path).stem}.png"


def _resolve_current_path(path):
    """Find a bare current filename in the standard data directory."""
    path = Path(path)
    if path.exists() or path.parent != Path("."):
        return path
    standard_path = config.DATA_DIRECTORY / path
    return standard_path if standard_path.exists() else path


def main(current_path=None, output_path=None):
    """Load one chosen current HDF5 file only, then save its PNG."""
    if current_path is None:
        times, ix, iy_per_ly = load_configured_current()
        _, _, _, metadata = load_current_file(config.CURRENT_DATA_FILE)
        current_path = config.CURRENT_DATA_FILE
    else:
        current_path = _resolve_current_path(current_path)
        times, ix, iy_per_ly, metadata = load_current_file(current_path)
    output_path = (config.CURRENT_FIGURE if current_path == config.CURRENT_DATA_FILE and output_path is None
                   else _default_output_path(current_path) if output_path is None else Path(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_metadata = _trajectory_metadata(metadata)
    frequency = trajectory_metadata["ac_frequency"]
    time_periods = times * frequency / (2 * np.pi)

    figure, (ix_axis, iy_axis) = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True, layout="constrained")
    for axis, values, color, label, title in (
        (ix_axis, ix, "#c43c39", r"$I_x=\int J_x\,dx$ (a.u.)", r"$I_x=\int J_x\,dx$ (diagnostic only, not a physical current)"),
        (iy_axis, iy_per_ly, "#007c78", r"$I_y/L_y=\int J_y\,dx$ (a.u.)", r"Transverse current $I_y$ per unit channel length $L_y$"),
    ):
        axis.plot(time_periods, values, color=color, lw=1.6)
        axis.axhline(0.0, color="0.35", lw=0.8, zorder=0)
        axis.grid(color="0.88", lw=0.8)
        axis.set_ylabel(label)
        axis.set_title(title, loc="left", color=color, fontsize=11, fontweight="bold")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        axis.spines[["top", "right"]].set_visible(False)
    iy_axis.set_xlabel(r"$t/T_{\rm ac}$")
    figure.suptitle(
        rf"Driven channel current  |  $k_y={trajectory_metadata['ky'] / config.AU_TO_NM:.2f}\,\mathrm{{nm}}^{{-1}}$"
        rf",  $V_{{\rm DC}}={trajectory_metadata['dc_potential'] / (trajectory_metadata['hbar'] * trajectory_metadata['cyclotron_frequency']):.2f}\hbar\omega_c$"
        rf",  $V_{{\rm AC}}={trajectory_metadata['ac_amplitude'] / (trajectory_metadata['hbar'] * trajectory_metadata['cyclotron_frequency']):.2f}\hbar\omega_c$"
        rf",  initial $n={trajectory_metadata['initial_state_index']}$"
        + (" (ground state)" if trajectory_metadata['initial_state_index'] == 0 else ""),
        fontsize=12,
    )
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print(f"Saved {output_path}")


def _validate_comparable(first_metadata, second_metadata):
    """Validate matching physics parameters apart from waveform; return the two waveforms."""
    trajectory_a = dict(_trajectory_metadata(first_metadata))
    trajectory_b = dict(_trajectory_metadata(second_metadata))
    waveform_a = trajectory_a.pop("ac_waveform")
    waveform_b = trajectory_b.pop("ac_waveform")
    trajectory_a.pop("run_tag", None)
    trajectory_b.pop("run_tag", None)
    if trajectory_a != trajectory_b:
        raise ValueError("Current files differ in physical or numerical parameters beyond AC waveform")
    return waveform_a, waveform_b, trajectory_a


def plot_comparison(first_path, second_path, output_path=None):
    """Overlay Ix(t) and Iy(t)/Ly from two saved HDF5 files."""
    first_path = _resolve_current_path(first_path)
    second_path = _resolve_current_path(second_path)
    times_a, ix_a, iy_a, metadata_a = load_current_file(first_path)
    times_b, ix_b, iy_b, metadata_b = load_current_file(second_path)
    if not np.array_equal(times_a, times_b):
        raise ValueError("Current files have different time grids and cannot be compared directly")
    waveform_a, waveform_b, trajectory = _validate_comparable(metadata_a, metadata_b)
    frequency = trajectory["ac_frequency"]
    time_periods = times_a * frequency / (2 * np.pi)

    figure, (ix_axis, iy_axis) = plt.subplots(2, 1, figsize=(8.2, 6.4), sharex=True, layout="constrained")
    for axis, values_a, values_b, ylabel, title in (
        (ix_axis, ix_a, ix_b, r"$I_x=\int J_x\,dx$ (a.u.)", r"$I_x=\int J_x\,dx$ (diagnostic only, not a physical current)"),
        (iy_axis, iy_a, iy_b, r"$I_y/L_y=\int J_y\,dx$ (a.u.)", r"Transverse current $I_y$ per unit channel length $L_y$"),
    ):
        axis.plot(time_periods, values_a, color="#c43c39", lw=1.6, label=waveform_a)
        axis.plot(time_periods, values_b, color="#007c78", lw=1.6, ls="--", label=waveform_b)
        axis.axhline(0.0, color="0.35", lw=0.8, zorder=0)
        axis.grid(color="0.88", lw=0.8)
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.legend(title="waveform")
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        axis.spines[["top", "right"]].set_visible(False)
    iy_axis.set_xlabel(r"$t/T_{\rm ac}$")
    figure.suptitle(
        rf"Current comparison: {waveform_a} versus {waveform_b}  |  "
        rf"$k_y={trajectory['ky'] / config.AU_TO_NM:.2f}\,\mathrm{{nm}}^{{-1}}$, "
        rf"$V_{{\rm DC}}={trajectory['dc_potential'] / (trajectory['hbar'] * trajectory['cyclotron_frequency']):.2f}\hbar\omega_c$, "
        rf"$V_{{\rm AC}}={trajectory['ac_amplitude'] / (trajectory['hbar'] * trajectory['cyclotron_frequency']):.2f}\hbar\omega_c$",
        fontsize=12,
    )
    base = Path(first_path).stem.replace(f"current_y6_3_{waveform_a}_", "")
    output = (config.OUTPUT_DIRECTORY / f"current_comparison_{waveform_a}_minus_{waveform_b}_{base}.png"
              if output_path is None else Path(output_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Saved {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the true current, or compare two saved current HDF5 files.")
    parser.add_argument("--compare", nargs=2, metavar=("FIRST_H5", "SECOND_H5"),
                        help="overlay Ix(t) and Iy(t)/Ly from two current files as PNG")
    parser.add_argument("current", nargs="?", help="current HDF5 to plot (default: configured current data)")
    parser.add_argument("--output", "-o", help="output PNG path")
    arguments = parser.parse_args()
    try:
        if arguments.compare:
            plot_comparison(*arguments.compare, output_path=arguments.output)
        else:
            main(arguments.current, arguments.output)
    except (FileNotFoundError, ValueError) as error:
        print(f"Cannot plot current: {error}", file=sys.stderr)
        raise SystemExit(1)
