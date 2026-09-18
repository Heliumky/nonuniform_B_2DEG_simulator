"""Stage 3a: plot saved probability-density data without recalculation."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

if not __package__:  # Support direct execution from the dynamics directory.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hamiltonian import effective_potential

if __package__:
    from . import config
    from .density import load_configured_density, load_density_file
    from .propagation import ac_factor
else:  # Support direct execution from the dynamics directory.
    from dynamics import config
    from dynamics.density import load_configured_density, load_density_file
    from dynamics.propagation import ac_factor


def _default_output_path(density_path):
    """Put a user-selected density movie in the normal figures directory."""
    return config.OUTPUT_DIRECTORY / f"{Path(density_path).stem}.gif"


def _resolve_density_path(path):
    """Find a bare density filename in the standard data directory."""
    path = Path(path)
    if path.exists() or path.parent != Path("."):
        return path
    standard_path = config.DATA_DIRECTORY / path
    return standard_path if standard_path.exists() else path


def main(density_path=None, output_path=None, frame_stride=None):
    """Load one chosen density HDF5 file only, then save its GIF."""
    if density_path is None:
        x, times, density = load_configured_density()
        _, _, _, metadata = load_density_file(config.DENSITY_DATA_FILE)
        density_path = config.DENSITY_DATA_FILE
    else:
        density_path = _resolve_density_path(density_path)
        x, times, density, metadata = load_density_file(density_path)
    output_path = (config.DENSITY_GIF if density_path == config.DENSITY_DATA_FILE and output_path is None
                   else _default_output_path(density_path) if output_path is None else Path(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_stride = config.FRAME_STRIDE if frame_stride is None else frame_stride
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    trajectory_metadata = metadata["trajectory_metadata"]
    sample = slice(None, None, frame_stride)
    x_nm = x * config.AU_TO_NM
    density, sampled_times = density[sample], times[sample]
    potentials = np.array([
        effective_potential(
            x, trajectory_metadata["ky"], trajectory_metadata["dc_potential"],
            trajectory_metadata["ac_amplitude"] * ac_factor(
                time, trajectory_metadata["ac_frequency"], trajectory_metadata["ac_waveform"]),
            hbar=trajectory_metadata["hbar"], charge=trajectory_metadata["elementary_charge"],
            field=trajectory_metadata["magnetic_field"], mass=trajectory_metadata["effective_mass"],
            length=trajectory_metadata["magnetic_length_scale"],
        ) * config.AU_TO_MEV
        for time in sampled_times
    ])

    figure, (potential_axis, density_axis) = plt.subplots(
        2, 1, figsize=config.FIGURE_SIZE, sharex=True, layout="constrained"
    )
    figure.suptitle(
        rf"Driven probability density | {trajectory_metadata['ac_waveform']} drive | "
        rf"initial $n={trajectory_metadata['initial_state_index']}$"
        + (" (ground state)" if trajectory_metadata['initial_state_index'] == 0 else "")
    )
    potential_line, = potential_axis.plot(x_nm, potentials[0])
    density_line, = density_axis.plot(x_nm, density[0])
    potential_axis.set_ylabel(r"$V_{\rm eff}$ (meV)")
    density_axis.set_ylabel(r"$|\psi(x,t)|^2$")
    density_axis.set_xlabel(r"$x$ (nm)")
    potential_axis.set_ylim(potentials.min() - 0.5, potentials.max() + 0.5)
    density_axis.set_ylim(0, density.max() * 1.1)
    time_text = potential_axis.text(0.98, 0.88, "", transform=potential_axis.transAxes,
                                    ha="right", va="top", fontsize=10,
                                    bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75"})

    def update(frame):
        potential_line.set_ydata(potentials[frame])
        density_line.set_ydata(density[frame])
        time_text.set_text(
            f"t / T_ac = {sampled_times[frame] * trajectory_metadata['ac_frequency'] / (2 * np.pi):.3f}"
        )
        return potential_line, density_line, time_text

    movie = animation.FuncAnimation(figure, update, frames=sampled_times.size, blit=False)
    movie.save(output_path, writer="pillow", fps=config.FRAMES_PER_SECOND)
    plt.close(figure)
    print(f"Saved {output_path}")


def _validate_comparable(first, second):
    """Require identical runs apart from the AC waveform for comparisons."""
    x_a, times_a, _, metadata_a = first
    x_b, times_b, _, metadata_b = second
    if not np.array_equal(x_a, x_b) or not np.array_equal(times_a, times_b):
        raise ValueError("Density files have different x or time grids and cannot be compared directly")
    try:
        trajectory_a = dict(metadata_a["trajectory_metadata"])
        trajectory_b = dict(metadata_b["trajectory_metadata"])
        waveform_a = trajectory_a.pop("ac_waveform")
        waveform_b = trajectory_b.pop("ac_waveform")
    except KeyError as error:
        raise ValueError(f"Density metadata is missing {error.args[0]!r}") from error
    trajectory_a.pop("run_tag", None)
    trajectory_b.pop("run_tag", None)
    if trajectory_a != trajectory_b:
        raise ValueError("Density files differ in physical or numerical parameters beyond AC waveform")
    remaining_a = dict(metadata_a)
    remaining_b = dict(metadata_b)
    remaining_a.pop("trajectory_metadata")
    remaining_b.pop("trajectory_metadata")
    if remaining_a != remaining_b:
        raise ValueError("Density files use different reconstruction-grid metadata")
    return waveform_a, waveform_b, trajectory_a


def plot_comparison(first_path, second_path, output_path=None):
    """Write normalization and local-density differences for two HDF5 files."""
    first = load_density_file(_resolve_density_path(first_path))
    second = load_density_file(_resolve_density_path(second_path))
    waveform_a, waveform_b, trajectory_metadata = _validate_comparable(first, second)
    x, times, density_a, _ = first
    _, _, density_b, _ = second
    x_nm = x * config.AU_TO_NM
    periods = times * trajectory_metadata["ac_frequency"] / (2 * np.pi)
    difference = density_a - density_b
    norms_a = np.trapezoid(density_a, x, axis=1)
    norms_b = np.trapezoid(density_b, x, axis=1)
    l2_difference = np.sqrt(np.trapezoid(difference**2, x, axis=1))

    figure = plt.figure(figsize=(11.5, 8.0), layout="constrained")
    layout = figure.add_gridspec(2, 2, height_ratios=(1, 1.45))
    norm_axis = figure.add_subplot(layout[0, 0])
    l2_axis = figure.add_subplot(layout[0, 1])
    difference_axis = figure.add_subplot(layout[1, :])
    for values, label, color in ((norms_a, waveform_a, "#c43c39"),
                                 (norms_b, waveform_b, "#007c78")):
        norm_axis.plot(periods, values, color=color, lw=1.6, label=label)
    norm_axis.set(title=r"Spatial normalization", xlabel=r"$t/T_{\rm ac}$",
                  ylabel=r"$\int |\psi|^2 dx$")
    norm_axis.grid(color="0.88", lw=0.8)
    norm_axis.legend(title="waveform")
    l2_axis.plot(periods, l2_difference, color="#6c4a9a", lw=1.6)
    l2_axis.set(title=r"Density separation", xlabel=r"$t/T_{\rm ac}$",
                ylabel=r"$\sqrt{\int (\Delta\rho)^2 dx}$")
    l2_axis.grid(color="0.88", lw=0.8)
    limit = np.max(np.abs(difference))
    image = difference_axis.pcolormesh(
        x_nm, periods, difference, shading="auto", cmap="RdBu_r",
        vmin=-limit if limit else None, vmax=limit if limit else None,
    )
    difference_axis.set(title=rf"$\Delta\rho=\rho^{{{waveform_a}}}-\rho^{{{waveform_b}}}$",
                        xlabel=r"$x$ (nm)", ylabel=r"$t/T_{\rm ac}$")
    figure.colorbar(image, ax=difference_axis, label=r"$\Delta\rho$ (a.u.)")
    figure.suptitle(rf"Density comparison: {waveform_a} versus {waveform_b}", fontsize=12)
    if output_path is None:
        base = Path(first_path).stem.replace("density_", "", 1)
        output_path = config.OUTPUT_DIRECTORY / f"density_comparison_{waveform_a}_minus_{waveform_b}_{base}.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"Saved {output_path}")
    print(f"RMS Delta density = {np.sqrt(np.mean(difference**2)):.6e} a.u.")
    return output_path


def plot_overlay(first_path, second_path, output_path=None, frame_stride=None):
    """Animate two comparable density datasets on shared spatial axes."""
    first = load_density_file(_resolve_density_path(first_path))
    second = load_density_file(_resolve_density_path(second_path))
    waveform_a, waveform_b, trajectory_metadata = _validate_comparable(first, second)
    x, times, density_a, _ = first
    _, _, density_b, _ = second
    frame_stride = (config.COMPARISON_FRAME_STRIDE if frame_stride is None
                    else frame_stride)
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    sample = slice(None, None, frame_stride)
    x_nm, sampled_times = x * config.AU_TO_NM, times[sample]
    density_a, density_b = density_a[sample], density_b[sample]

    figure, axis = plt.subplots(figsize=(8.2, 4.9), layout="constrained")
    line_a, = axis.plot(x_nm, density_a[0], color="#c43c39", lw=2.1,
                        label=rf"{waveform_a}: solid")
    line_b, = axis.plot(x_nm, density_b[0], color="#007c78", lw=2.1, ls="--",
                        label=rf"{waveform_b}: dashed")
    axis.set(xlabel=r"$x$ (nm)", ylabel=r"$|\psi(x,t)|^2$",
             title=rf"Density overlay: {waveform_a} (solid) vs {waveform_b} (dashed)")
    axis.set_ylim(0.0, 1.1 * max(np.max(density_a), np.max(density_b)))
    axis.axvline(0.0, color="0.55", lw=0.8, ls=":", zorder=0)
    axis.grid(axis="y", color="0.88", lw=0.8)
    axis.legend()
    time_text = axis.text(0.98, 0.92, "", transform=axis.transAxes, ha="right", va="top",
                          bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75"})

    def update(frame):
        line_a.set_ydata(density_a[frame])
        line_b.set_ydata(density_b[frame])
        time_text.set_text(
            f"t / T_ac = {sampled_times[frame] * trajectory_metadata['ac_frequency'] / (2 * np.pi):.3f}"
        )
        return line_a, line_b, time_text

    movie = animation.FuncAnimation(figure, update, frames=sampled_times.size,
                                    interval=50, blit=False)
    if output_path is None:
        base = Path(first_path).stem.replace("density_", "", 1)
        output_path = config.OUTPUT_DIRECTORY / f"density_overlay_{waveform_a}_vs_{waveform_b}_{base}.gif"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output_path, writer="pillow", fps=config.FRAMES_PER_SECOND)
    plt.close(figure)
    print(f"Saved {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot one density GIF or compare two density HDF5 files.")
    comparison_mode = parser.add_mutually_exclusive_group()
    comparison_mode.add_argument("--compare", nargs=2, metavar=("FIRST_H5", "SECOND_H5"),
                                 help="write normalization and local-density differences as PNG")
    comparison_mode.add_argument("--overlay", nargs=2, metavar=("FIRST_H5", "SECOND_H5"),
                                 help="write a spatial-density overlay GIF")
    parser.add_argument("density", nargs="?", help="density HDF5 to animate (default: configured density data)")
    parser.add_argument("--output", "-o", help="output path (GIF for animation; PNG for --compare)")
    parser.add_argument("--frame-stride", type=int, help="use every Nth stored time snapshot")
    try:
        arguments = parser.parse_args()
        if arguments.compare:
            plot_comparison(*arguments.compare, output_path=arguments.output)
        elif arguments.overlay:
            plot_overlay(*arguments.overlay, output_path=arguments.output,
                         frame_stride=arguments.frame_stride)
        else:
            main(arguments.density, arguments.output, arguments.frame_stride)
    except (FileNotFoundError, ValueError) as error:
        print(f"Cannot plot density: {error}", file=sys.stderr)
        raise SystemExit(1)
