"""Public stage 3: plot a saved HDF5 current-density dataset without recalculation."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

if __package__:
    from . import config
    from .current_density import load_configured_current_density, load_current_density_file
else:  # Support ``cd dynamics && python plot_current_density.py``.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dynamics import config
    from dynamics.current_density import load_configured_current_density, load_current_density_file


def _default_output_path(current_density_path):
    """Put a user-selected current-density movie in the normal figures directory."""
    return config.OUTPUT_DIRECTORY / f"{Path(current_density_path).stem}.gif"


def _resolve_current_density_path(path):
    """Find a bare current-density filename in the standard data directory."""
    path = Path(path)
    if path.exists() or path.parent != Path("."):
        return path
    standard_path = config.DATA_DIRECTORY / path
    return standard_path if standard_path.exists() else path


def main(current_density_path=None, output_path=None, frame_stride=None):
    """Load one chosen current-density HDF5 file only, then save its GIF."""
    if current_density_path is None:
        x, times, jx, jy = load_configured_current_density()
        _, _, _, _, metadata = load_current_density_file(config.CURRENT_DENSITY_DATA_FILE)
        current_density_path = config.CURRENT_DENSITY_DATA_FILE
    else:
        current_density_path = _resolve_current_density_path(current_density_path)
        x, times, jx, jy, metadata = load_current_density_file(current_density_path)
    output_path = (config.CURRENT_DENSITY_GIF
                   if current_density_path == config.CURRENT_DENSITY_DATA_FILE and output_path is None
                   else _default_output_path(current_density_path) if output_path is None else Path(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_stride = config.FRAME_STRIDE if frame_stride is None else frame_stride
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    trajectory_metadata = metadata["trajectory_metadata"]
    sampled_times = times[::frame_stride]
    jx, jy = jx[::frame_stride], jy[::frame_stride]
    x_nm = x * config.AU_TO_NM

    figure, (jx_axis, jy_axis) = plt.subplots(
        2, 1, figsize=(8.2, 7.0), sharex=True, layout="constrained"
    )
    styles = ((jx_axis, jx, "#c43c39", r"$J_x$ (a.u.)", r"Longitudinal current density $J_x$"),
              (jy_axis, jy, "#007c78", r"$L_y j_y$ (a.u.)", r"Transverse current density $J_y$"))
    lines = []
    for axis, values, color, label, title in styles:
        line, = axis.plot(x_nm, values[0], color=color, lw=2.2)
        lines.append(line)
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
        rf"Driven channel current densities  |  $k_y={trajectory_metadata['ky'] / config.AU_TO_NM:.2f}\,\mathrm{{nm}}^{{-1}}$"
        rf",  $V_{{\rm DC}}={trajectory_metadata['dc_potential'] / (trajectory_metadata['hbar'] * trajectory_metadata['cyclotron_frequency']):.2f}\hbar\omega_c$"
        rf",  $V_{{\rm AC}}={trajectory_metadata['ac_amplitude'] / (trajectory_metadata['hbar'] * trajectory_metadata['cyclotron_frequency']):.2f}\hbar\omega_c$"
        rf",  initial $n={trajectory_metadata['initial_state_index']}$"
        + (" (ground state)" if trajectory_metadata['initial_state_index'] == 0 else "")
        + "  (fixed current scale)",
        fontsize=12,
    )

    def update(frame):
        lines[0].set_ydata(jx[frame])
        lines[1].set_ydata(jy[frame])
        time_text.set_text(
            f"t / T_ac = {sampled_times[frame] * trajectory_metadata['ac_frequency'] / (2 * np.pi):.3f}"
        )
        return *lines, time_text

    movie = animation.FuncAnimation(figure, update, frames=len(jx), interval=50, blit=False)
    movie.save(output_path, writer="pillow", fps=config.FRAMES_PER_SECOND)
    plt.close(figure)
    print(f"Saved {output_path}")


def _validate_comparable(first, second):
    """Validate matching grids and physics parameters apart from waveform."""
    x_a, times_a, _, _, metadata_a = first
    x_b, times_b, _, _, metadata_b = second
    if not np.array_equal(x_a, x_b) or not np.array_equal(times_a, times_b):
        raise ValueError("Current-density files have different x or time grids and cannot be compared directly")

    trajectory_a = dict(metadata_a["trajectory_metadata"])
    trajectory_b = dict(metadata_b["trajectory_metadata"])
    waveform_a = trajectory_a.pop("ac_waveform")
    waveform_b = trajectory_b.pop("ac_waveform")
    trajectory_a.pop("run_tag", None)
    trajectory_b.pop("run_tag", None)
    if trajectory_a != trajectory_b:
        raise ValueError("Current-density files differ in physical or numerical parameters beyond AC waveform")
    remaining_a = dict(metadata_a)
    remaining_b = dict(metadata_b)
    remaining_a.pop("trajectory_metadata")
    remaining_b.pop("trajectory_metadata")
    if remaining_a != remaining_b:
        raise ValueError("Current-density files use different reconstruction-grid metadata")
    return waveform_a, waveform_b, trajectory_a


def plot_comparison(first_path, second_path, output_path=None):
    """Plot local current-density differences for two saved HDF5 files.

    For the true (x-integrated) current comparison, see
    ``dynamics.plot_current --compare``.
    """
    first_path = _resolve_current_density_path(first_path)
    second_path = _resolve_current_density_path(second_path)
    first = load_current_density_file(first_path)
    second = load_current_density_file(second_path)
    waveform_a, waveform_b, metadata = _validate_comparable(first, second)
    x, times, jx_a, jy_a, _ = first
    _, _, jx_b, jy_b, _ = second
    x_nm = x * config.AU_TO_NM
    frequency = metadata["ac_frequency"]
    time_periods = times * frequency / (2 * np.pi)
    delta_jx, delta_jy = jx_a - jx_b, jy_a - jy_b

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for axis, values, title, label in (
        (axes[0], delta_jx, rf"$\Delta J_x=J_x^{{{waveform_a}}}-J_x^{{{waveform_b}}}$", r"$\Delta J_x$ (a.u.)"),
        (axes[1], delta_jy, rf"$\Delta J_y=J_y^{{{waveform_a}}}-J_y^{{{waveform_b}}}$", r"$\Delta J_y$ (a.u.)"),
    ):
        limit = np.max(np.abs(values))
        norm = colors.TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit) if limit else None
        image = axis.pcolormesh(x_nm, time_periods, values, shading="auto", cmap="RdBu_r", norm=norm)
        axis.set_title(title)
        axis.set_xlabel(r"$x$ (nm)")
        axis.set_ylabel(r"$t/T_{\rm ac}$")
        figure.colorbar(image, ax=axis, label=label)

    figure.suptitle(
        rf"Current-density comparison: {waveform_a} versus {waveform_b}  |  "
        rf"$k_y={metadata['ky'] / config.AU_TO_NM:.2f}\,\mathrm{{nm}}^{{-1}}$, "
        rf"$V_{{\rm DC}}={metadata['dc_potential'] / (metadata['hbar'] * metadata['cyclotron_frequency']):.2f}\hbar\omega_c$, "
        rf"$V_{{\rm AC}}={metadata['ac_amplitude'] / (metadata['hbar'] * metadata['cyclotron_frequency']):.2f}\hbar\omega_c$",
        fontsize=12,
    )
    base = Path(first_path).stem.replace(f"current_density_y6_3_{waveform_a}_", "")
    output = (config.OUTPUT_DIRECTORY / f"current_density_comparison_{waveform_a}_minus_{waveform_b}_{base}.png"
              if output_path is None else Path(output_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    print(f"Saved {output}")
    print(f"RMS Delta Jx = {np.sqrt(np.mean(delta_jx**2)):.6e} a.u.")
    print(f"RMS Delta Jy = {np.sqrt(np.mean(delta_jy**2)):.6e} a.u.")
    return output


def plot_overlay(first_path, second_path, output_path=None, frame_stride=None):
    """Animate spatial current densities from two comparable HDF5 files on shared axes."""
    first_path = _resolve_current_density_path(first_path)
    second_path = _resolve_current_density_path(second_path)
    first = load_current_density_file(first_path)
    second = load_current_density_file(second_path)
    waveform_a, waveform_b, metadata = _validate_comparable(first, second)
    x, times, jx_a, jy_a, _ = first
    _, _, jx_b, jy_b, _ = second
    frame_stride = (config.COMPARISON_FRAME_STRIDE if frame_stride is None
                    else frame_stride)
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive")
    sample = slice(None, None, frame_stride)
    x_nm, sampled_times = x * config.AU_TO_NM, times[sample]
    jx_a, jy_a, jx_b, jy_b = jx_a[sample], jy_a[sample], jx_b[sample], jy_b[sample]

    figure, (jx_axis, jy_axis) = plt.subplots(2, 1, figsize=(8.4, 7.2),
                                                sharex=True, layout="constrained")
    data = ((jx_axis, jx_a, jx_b, r"$J_x$ (a.u.)", r"Longitudinal current density $J_x$"),
            (jy_axis, jy_a, jy_b, r"$L_yj_y$ (a.u.)", r"Transverse current density $J_y$"))
    lines = []
    for axis, values_a, values_b, ylabel, title in data:
        line_a, = axis.plot(x_nm, values_a[0], color="#c43c39", lw=2.0,
                             label=rf"{waveform_a}: solid")
        line_b, = axis.plot(x_nm, values_b[0], color="#007c78", lw=2.0, ls="--",
                             label=rf"{waveform_b}: dashed")
        lines.extend((line_a, line_b))
        magnitude = max(np.max(np.abs(values_a)), np.max(np.abs(values_b)))
        limit = 1.12 * magnitude if magnitude else 1.0
        axis.set_ylim(-limit, limit)
        axis.axhline(0.0, color="0.35", lw=0.8, zorder=0)
        axis.axvline(0.0, color="0.55", lw=0.8, ls=":", zorder=0)
        axis.grid(axis="y", color="0.88", lw=0.8)
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left", fontsize=11, fontweight="bold")
        axis.legend(loc="upper left", fontsize=9)
        axis.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        axis.spines[["top", "right"]].set_visible(False)
    jy_axis.set_xlabel(r"$x$ (nm)")
    time_text = jx_axis.text(0.98, 0.88, "", transform=jx_axis.transAxes,
                               ha="right", va="top", fontsize=10,
                               bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "0.75"})
    figure.suptitle(
        rf"Spatial current-density overlay: {waveform_a} (solid) vs {waveform_b} (dashed)"
        + "\n"
        + rf"$k_y={metadata['ky'] / config.AU_TO_NM:.2f}\,\mathrm{{nm}}^{{-1}}$, "
        + rf"initial $n={metadata['initial_state_index']}$",
        fontsize=12,
    )

    def update(frame):
        lines[0].set_ydata(jx_a[frame])
        lines[1].set_ydata(jx_b[frame])
        lines[2].set_ydata(jy_a[frame])
        lines[3].set_ydata(jy_b[frame])
        time_text.set_text(
            f"t / T_ac = {sampled_times[frame] * metadata['ac_frequency'] / (2 * np.pi):.3f}"
        )
        return *lines, time_text

    movie = animation.FuncAnimation(figure, update, frames=sampled_times.size, interval=50, blit=False)
    base = Path(first_path).stem.replace(f"current_density_y6_3_{waveform_a}_", "")
    output = (config.OUTPUT_DIRECTORY / f"current_density_overlay_{waveform_a}_vs_{waveform_b}_{base}.gif"
              if output_path is None else Path(output_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    movie.save(output, writer="pillow", fps=config.FRAMES_PER_SECOND)
    plt.close(figure)
    print(f"Saved {output}")
    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Plot one current-density GIF or compare two current-density HDF5 files.")
    comparison_mode = parser.add_mutually_exclusive_group()
    comparison_mode.add_argument("--compare", nargs=2, metavar=("FIRST_H5", "SECOND_H5"),
                                 help="write local current-density differences as PNG")
    comparison_mode.add_argument("--overlay", nargs=2, metavar=("FIRST_H5", "SECOND_H5"),
                                 help="write a spatial current-density overlay GIF")
    parser.add_argument("current_density", nargs="?", help="current-density HDF5 to animate (default: configured current-density data)")
    parser.add_argument("--output", "-o", help="output path (GIF for animation; PNG for --compare)")
    parser.add_argument("--frame-stride", type=int, help="use every Nth stored time snapshot")
    arguments = parser.parse_args()
    try:
        if arguments.compare:
            plot_comparison(*arguments.compare, output_path=arguments.output)
        elif arguments.overlay:
            plot_overlay(*arguments.overlay, output_path=arguments.output,
                         frame_stride=arguments.frame_stride)
        else:
            main(arguments.current_density, arguments.output, arguments.frame_stride)
    except (FileNotFoundError, ValueError) as error:
        print(f"Cannot plot current density: {error}", file=sys.stderr)
        raise SystemExit(1)
