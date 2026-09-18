"""Driven 2DEG workflow: model setup, initial state, propagation, diagnostics."""

import numpy as np

from src.basis import sho_basis
from src.hamiltonian import build_hamiltonian, split_hamiltonian
from src.lanczos import expm_action, lanczos_eigenpairs
from src.propagators import evolve_y6_3
from src.state_index import validate_state_index

from . import config
from .trajectory import load_trajectory, save_trajectory


def ac_factor(time, ac_frequency, waveform="sin"):
    """Return the dimensionless AC waveform ``f(ω_ac t)``."""
    if waveform == "sin":
        return np.sin(ac_frequency * time)
    if waveform == "cos":
        return np.cos(ac_frequency * time)
    raise ValueError(f"Unsupported AC waveform {waveform!r}; use 'sin' or 'cos'")


def initial_eigenstate(ky, dc_potential, basis_size, state_index, eigsh_ncv):
    """Return static eigenstate ``n=state_index`` used to prepare dynamics.

    The AC field is deliberately excluded here, so the same prepared state is
    used for either sine or cosine propagation.  In particular, cosine driving
    applies its nonzero ``t=0`` field as a quench to this static eigenstate.
    Energy eigenstates are sorted by increasing energy, so ``n=0`` is the
    ground state.  At exactly zero DC bias the static Hamiltonian has parity
    symmetry and exponentially close double-well partners.  Dense ``eigh`` is
    used there so these partners retain their deterministic even/odd labels;
    ARPACK can otherwise return an arbitrary rotation of a near-degenerate
    pair.  For nonzero DC bias, ARPACK obtains only the requested low states.
    """
    state_index = validate_state_index(state_index, basis_size)
    static_hamiltonian = build_hamiltonian(
        ky, dc_potential, basis_size, hbar=config.HBAR,
        charge=config.ELEMENTARY_CHARGE, field=config.MAGNETIC_FIELD,
        mass=config.EFFECTIVE_MASS, length=config.MAGNETIC_LENGTH_SCALE,
        omega=config.CYCLOTRON_FREQUENCY,
    )
    if dc_potential == 0.0 or state_index == basis_size - 1:
        return np.linalg.eigh(static_hamiltonian)[1][:, state_index].astype(complex)
    _, vectors = lanczos_eigenpairs(static_hamiltonian, state_count=state_index + 1,
                                    k=eigsh_ncv)
    return vectors[:, state_index].astype(complex)


def run_dynamics(*, ky, dc_potential, ac_amplitude, ac_frequency, basis_size,
                 initial_state_index, total_time, number_of_steps,
                 initial_eigsh_ncv, ac_waveform="sin"):
    """Run one driven channel with explicit atomic-unit parameters.

    Returns ``(states, times)`` with shapes ``(number_of_steps,basis_size)``
    and ``(number_of_steps,)``.
    """
    if number_of_steps < 1:
        raise ValueError("number_of_steps must be positive")
    initial = initial_eigenstate(ky, dc_potential, basis_size, initial_state_index,
                                 initial_eigsh_ncv)
    kinetic, static, ac_operator = split_hamiltonian(ky, dc_potential, ac_amplitude, basis_size,
        hbar=config.HBAR, charge=config.ELEMENTARY_CHARGE, field=config.MAGNETIC_FIELD,
        mass=config.EFFECTIVE_MASS, length=config.MAGNETIC_LENGTH_SCALE,
        omega=config.CYCLOTRON_FREQUENCY)
    times = np.linspace(0.0, total_time, number_of_steps + 1)
    potential = lambda time: static + ac_factor(time, ac_frequency, ac_waveform) * ac_operator
    action = lambda matrix, state, scale: expm_action(matrix, state, scale)
    return evolve_y6_3(initial, kinetic, potential, times, config.HBAR, action)


def reconstruct_probability(states, x, basis_size):
    """Reconstruct ``|ψ(x,t)|²`` from coefficient snapshots.

    ``states`` has shape ``(nt,basis_size)`` and output has shape ``(nt,len(x))``.
    """
    basis = sho_basis(x, basis_size, config.EFFECTIVE_MASS,
                      config.CYCLOTRON_FREQUENCY, config.HBAR)
    return np.abs(states @ basis) ** 2


def configured_run_metadata():
    """Return every physics/numerics setting needed to validate a cache file."""
    return {
        "run_tag": config.RUN_TAG,
        "units": "Hartree atomic units",
        "hbar": config.HBAR,
        "elementary_charge": config.ELEMENTARY_CHARGE,
        "effective_mass": config.EFFECTIVE_MASS,
        "magnetic_field": config.MAGNETIC_FIELD,
        "magnetic_length_scale": config.MAGNETIC_LENGTH_SCALE,
        "cyclotron_frequency": config.CYCLOTRON_FREQUENCY,
        "ky": config.KY,
        "dc_potential": config.V_DC,
        "ac_amplitude": config.V_AC,
        "ac_frequency": config.AC_FREQUENCY,
        "ac_waveform": config.AC_WAVEFORM,
        "basis_size": config.BASIS_SIZE,
        "initial_state_index": config.INITIAL_STATE_INDEX,
        "initial_eigsh_ncv": config.INITIAL_EIGSH_NCV,
        "initial_state_solver": "dense_eigh_if_vdc_zero_else_arpack",
        "total_time": config.TOTAL_TIME,
        "number_of_steps": config.NUMBER_OF_STEPS,
    }


def _run_configured_dynamics():
    """Run the user configuration without reading or writing a cache file."""
    return run_dynamics(ky=config.KY, dc_potential=config.V_DC,
        ac_amplitude=config.V_AC, ac_frequency=config.AC_FREQUENCY, basis_size=config.BASIS_SIZE,
        initial_state_index=config.INITIAL_STATE_INDEX, total_time=config.TOTAL_TIME,
        number_of_steps=config.NUMBER_OF_STEPS,
        initial_eigsh_ncv=config.INITIAL_EIGSH_NCV, ac_waveform=config.AC_WAVEFORM)


def load_configured_dynamics():
    """Load the saved trajectory for the exact current configuration.

    Plotting callers use this function and deliberately never start an
    expensive propagation implicitly.
    """
    path = config.TRAJECTORY_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No trajectory at {path}. Run `python dynamics.py` from dynamics/ "
            "or `python -m dynamics.dynamics` from the repository root first."
        )
    states, times, metadata = load_trajectory(path)
    expected = configured_run_metadata()
    if metadata != expected:
        raise ValueError(
            f"Trajectory metadata in {path} does not match dynamics/config.py. "
            "Run `python dynamics.py --force` from dynamics/ or "
            "`python -m dynamics.dynamics --force` from the repository root."
        )
    return states, times


def run_and_save_configured_dynamics(force=False):
    """Load a matching trajectory or calculate and persist it once.

    Set ``force=True`` to replace the trajectory for the current run tag.
    """
    if not force and config.TRAJECTORY_FILE.exists():
        return (*load_configured_dynamics(), False)
    states, times = _run_configured_dynamics()
    save_trajectory(config.TRAJECTORY_FILE, states, times, configured_run_metadata())
    return states, times, True


def main(force=False):
    """Create or reuse the HDF5 trajectory for :mod:`dynamics.config`."""
    states, times, calculated = run_and_save_configured_dynamics(force=force)
    norms = np.linalg.norm(states, axis=1)
    action = "Saved" if calculated else "Reused"
    print(f"{action} {config.TRAJECTORY_FILE}")
    print(f"states: {states.shape}; norm range: {norms.min():.12f} .. {norms.max():.12f}")


if __name__ == "__main__":
    main()
