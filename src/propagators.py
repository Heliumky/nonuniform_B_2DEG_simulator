"""Core fully commutator-free sixth-order ``Y^6_3`` propagator.

All functions are independent of a particular physical model. Matrices and
times are supplied explicitly in atomic units; the state-vector shape is ``(N,)``.
"""

import numpy as np

GAUSS_LEGENDRE_6_NODES = np.array([0.5 - np.sqrt(15) / 10, 0.5, 0.5 + np.sqrt(15) / 10])
Y6_3_A1 = np.array([0.01994096265093610745, 0.0, -0.01994096265093610745])
Y6_3_A2 = np.array([0.4882524910228221957, -0.0046136830175630621, 0.0834019108602182940])
Y6_3_A3 = np.array([-0.29387662410526271191, 0.4536718104795705687, -0.29387662410526271191])


def _store_steps(state, step_function, times):
    states = []
    for start, end in zip(times[:-1], times[1:]):
        state = step_function(state, start, end - start)
        states.append(state.copy())
    return np.asarray(states), np.asarray(times[1:])


def evolve_y6_3(initial_state, kinetic, potential_at_time, times, hbar, expm_action):
    """Apply the five-factor, fully commutator-free Eq. (18) of method.pdf."""
    t2, t3 = Y6_3_A2.sum(), Y6_3_A3.sum()
    def step(state, start, dt):
        potentials = np.array([potential_at_time(start + node * dt) for node in GAUSS_LEGENDRE_6_NODES])
        bars = tuple(np.tensordot(coefficients, potentials, axes=(0, 0))
                     for coefficients in (Y6_3_A1, Y6_3_A2, Y6_3_A3,
                                          Y6_3_A2[::-1], Y6_3_A1[::-1]))
        state = expm_action(bars[0], state, -1j * dt / hbar)
        state = expm_action(t2 * kinetic + bars[1], state, -1j * dt / hbar)
        state = expm_action(t3 * kinetic + bars[2], state, -1j * dt / hbar)
        state = expm_action(t2 * kinetic + bars[3], state, -1j * dt / hbar)
        return expm_action(bars[4], state, -1j * dt / hbar)
    return _store_steps(np.asarray(initial_state, dtype=complex), step, times)
