"""Pure Hamiltonian and effective-potential construction functions."""

import numpy as np

from .basis import sho_operator_matrices


def effective_potential(x, ky, dc_potential, ac_potential=0.0, *, hbar, charge, field, mass, length):
    """Evaluate the static-plus-AC effective potential in atomic units.

    The linear term is ``e (V_DC + V_AC(t)) x / L``.  Set
    ``ac_potential`` to zero for the static Hamiltonian.
    """
    x = np.asarray(x)
    return ((hbar * ky + charge * field * x**2 / (2 * length))**2 / (2 * mass)
            + charge * (dc_potential + ac_potential) * x / length)


def build_hamiltonian(ky, dc_potential, basis_size, *, hbar, charge, field, mass, length, omega):
    """Build the Hermitian static channel Hamiltonian in an SHO basis.

    ``ky`` and ``dc_potential`` are in atomic units. The returned array
    has shape ``(basis_size, basis_size)``.
    """
    _, _, x, x2, x4, p2 = sho_operator_matrices(basis_size, mass, omega, hbar)
    return (p2 / (2 * mass) + charge * dc_potential * x / length
            + hbar * ky * charge * field * x2 / (2 * mass * length)
            + charge**2 * field**2 * x4 / (8 * mass * length**2)
            + hbar**2 * ky**2 * np.eye(basis_size) / (2 * mass))


def split_hamiltonian(ky, dc_potential, ac_amplitude, basis_size, *, hbar, charge, field, mass, length, omega):
    """Return ``T, V_static, V_AC`` for the driven channel Hamiltonian.

    ``T + V_static`` is ``H_static(V_DC)`` and ``V_AC`` is the operator
    multiplied by ``f(ω_ac t)`` during propagation.
    """
    _, _, x, x2, x4, p2 = sho_operator_matrices(basis_size, mass, omega, hbar)
    kinetic = p2 / (2 * mass)
    static = (hbar**2 * ky**2 * np.eye(basis_size) / (2 * mass)
              + charge * dc_potential * x / length
              + hbar * ky * charge * field * x2 / (2 * mass * length)
              + charge**2 * field**2 * x4 / (8 * mass * length**2))
    return kinetic, static, charge * ac_amplitude * x / length
