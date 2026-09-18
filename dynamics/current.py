"""Dynamic wavefunction reconstruction and channel-current observables."""

import numpy as np

from src.basis import sho_basis_and_derivative


def reconstruct_wavefunction(states, x, *, mass, omega, hbar):
    """Reconstruct ``ψ`` and ``∂xψ`` from ``states`` of shape ``(nt,N)``.

    Returned complex arrays have shape ``(nt,len(x))``; all values use au.
    """
    basis, derivative = sho_basis_and_derivative(x, states.shape[-1], mass, omega, hbar)
    return states @ basis, states @ derivative


def line_current_density(psi, derivative, x, ky, *, hbar, charge, field, mass, length):
    """Return line currents ``(Jx,Jy)`` in atomic units with ``psi`` shape."""
    density = np.abs(psi) ** 2
    jx = -charge * hbar / mass * np.imag(np.conj(psi) * derivative)
    py = hbar * ky + charge * field * np.asarray(x)**2 / (2 * length)
    return jx, -charge * py * density / mass


def continuity_residual(psi, jx, times, x, charge):
    """Return ``∂t(-e|ψ|²)+∂xJx`` for ``(nt,nx)`` sampled arrays."""
    return (np.gradient(-charge * np.abs(psi)**2, times, axis=0, edge_order=2)
            + np.gradient(jx, x, axis=1, edge_order=2))
