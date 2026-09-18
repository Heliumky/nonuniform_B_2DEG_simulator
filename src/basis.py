"""Simple-harmonic-oscillator basis functions and operator matrices."""

import numpy as np


def sho_operator_matrices(basis_size, mass, omega, hbar):
    """Build SHO ``x, x², x⁴, p²`` matrices in atomic units.

    Returns ``(x0, p0, x, x2, x4, p2)``; all matrices have shape
    ``(basis_size, basis_size)``.
    """
    if basis_size < 1:
        raise ValueError("basis_size must be positive")
    x0 = np.sqrt(hbar / (2 * mass * omega))
    p0 = np.sqrt(mass * hbar * omega / 2)
    n = np.arange(basis_size)
    x = np.diag(x0 * np.sqrt(n[1:]), 1) + np.diag(x0 * np.sqrt(n[1:]), -1)
    x2 = np.diag(x0**2 * (2 * n + 1))
    if basis_size > 2:
        off2 = x0**2 * np.sqrt(n[2:] * n[1:-1])
        x2 += np.diag(off2, 2) + np.diag(off2, -2)
    x4 = np.diag(x0**4 * (6 * n**2 + 6 * n + 3))
    if basis_size > 2:
        off2 = x0**4 * (4 * n[:-2] + 6) * np.sqrt(n[2:] * n[1:-1])
        x4 += np.diag(off2, 2) + np.diag(off2, -2)
    if basis_size > 4:
        off4 = x0**4 * np.sqrt(n[4:] * n[3:-1] * n[2:-2] * n[1:-3])
        x4 += np.diag(off4, 4) + np.diag(off4, -4)
    p2 = np.diag(p0**2 * (2 * n + 1))
    if basis_size > 2:
        off2 = -p0**2 * np.sqrt(n[2:] * n[1:-1])
        p2 += np.diag(off2, 2) + np.diag(off2, -2)
    return x0, p0, x, x2, x4, p2


def sho_basis(x, basis_size, mass, omega, hbar):
    """Evaluate normalized SHO functions on atomic-unit positions ``x``.

    Returns ``np.ndarray`` with shape ``(basis_size, len(x))``.
    """
    x = np.asarray(x)
    length = np.sqrt(hbar / (mass * omega))
    q = x / length
    values = np.zeros((basis_size, x.size))
    values[0] = (np.pi * length**2) ** -0.25 * np.exp(-q**2 / 2)
    if basis_size > 1:
        values[1] = np.sqrt(2) * q * values[0]
    for n in range(1, basis_size - 1):
        values[n + 1] = np.sqrt(2 / (n + 1)) * q * values[n] - np.sqrt(n / (n + 1)) * values[n - 1]
    return values


def sho_basis_and_derivative(x, basis_size, mass, omega, hbar):
    """Return SHO values and analytic derivatives, both shape ``(N, nx)``."""
    extended = sho_basis(x, basis_size + 1, mass, omega, hbar)
    length = np.sqrt(hbar / (mass * omega))
    derivative = np.empty((basis_size, len(x)))
    for n in range(basis_size):
        lower = np.sqrt(n) * extended[n - 1] if n else 0.0
        derivative[n] = (lower - np.sqrt(n + 1) * extended[n + 1]) / (np.sqrt(2) * length)
    return extended[:basis_size], derivative
