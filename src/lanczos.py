"""Hermitian eigenpair and matrix-exponential action helpers.

This module uses SciPy's sparse linear-algebra routines instead of maintaining
a custom Lanczos recurrence here:

- `eigsh` is ARPACK's implicitly restarted Lanczos solver for Hermitian
  eigenproblems.
- `expm_multiply` applies the matrix exponential to a vector without building
  the full exponential matrix.  SciPy implements this through adaptive
  scaling and truncated Taylor series, rather than a user-configurable
  Lanczos/Krylov dimension.
"""

import numpy as np
from scipy.sparse.linalg import eigsh, expm_multiply


def _normalized_start_vector(n, psi0=None, complex_start=False):
    if psi0 is not None:
        psi0 = np.asarray(psi0)
        norm = np.linalg.norm(psi0)
        if norm == 0:
            raise ValueError("psi0 should not be the zero vector")
        return psi0 / norm

    rng = np.random.default_rng(1234)
    if complex_start:
        psi0 = rng.normal(size=n) + 1j * rng.normal(size=n)
    else:
        psi0 = rng.normal(size=n)
    return psi0 / np.linalg.norm(psi0)


def lanczos_eigenpairs(H, state_count=1, k=60, tol=1e-10, psi0=None):
    """Return the lowest ``state_count`` eigenpairs using restarted Lanczos.

    Args:
        H: Hermitian matrix or sparse/linear operator.
        state_count: Number of lowest eigenpairs to compute, i.e. indices
            ``n=0, ..., state_count-1``.
        k: Krylov subspace dimension passed to ARPACK as `ncv`.
        tol: ARPACK convergence tolerance.
        psi0: Optional starting vector.
    """
    n = H.shape[0]
    if H.shape[0] != H.shape[1]:
        raise ValueError("H should be a square matrix")
    if state_count < 1:
        raise ValueError("state_count should be at least 1")
    if state_count >= n:
        raise ValueError("state_count should be smaller than matrix dimension")

    if state_count >= n - 1:
        evals, evecs = np.linalg.eigh(H)
        return evals[:state_count], evecs[:, :state_count]

    start = _normalized_start_vector(
        n,
        psi0=psi0,
        complex_start=np.iscomplexobj(H),
    )
    ncv = min(n, max(k, 2 * state_count + 1))
    evals, evecs = eigsh(
        H,
        k=state_count,
        which="SA",
        ncv=ncv,
        tol=tol,
        v0=start,
    )
    order = np.argsort(evals)
    return evals[order], evecs[:, order]


def lanczos_ground_state(H, psi0=None, k=60, dtype=complex, tol=1e-10):
    """Return the lowest eigenvalue/eigenvector using restarted Lanczos."""
    evals, evecs = lanczos_eigenpairs(
        H,
        state_count=1,
        k=k,
        tol=tol,
        psi0=psi0,
    )
    return evals[0], evecs[:, 0].astype(dtype, copy=False)


def expm_action(H, psi0, scale, dtype=complex):
    """Return ``expm(scale * H) @ psi0`` using SciPy's adaptive action.

    SciPy chooses scaling and truncated-Taylor parameters internally, with a
    double-precision target.  It exposes no Krylov-dimension or tolerance
    argument for this routine.
    """
    psi0 = np.asarray(psi0, dtype=dtype)
    if psi0.ndim != 1:
        raise ValueError("psi0 should be a vector")
    if H.shape[1] != psi0.shape[0]:
        raise ValueError("Shape of H doesn't match len of psi0.")
    return expm_multiply(scale * H, psi0)
