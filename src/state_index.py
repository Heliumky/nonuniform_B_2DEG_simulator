"""Shared convention for indexing energy eigenstates.

Throughout the static and driven workflows, ``n`` is an energy-eigenstate
index sorted in ascending energy: ``n = 0`` is the ground state.
"""

from numbers import Integral


def validate_state_index(state_index, basis_size, *, name="state_index"):
    """Validate and return a zero-based energy-eigenstate index."""
    if isinstance(state_index, bool) or not isinstance(state_index, Integral):
        raise TypeError(f"{name} must be an integer energy-eigenstate index")
    if not 0 <= state_index < basis_size:
        raise ValueError(f"{name} must satisfy 0 <= {name} < {basis_size}; n=0 is the ground state")
    return int(state_index)


def state_index_label(state_index):
    """Return a concise human-readable label for the shared convention."""
    return "n=0 (ground state)" if state_index == 0 else f"n={state_index}"
