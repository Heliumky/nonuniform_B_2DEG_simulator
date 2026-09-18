r"""User-editable parameters for static calculations and figures.

Every static workflow uses

.. math::

   H_{k_y}^{\rm static}(V_{\rm DC}) = \frac{p_x^2}{2m^*}
       + \frac{(\hbar k_y + eB_0x^2/(2L))^2}{2m^*}
       + \frac{eV_{\rm DC}}{L}x.

Each figure has its own ``*_V_DC`` setting below, so its DC bias is directly
editable without any AC term.
"""

from pathlib import Path
import numpy as np

# SHARED PHYSICAL PARAMETERS (atomic units unless noted otherwise)
# Adjust these values here to configure the static calculation's material and
# magnetic-field model.
AU_TO_MEV = 2.7e4
AU_TO_NM = 5.29e-2
HBAR = 1.0
ELEMENTARY_CHARGE = 1.0
ELECTRON_MASS = 1.0
EFFECTIVE_MASS = 0.067 * ELECTRON_MASS
MAGNETIC_FIELD = 1.65 / 2.35e5
CYCLOTRON_FREQUENCY = ELEMENTARY_CHARGE * MAGNETIC_FIELD / EFFECTIVE_MASS
MAGNETIC_LENGTH_SCALE = 16 * np.pi**2 / AU_TO_NM

def _number_token(value):
    """Format a number compactly and safely for a filename."""
    return f"{value:.6g}".replace("-", "minus").replace("+", "plus").replace(".", "p")


def _ky_range_token(lower, upper):
    return f"ky{_number_token(lower)}to{_number_token(upper)}"


# USER PARAMETERS: spectrum
SPECTRUM_KY_RANGE_NM = (-0.5, 0.5)
SPECTRUM_KY_POINTS = 101
# DC potential in H_static: e V_DC x / L.
SPECTRUM_V_DC = 0.5 * HBAR * CYCLOTRON_FREQUENCY
SPECTRUM_BASIS_SIZE = 601
# Plot the lowest SPECTRUM_STATE_COUNT eigenstates: n=0, ..., count-1.
# In both statics and dynamics, n=0 is the ground state.
SPECTRUM_STATE_COUNT = 12
SPECTRUM_SOLVER = "lanczos"
LANCZOS_DIMENSION = 30

# USER PARAMETERS: wavefunction and potential figures
WAVEFUNCTION_BASIS_SIZE = 201
WAVEFUNCTION_KY_NM = -0.15
WAVEFUNCTION_V_DC = 0.0 * HBAR * CYCLOTRON_FREQUENCY
WAVEFUNCTION_STATE_INDEX = 0  # n=0 is the ground state.
WAVEFUNCTION_X_POINTS = 501
# Atomic-unit values retained exactly from the original potential script.
POTENTIAL_KY_AU = (0.15 / 18.9, -0.05 / 18.9, -0.15 / 18.9)
POTENTIAL_V_DC = 0.0 * HBAR * CYCLOTRON_FREQUENCY
POTENTIAL_X_POINTS = 500

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = PROJECT_ROOT / "figures" / "statics"

# Output names are derived from every run-specific numerical setting. Thus a
# changed configuration produces a distinct image rather than replacing one.
SPECTRUM_OUTPUT = OUTPUT_DIRECTORY / (
    "spectrum_"
    f"{_ky_range_token(*SPECTRUM_KY_RANGE_NM)}_Nky{SPECTRUM_KY_POINTS}_"
    f"Vdc{_number_token(SPECTRUM_V_DC / (HBAR * CYCLOTRON_FREQUENCY))}_Nbasis{SPECTRUM_BASIS_SIZE}_"
    f"states_n0to{SPECTRUM_STATE_COUNT - 1}_{SPECTRUM_SOLVER}_M{LANCZOS_DIMENSION}.png"
)
WAVEFUNCTION_OUTPUT = OUTPUT_DIRECTORY / (
    "probability_"
    f"n{WAVEFUNCTION_STATE_INDEX}_ky{_number_token(WAVEFUNCTION_KY_NM)}_"
    f"Vdc{_number_token(WAVEFUNCTION_V_DC / (HBAR * CYCLOTRON_FREQUENCY))}_Nbasis{WAVEFUNCTION_BASIS_SIZE}_"
    f"Nx{WAVEFUNCTION_X_POINTS}.png"
)
POTENTIAL_OUTPUT = OUTPUT_DIRECTORY / (
    "potential_"
    f"Vdc{_number_token(POTENTIAL_V_DC / (HBAR * CYCLOTRON_FREQUENCY))}_"
    f"ky{'_'.join(_number_token(ky * 18.9) for ky in POTENTIAL_KY_AU)}_"
    f"Nx{POTENTIAL_X_POINTS}.png"
)
