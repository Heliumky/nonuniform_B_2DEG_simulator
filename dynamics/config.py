r"""User-editable parameters for the default driven 2DEG calculation.

The initial state is prepared from the DC-biased static Hamiltonian

.. math::

   H_{\rm static} = \frac{p_x^2}{2m^*}
       + \frac{\left(\hbar k_y + eB_0x^2/(2L)\right)^2}{2m^*}
       + \frac{eV_{\rm DC}x}{L},
   \qquad H_{\rm static}\phi_n=E_n\phi_n,
   \qquad \psi(0)=\phi_{\mathtt{INITIAL\_STATE\_INDEX}}.

The propagation Hamiltonian adds the AC field to that fixed initial state:

.. math::

   H(t) = H_{\rm static}(V_{\rm DC})
       + \frac{eV_{\rm AC}x}{L}f(\omega_{\rm ac}t).

Consequently, ``AC_WAVEFORM = \"sin\"`` has ``f(0)=0`` and begins
continuously from ``H_static``.  ``\"cos\"`` has ``f(0)=1`` and applies the
maximum AC term suddenly at ``t=0`` (a quench), while retaining exactly the
same prepared state.  The driven Hamiltonian at ``t=0`` is therefore

.. math::

   H_{\cos}(0) = H_{\rm static}(V_{\rm DC}) + \frac{eV_{\rm AC}x}{L},

whereas

.. math::

   H_{\sin}(0) = H_{\rm static}(V_{\rm DC}).

At any later time ``t_prime``,

.. math::

   H(t') = \frac{p_x^2}{2m^*}
       + \frac{\left(\hbar k_y + eB_0x^2/(2L)\right)^2}{2m^*}
       + \frac{eV_{\rm DC}x}{L}
       + \frac{eV_{\rm AC}x}{L}f(\omega_{\rm ac}t').
"""

import numpy as np
from pathlib import Path

# SHARED PHYSICAL PARAMETERS (atomic units unless noted otherwise)
# Adjust these values here to configure the driven calculation's material and
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

# USER PARAMETERS
BASIS_SIZE = 501
KY = -0.15 * AU_TO_NM                 # au; input value is -0.15 nm^-1
# ARPACK's ``ncv`` subspace size used only to prepare a DC-biased (V_DC != 0)
# initial static eigenstate.  The exactly symmetric V_DC = 0 double well uses
# dense eigh to keep its near-degenerate even/odd states well defined.  This
# setting does not affect SciPy's adaptive ``expm_multiply`` propagation.
INITIAL_EIGSH_NCV = 30
V_DC = 0 * HBAR * CYCLOTRON_FREQUENCY
V_AC = 0.5 * HBAR * CYCLOTRON_FREQUENCY
AC_FREQUENCY = CYCLOTRON_FREQUENCY
# Eigenstate index of H_static(V_DC); n=0 is the ground state.
INITIAL_STATE_INDEX = 0
NUMBER_OF_STEPS = 1000
TOTAL_TIME = 10 * 2 * np.pi / AC_FREQUENCY
SPATIAL_GRID_POINTS = 401
# Propagation and plotted potentials use this same waveform. ``sin`` and
# ``cos`` are supported.
AC_WAVEFORM = "sin"

# ANIMATION USER PARAMETERS
FRAME_STRIDE = 3
# Comparisons use the same temporal sampling as ordinary one-file GIFs.
# Increase this only when deliberately trading temporal resolution for a
# smaller/faster-to-render comparison animation.
COMPARISON_FRAME_STRIDE = FRAME_STRIDE
FRAMES_PER_SECOND = 20
FIGURE_SIZE = (7, 7)
# The potential-comparison animation always renders both sin and cos. It is
# independent of ``AC_WAVEFORM`` above.
POTENTIAL_COMPARISON_FIGURE_SIZE = (11, 5.2)
POTENTIAL_COMPARISON_PERIODS = 1
POTENTIAL_COMPARISON_FRAMES = 121
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _number_token(value):
    """Return a compact, filesystem-safe token for a numeric parameter."""
    return f"{value:.6g}".replace("-", "m").replace("+", "p").replace(".", "p")


RUN_TAG = (
    f"y6_3_{AC_WAVEFORM}_ky{_number_token(KY / AU_TO_NM)}nm1_"
    f"Vdc{_number_token(V_DC / (HBAR * CYCLOTRON_FREQUENCY))}hwc_"
    f"Vac{_number_token(V_AC / (HBAR * CYCLOTRON_FREQUENCY))}hwc_"
    f"w{_number_token(AC_FREQUENCY / CYCLOTRON_FREQUENCY)}wc_"
    f"n{INITIAL_STATE_INDEX}_Nbasis{BASIS_SIZE}_Nt{NUMBER_OF_STEPS}_"
    f"T{_number_token(TOTAL_TIME * AC_FREQUENCY / (2 * np.pi))}periods"
)
OUTPUT_DIRECTORY = PROJECT_ROOT / "figures" / "dynamics"
DATA_DIRECTORY = PROJECT_ROOT / "data" / "dynamics"
TRAJECTORY_FILE = DATA_DIRECTORY / f"trajectory_{RUN_TAG}.h5"
# Stage 2: spatially resolved line current density (Jx(x,t), Jy(x,t)).
CURRENT_DENSITY_DATA_FILE = DATA_DIRECTORY / f"current_density_{RUN_TAG}_Nx{SPATIAL_GRID_POINTS}.h5"
DENSITY_DATA_FILE = DATA_DIRECTORY / f"density_{RUN_TAG}_Nx{SPATIAL_GRID_POINTS}.h5"
DENSITY_GIF = OUTPUT_DIRECTORY / f"density_{RUN_TAG}.gif"
CURRENT_DENSITY_GIF = OUTPUT_DIRECTORY / f"current_density_{RUN_TAG}.gif"
# Stage 3: x-integrated Ix(t) (diagnostic only) and Iy(t) per unit channel
# length Ly (not an absolute current -- see dynamics/current.py), derived
# from the current-density file above.
CURRENT_DATA_FILE = DATA_DIRECTORY / f"current_{RUN_TAG}_Nx{SPATIAL_GRID_POINTS}.h5"
CURRENT_FIGURE = OUTPUT_DIRECTORY / f"current_{RUN_TAG}_Nx{SPATIAL_GRID_POINTS}.png"
POTENTIAL_COMPARISON_GIF = OUTPUT_DIRECTORY / (
    "potential_sin_cos_"
    f"ky{_number_token(KY / AU_TO_NM)}nm1_"
    f"Vdc{_number_token(V_DC / (HBAR * CYCLOTRON_FREQUENCY))}hwc_"
    f"Vac{_number_token(V_AC / (HBAR * CYCLOTRON_FREQUENCY))}hwc_"
    f"w{_number_token(AC_FREQUENCY / CYCLOTRON_FREQUENCY)}wc_"
    f"Nx{SPATIAL_GRID_POINTS}_"
    f"T{_number_token(POTENTIAL_COMPARISON_PERIODS)}periods.gif"
)
