"""Shared, human-readable parameter labels for static figures."""

from . import config


def material_parameters_label():
    """Return the material and field parameters used by every static plot."""
    field_tesla = config.MAGNETIC_FIELD * 2.35e5
    length_nm = config.MAGNETIC_LENGTH_SCALE * config.AU_TO_NM
    cyclotron_energy_mev = config.CYCLOTRON_FREQUENCY * config.AU_TO_MEV
    return (rf"$B_0={field_tesla:.3g}\,\mathrm{{T}}$, "
            rf"$m^*={config.EFFECTIVE_MASS:.3g}m_e$, "
            rf"$L={length_nm:.3g}\,\mathrm{{nm}}$, "
            rf"$\hbar\omega_c={cyclotron_energy_mev:.3g}\,\mathrm{{meV}}$")


def title_with_parameters(name, run_parameters):
    """Build a compact two-line plot title with shared and run-specific settings."""
    return f"{name}\n{material_parameters_label()}\n{run_parameters}"
