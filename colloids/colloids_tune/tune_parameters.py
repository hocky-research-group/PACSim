from dataclasses import dataclass, field
from typing import Optional
from openmm import unit
from colloids.abstracts import Parameters


@dataclass(order=True, frozen=True)
class TuneParameters(Parameters):
    """
    Data class for the parameters used in colloids-tune which tunes the surface potential of a colloid with a given
    radius so that the potential depth of the combined steric and electrostatic potentials with another colloid is equal
    to the given potential depth.

    The parameters of the colloids and the steric and electostatic potentials are given in the RunParameters class.

    This dataclass can be written to and read from a yaml file. The yaml file contains the parameters as key-value
    pairs. Any OpenMM quantities are converted to Quantity objects that can be represented in a readable way in the
    yaml file.

    :param tuned_type:
        The type of the colloid whose surface potential will be tuned.
        This type must be present in the radii, masses, and surface_potentials dictionaries (with a dummy value) in the
        RunParameters class.
    :type tuned_type: str
    :param tuned_potential_depth:
        The desired potential depth of the combined steric and electrostatic potential with the other colloid.
        The unit of the potential_depth must be compatible with kilojoules per mole and the value must be smaller
        than zero.
    :type tuned_potential_depth: unit.Quantity
    :param other_type:
        The type of the other colloid that is used to tune the potential depth.
        This type must be present in the radii, masses, and surface_potentials dictionaries in the RunParameters class.
    :type other_type: str
    :param plot_filename:
        If not None, filename for the plot of the potential energy of the colloid with the tuned potential.
        Defaults to None.
    :type plot_filename: Optional[str]

    :raises TypeError:
        If any of the quantities has an incompatible unit.
    :raises ValueError:
        If any of the parameters has an invalid value.
    """
    tuned_type: str = "S"
    tuned_potential_depth: unit.Quantity = field(default_factory=lambda: -7.4331295806289965 * unit.kilojoule_per_mole)
    other_type: str = "P"
    plot_filename: Optional[str] = None
