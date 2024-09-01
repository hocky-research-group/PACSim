from dataclasses import dataclass, field
from typing import Optional
from openmm import unit
from colloids.abstracts import Parameters


@dataclass(order=True, frozen=True)
class TuneParameters(Parameters):
    """
    Data class for the parameters used in colloids-tune which tunes a specified per-particle or global parameter for a 
    binary ionic colloid solution so that the potential depth of the  steric and electrostatic potentials is equal to
    a specified desired value.

    The parameters of the colloids and the steric and electostatic potentials are given in the RunParameters class.

    This dataclass can be written to and read from a yaml file. The yaml file contains the parameters as key-value
    pairs. Any OpenMM quantities are converted to Quantity objects that can be represented in a readable way in the
    yaml file.

    :param tuned_parameter_name:
        The name of the parameter being tuned.
        This parameter must be part of the RunParameters class.
    :type tuned_parameter_name: str
    :param tuned_parameter_type:
        The type of parameter being tuned.
        Possible values are "Global" and "PerParticle." This corresponds to the parameter type as it appears in the 
        force expressions for the steric and electrostatic potentials.
    :type tuned_parameter_type: str
    :param tuned_type:
        If tuning a per-particle parameter,the type of the colloid whose parameter value will be tuned.
        This type must be present in the radii, masses, and surface_potentials dictionaries (with a dummy value) in the
        RunParameters class.
        If tuning a global parameter, tuned_type must be None.
    :type tuned_type: Optional(str)
    :param tuned_potential_depth:
        The desired potential depth of the combined steric and electrostatic potential.
        The unit of the potential_depth must be compatible with kilojoules per mole and the value must be smaller
        than zero.
    :type tuned_potential_depth: unit.Quantity
    :param other_type:
        If tuning a per-particle parameter, the type of the other colloid that is used to tune the potential depth.
        This type must be present in the radii, masses, and surface_potentials dictionaries in the RunParameters class.
        If tuning a global parameter, tuned_type must be None.
    :type other_type: Optional(str)
    :param plot_filename:
        If not None, filename for the plot of the potential energy of the colloid with the tuned potential.
        Defaults to None.
    :type plot_filename: Optional[str]

    :raises TypeError:
        If any of the quantities has an incompatible unit.
    :raises ValueError:
        If any of the parameters has an invalid value.
        If the tuned_parameter_name is not part of the RunParameters class.
    """
    tuned_parameter_name: str = "surface_potential"
    tuned_parameter_type: str = "PerParticle" 
    tuned_type: Optional[str] = "S"
    tuned_potential_depth: unit.Quantity = field(default_factory=lambda: -7.4331295806289965 * unit.kilojoule_per_mole)
    other_type: Optional[str] = "P"
    plot_filename: Optional[str] = None
