from dataclasses import dataclass, field
from typing import Optional, Union
import warnings
from openmm import unit
from colloids.abstracts import Parameters


@dataclass(order=True, frozen=True)
class ConfigurationParameters(Parameters):
    """
    Data class for the parameters of the colloids configuration to be created for an OpenMM simulation.

    This dataclass can be written to and read from a yaml file. The yaml file contains the parameters as key-value
    pairs. Any OpenMM quantities are converted to Quantity objects that can be represented in a readable way in the
    yaml file.

    The base configuration consists of two types of colloids. It is created by placing the colloidal particles with the
    bigger radius on a cubic lattice structure. The colloidal particles with a smaller radius are then placed as
    satellites around these centers.

    After the base configuration has been created, it can be modified by adding a substrate at the bottom of the
    simulation box, and by adding snowman heads to given colloid types.

    :param lattice_type:
        The type of lattice to be used for the configuration. The possible options are "sc" (simple
        cubic), "fcc" (face-centered cubic), and "bcc" (body-centered cubic). Defaults to "sc".
    :type lattice_type: str
    :param lattice_spacing_factor:
        The factor by which the diameter of the bigger center colloid is multiplied to determine the lattice constant
        of the cubic lattice.
        Must be positive.
        Defaults to 8.0.
    :type lattice_spacing_factor: float
    :param lattice_repeats:
        The number of repeats of the lattice in the x, y, and z directions. If an integer is given, the same number of
        repeats is used in all directions.
        Every repeat must be positive.
        Defaults to 8.
    :type lattice_repeats: Union[int, list[int]]
    :param orbit_factor:
        The factor by which the sum of the diameters of the two types of colloids and twice the brush length is
        multiplied to determine the distance of the satellites from the center colloids.
        Must be positive.
        Defaults to 1.3.
    :type orbit_factor: float
    :param satellites_per_center:
        The number of satellites to be placed around each center colloid.
        Must be non-negative.
        Defaults to 1.
    :type satellites_per_center: int
    :param padding_factor:
        The factor by which the radius of the bigger center colloid is multiplied to determine the distance of the
        satellites from the walls of the simulation box.
        Must be non-negative.
        Defaults to 0.0.
    :type padding_factor: float
    :param masses:
        The masses of the different types of colloidal particles that appear in the initial configuration file.
        The keys of the dictionary are the types of the colloidal particles and the values are the masses.
        The unit of the masses must be compatible with atomic mass units and the values must be greater than zero,
        except for immobile particles (as the substrate), which should have a mass of zero.
        Defaults to {"P": 1.0 * unit.amu, "N": (95.0 / 105.0) ** 3 * unit.amu}.
    :type masses: dict[str, unit.Quantity]
    :param radii:
        The radii of the different types of colloidal particles that appear in the initial configuration file.
        The keys of the dictionary are the types of the colloidal particles and the values are the radii.
        The unit of the radii must be compatible with nanometers and the values must be greater than zero.
        Defaults to {"P": 105.0 * (unit.nano * unit.meter), "N": 95.0 * (unit.nano * unit.meter)}.
    :type radii: dict[str, unit.Quantity]
    :param surface_potentials:
        The surface potentials of the different types of colloidal particles that appear in the initial configuration
        file.
        The keys of the dictionary are the types of the colloidal particles and the values are the surface potentials.
        The unit of the surface potentials must be compatible with millivolts.
        Defaults to {"P": 44.0 * (unit.milli * unit.volt), "N": -54.0 * (unit.milli * unit.volt)}.
    :type surface_potentials: dict[str, unit.Quantity]
    :param use_substrate:
        A boolean indicating whether to use a substrate at the bottom of the simulation box.
        A substrate can only be used when all walls are active. The bottom wall is then replaced by the substrate.
        If True, the substrate radius and the substrate potential depth must be specified.
        A substrate can only be used with the algebraic colloid potentials (use_tabulated=False).
        Defaults to False.
    :type use_substrate: bool
    :param substrate_type:
        The type of the substrate that is used at the bottom of the simulation box.
        If a substrate is used, the substrate type must not be None and it must appear in the radii, masses, and
        surface_potentials dictionaries.
        Defaults to None.
    :type substrate_type: Optional[str]
    :param use_snowman:
        A boolean indicating whether to use the snowman colloids in the simulation.
        In a snowman colloid, a colloidal head particle is attached to a colloidal base particle at a fixed distance.
        If True, the snowman bond types, the snowman distances, and optionally the snowman seed must be specified.
        Defaults to False.
    :type use_snowman: bool
    :param snowman_seed:
        The seed for the random number generator that is used to sample the positions of the snowman heads.
        If zero or smaller than zero, the positions of the snowman heads are not randomized.
        If None, a random seed is used.
        Defaults to None.
    :type snowman_seed: Optional[int]
    :param snowman_bond_types:
        Dictionary mapping from the type of the base particle to the type of the head particle in the snowman colloid.
        Snowman heads are attached to every base particle type in this dictionary.
        Every snowman head type must appear in the masses, radii, and surface potentials dictionaries.
        Defaults to None.
    :type snowman_bond_types: Optional[dict[str, str]]
    :param snowman_distances:
        Dictionary mapping from the type of the base particle to the desired distance to the snowman head.
        Every type appearing in the snowman bond types dictionary must have a corresponding distance in this dictionary.
        The unit of every distance must be compatible with nanometers and the value must be greater than zero.
        Defaults to None.
    :type snowman_distances: Optional[dict[str, unit.Quantity]]
    """
    lattice_type: str = "sc"
    lattice_spacing_factor: float = 8.0
    lattice_repeats: Union[int, list[int]] = 8
    orbit_factor: float = 1.3
    satellites_per_center: int = 1
    padding_factor: float = 0.0
    masses: dict[str, unit.Quantity] = field(
        default_factory=lambda: {"P": 1.0 * unit.amu, "N": (95.0 / 105.0) ** 3 * unit.amu})
    radii: dict[str, unit.Quantity] = field(
        default_factory=lambda: {"P": 105.0 * (unit.nano * unit.meter), "N": 95.0 * (unit.nano * unit.meter)})
    surface_potentials: dict[str, unit.Quantity] = field(
        default_factory=lambda: {"P": 44.0 * (unit.milli * unit.volt), "N": -54.0 * (unit.milli * unit.volt)})
    use_substrate: bool = False
    substrate_type: Optional[str] = None
    use_snowman: bool = False
    snowman_seed: Optional[int] = None
    snowman_bond_types: Optional[dict[str, str]] = None
    snowman_distances: Optional[dict[str, unit.Quantity]] = None

    def __post_init__(self):
        if self.lattice_spacing_factor <= 0.0:
            raise ValueError("The lattice spacing factor must be positive.")
        if isinstance(self.lattice_repeats, int):
            if self.lattice_repeats <= 0:
                raise ValueError("The number of lattice repeats must be positive.")
        else:
            if not (isinstance(self.lattice_repeats, list)
                    and all(isinstance(repeat, int) for repeat in self.lattice_repeats)
                    and len(self.lattice_repeats) == 3):
                raise TypeError("The lattice repeats must be an integer or a list of three integers.")
            if not all(repeat > 0 for repeat in self.lattice_repeats):
                raise ValueError("All lattice repeats must be positive.")
        if self.orbit_factor <= 0.0:
            raise ValueError("The orbit factor must be positive.")
        if self.satellites_per_center < 0:
            raise ValueError("The number of satellites per center must be zero or positive.")
        if self.lattice_type not in ["sc", "bcc", "fcc"]:
            raise ValueError("The lattice type must be sc, bcc, or fcc.")
        if self.padding_factor < 0.0:
            raise ValueError("The padding factor must be non-negative.")

        for t in self.masses:
            if not self.masses[t].unit.is_compatible(unit.amu):
                raise TypeError(f"Mass of type {t} must have a unit compatible with atomic mass units.")
            if self.masses[t] < 0.0 * unit.amu:
                raise ValueError(f"Mass of type {t} must be greater than zero.")
            if t != self.substrate_type and self.masses[t] == 0.0 * unit.amu:
                raise ValueError(f"Mass of type {t} must be greater than zero unless it is the substrate.")
            if t not in self.radii:
                raise ValueError(f"Type {t} of the masses dictionary is not in radii dictionary.")
            if t not in self.surface_potentials:
                raise ValueError(f"Type {t} of the masses dictionary is not in surface potentials dictionary.")
        for t in self.radii:
            if not self.radii[t].unit.is_compatible(unit.nano * unit.meter):
                raise TypeError(f"Radius of type {t} must have a unit compatible with nanometers.")
            if self.radii[t] <= 0.0 * (unit.nano * unit.meter):
                raise ValueError(f"Radius of type {t} must be greater than zero.")
            if t not in self.masses:
                raise ValueError(f"Type {t} of the radii dictionary is not in masses dictionary.")
            if t not in self.surface_potentials:
                raise ValueError(f"Type {t} of the radii dictionary is not in surface potentials dictionary.")
        for t in self.surface_potentials:
            if not self.surface_potentials[t].unit.is_compatible(unit.milli * unit.volt):
                raise TypeError(f"Surface potential of type {t} must have a unit compatible with millivolts.")
            if t not in self.masses:
                raise ValueError(f"Type {t} of the surface potentials dictionary is not in masses dictionary.")
            if t not in self.radii:
                raise ValueError(f"Type {t} of the surface potentials dictionary is not in radii dictionary.")

        if self.use_substrate:
            if self.substrate_type is None:
                raise ValueError("The substrate type must be specified if a substrate is used.")
            if self.substrate_type not in self.radii:
                raise ValueError("The substrate type must be in the radii dictionary.")
            if self.substrate_type not in self.masses:
                raise ValueError("The substrate type must be in the masses dictionary.")
            if self.substrate_type not in self.surface_potentials:
                raise ValueError("The substrate type must be in the surface potentials dictionary.")
            if self.masses[self.substrate_type] != 0.0 * unit.amu:
                warnings.warn("The mass of the substrate type is not zero. Substrate will move during the simulation.")
        else:
            if self.substrate_type is not None:
                raise ValueError("The substrate type must not be specified if a substrate is not used.")
        if self.use_snowman:
            if self.snowman_bond_types is None:
                raise ValueError("Snowman bond types must be specified if snowman is on.")
            if self.snowman_distances is None:
                raise ValueError("Snowman distances must be specified if snowman is on.")
            for t in self.snowman_bond_types:
                st = self.snowman_bond_types[t]
                if st not in self.masses:
                    raise ValueError(f"Type {st} of the snowman bond types dictionary is not in masses dictionary.")
                if st not in self.radii:
                    raise ValueError(f"Type {st} of the snowman bond types dictionary is not in radii dictionary.")
                if st not in self.surface_potentials:
                    raise ValueError(f"Type {st} of the snowman bond types dictionary is not in surface potentials "
                                     f"dictionary.")
                if t not in self.snowman_distances:
                    raise ValueError(f"Type {t} of the snowman bond types dictionary is not in snowman distances "
                                     f"dictionary.")
            for t in self.snowman_distances:
                if t not in self.snowman_bond_types:
                    raise ValueError(f"Type {t} of the snowman distances dictionary is not in snowman bond types "
                                     f"dictionary.")
                if not self.snowman_distances[t].unit.is_compatible(unit.nano * unit.meter):
                    raise TypeError(f"Distance of type {t} must have a unit compatible with nanometers.")
                if self.snowman_distances[t] <= 0.0 * (unit.nano * unit.meter):
                    raise ValueError(f"Distance of type {t} must be greater than zero.")
        else:
            if self.snowman_bond_types is not None:
                raise ValueError("Snowman bond types must not be specified if snowman is not on.")
            if self.snowman_distances is not None:
                raise ValueError("Snowman distances must not be specified if snowman is not on.")
            if self.snowman_seed is not None:
                raise ValueError("Snowman seed must not be specified if snowman is not on.")
