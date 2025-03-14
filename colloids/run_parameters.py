from dataclasses import dataclass, field
import inspect
from typing import Any, Optional
from openmm import unit
from colloids.abstracts import Parameters
import colloids.integrators as integrators
import colloids.update_reporters as update_reporters
from colloids.helper_functions import read_xyz_file
import warnings


@dataclass(order=True, frozen=True)
class RunParameters(Parameters):
    """
    Data class for the parameters of an OpenMM simulation of colloidal particles periodic boundary conditions.

    This dataclass can be written to and read from a yaml file. The yaml file contains the parameters as key-value
    pairs. Any OpenMM quantities are converted to Quantity objects that can be represented in a readable way in the
    yaml file.

    The potentials are given in Hueckel, Hocky, Palacci & Sacanna, Nature 580, 487--490 (2020)
    (see https://doi.org/10.1038/s41586-020-2205-0). Any references to equations or symbols in the code refer to this
    paper.

    :param initial_configuration:
        The path to the initial configuration of the system in an xyz file.
        The filename must end with ".xyz".
        Defaults to "colloids/tests/first_frame.xyz".
    :type initial_configuration: str
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
    :param platform_name:
        The name of the platform to use for the simulation.
        Defaults to "Reference". Other possible choices are "CPU", "CUDA", or "OpenCL".
    :type platform_name: str
    :param potential_temperature:
        The temperature that is used for the colloid potentials.
        The unit of the temperature must be compatible with kelvin and the value must be greater than zero.
        Defaults to 298.0 * unit.kelvin.
    :type potential_temperature: unit.Quantity
    :param integrator:
        The name of the OpenMM integrator to use for the molecular-dynamics simulations.
        Possible choices are "BrownianIntegrator", "LangevinIntegrator", LangevinMiddleIntegrator",
        "NoseHooverIntegrator", "VariableLangevinIntegrator", "VariableVerletIntegrator", and "VerletIntegrator".
        Defaults to "LangevinIntegrator".
    :type integrator: str
    :param integrator_parameters:
        The parameters that are forwarded to initialize the OpenMM integrator.
        Each integrator has specific parameters, and the parameters passed in here must be compatible with the chosen
        integrator. See the corresponding integrator in the OpenMM documentation
        http://docs.openmm.org/latest/api-python/library.html#integrators for the possible arguments (or, alternatively,
        the colloids.integrators module).
        Defaults to sensible values for the LangevinIntegrator (temperature of 298 K, frictionCoeff of
        0.001574074286750681 / ps, stepSize of 0.00317647015905543 ps, and no specified random number seed).
    :type integrator_parameters: dict[str, Any]
    :param brush_density:
        The polymer surface density in the Alexander-de Gennes polymer brush model [i.e., sigma in eq. (1)].
        The unit of the brush_density must be compatible with 1/nanometer^2 and the value must be greater than zero.
        Defaults to 0.09 / ((unit.nano * unit.meter) ** 2).
    :type brush_density: unit.Quantity
    :param brush_length:
        The thickness of the brush in the Alexander-de Gennes polymer brush model [i.e., L in eq. (1)].
        The unit of the brush_length must be compatible with nanometers and the value must be greater than zero.
        Defaults to 10.6 * (unit.nano * unit.meter).
    :type brush_length: unit.Quantity
    :param debye_length:
        The Debye screening length within DLVO theory [i.e., lambda_D].
        The unit of the debye_length must be compatible with nanometers and the value must be greater than zero.
        Defaults to 5.726968 * (unit.nano * unit.meter).
    :type debye_length: unit.Quantity
    :param dielectric_constant:
        The dielectric constant of the solvent [i.e., epsilon].
        The value of the dielectric constant must be greater than zero.
        Defaults to 80.0.
    :type dielectric_constant: float
    :param use_log:
        If True, the electrostatic force uses the more accurate equation involving a logarithm [i.e., eq. (12.5.2) in
        Hunter, Foundations of Colloid Science (Oxford University Press, 2001), 2nd edition] instead of the simpler
        equation that only involves an exponential [i.e., eq. (12.5.5) in Hunter, Foundations of Colloid Science
        (Oxford University Press, 2001), 2nd edition].
        Defaults to False.
    :type use_log: bool
    :param use_tabulated:
        If True, the steric and electrostatic forces are computed based on tabulated functions.
        If False, the steric and electrostatic forces are computed based on algebraic expressions.
        Defaults to False.
    :type use_tabulated: bool
    :param velocity_seed:
        The seed for the random number generator that is used to sample the initial velocities.
        If None, a random seed is used.
        Defaults to None.
    :type velocity_seed: Optional[int]
    :param run_steps:
        The number of time steps to run the simulation.
        The number of time steps must be greater than zero.
        Defaults to 100.
    :type run_steps: int
    :param state_data_interval:
        The interval at which state data is written to a csv file.
        The interval must be greater than zero.
        Defaults to 100.
    :type state_data_interval: int
    :param state_data_filename:
        The name of the csv file to which the state data is written.
        The filename must end with ".csv".
        Defaults to "state_data.csv".
    :type state_data_filename: str
    :param trajectory_interval:
        The interval at which the trajectory is written to a gsd file.
        The interval must be greater than zero.
        Defaults to 100.
    :type trajectory_interval: int
    :param trajectory_filename:
        The name of the gsd file to which the trajectory is written.
        The filename must end with ".gsd".
        Defaults to "trajectory.gsd".
    :type trajectory_filename: str
    :param checkpoint_interval:
        The interval at which the checkpoint is written to a chk file.
        The interval must be greater than zero.
        Defaults to 100.
    :type checkpoint_interval: int
    :param checkpoint_filename:
        The name of the chk file to which the checkpoint is written.
        The filename must end with ".chk".
        Defaults to "checkpoint.chk".
    :type checkpoint_filename: str
    :param minimize_energy_initially:
        If True, the energy of the system is minimized before the simulation starts.
        Defaults to False.
    :type minimize_energy_initially: bool
    :param final_configuration_gsd_filename:
        The name of the gsd file to which the final configuration is written.
        If None, the final configuration is not written to a gsd file.
        The filename must end with ".gsd".
        Defaults to "final_frame.gsd".
    :type final_configuration_gsd_filename: Optional[str]
    :param final_configuration_xyz_filename:
        The name of the xyz file to which the final configuration is written.
        If None, the final configuration is not written to an xyz file.
        The filename must end with ".xyz".
        Defaults to "final_frame.xyz".
    :type final_configuration_xyz_filename: Optional[str]
    :param wall_directions:
        A list of three booleans indicating whether the walls in the x, y, and z directions are active for
        closed-wall simulations with shifted Lennard-Jones potential walls.
        If any of the wall directions is active, epsilon and alpha must be specified.
        Defaults to [False, False, False].
    :type wall_directions: list[bool]
    :param epsilon:
        The unshifted Lennard-Jones potential well-depth for closed-wall simulations with shifted Lennard-Jones
        potential walls.
        If any wall direction is True, epsilon must be not None, its unit must be compatible with kilojoules per mole
        and the value must be greater than zero.
        Defaults to None.
    :type epsilon: Optional[unit.Quantity]
    :param alpha:
        Factor determining the strength of the attractive part of the Lennard-Jones potential for closed-wall
        simulations with shifted Lennard-Jones potential walls.
        If any wall direction is True, alpha must be not None and 0 <= alpha <= 1.
        Note that the force of this potential is only continuous if alpha = 1.
    :type alpha: Optional[float]
    :param use_depletion:
        A boolean indicating whether to turn on the depletion attraction for the simulation.
        If depletion attraction is on, depletion_phi and depletant_radius must be specified.
        Defaults to False.
    :type use_depletion: bool
    :param depletion_phi:
        The number density of polymers in the solution.
        If depletion attraction is on, the value of depletion_phi must not be None and 0 <= depletion_phi <=1.
        Defaults to None.
    :type depletion_phi: Optional[float]
    :param depletant_radius:
        The radius of the polymers in solution for a system with depletion attraction.
        If depletion attraction is on, depletant_radius must not be None, its unit must be compatible with nanometers,
        and the value must be greater than zero.
    :type depletant_radius: Optional[unit.Quantity]
    :param use_gravity: bool
        A boolean indicating whether the gravitational force is turned on for the simulation.
        If true, the gravitational acceleration, particle density, and water density parameters must be specified.
        Defaults to False.
    :param gravitational_acceleration:
        The acceleration due to gravity.
        If gravity is on, the value of the gravitational constant must be specified.
        The unit must be compatible with meters per second squared.
        Defaults to None.
    :type gravitational_acceleration: Optional[unit.Quantity]
    :param water_density:
        The density of water. This is used to compute the effective particle density when calculating the gravitational
        force.
        If gravity is on, the density of water must be specified, its unit must be compatible with grams per centimeter
        cubed, and the value must be greater than zero.
        Defaults to None.
    :type water_density: Optional[unit.Quantity]
    :param particle_density:
        The density of the colloidal particles. This is used to compute the effective particle density when calculating
        the gravitational force.
        If gravity is on, the particle density must be specified, its unit must be compatible with grams per centimeter
        cubed, and the value must be greater than zero.
        Defaults to None.
    :type particle_density: Optional[unit.Quantity]
    :param update_reporter:
        The name of the update reporter used to vary the value of a force-related global parameter over time
        in a simulation.
        Possible choices can be found in the update_reporters.py file.
        If an update reporter is specified, its update reporter parameters must be specified.
        Defaults to None.
    :type update_reporter: Optional[str]
    :param update_reporter_parameters:
        The parameters that are forwarded to the initialization method of the UpdateReporter, if enabled for a
        simulation. Note that the initialization method of the UpdateReporter class expects an OpenMM simulation object
        and an append_file boolean that should not appear in this dictionary.
        Defaults to None.
    :type update_reporter_parameters: Optional[dict[str, Any]]
    :param use_substrate:
        A boolean indicating whether to use a substrate at the bottom of the simulation box.
        A substrate can only be used when all walls are active. The bottom wall is then replaced by the substrate.
        If True, the substrate potential depth must be specified.
        A substrate can only be used with the algebraic colloid potentials (use_tabulated=False).
        Defaults to False.
    :type use_substrate: bool
    :param substrate_type:
        The type of the substrate that is used at the bottom of the simulation box.
        If a substrate is used, the substrate type must not be None. 
        If the substrate_type is "wall," an implicit substrate will be implmented using a Custom Nonbonded Force.
        Otherwise, substrate_type specifies the particle type of explicit substrate particles and this type 
        must appear in the radii, masses, and surface_potentials dictionaries.
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

    :raises TypeError:
        If any of the quantities has an incompatible unit.
    :raises ValueError:
        If any of the parameters has an invalid value.
    """

    initial_configuration: str = "colloids/tests/first_frame.xyz"
    masses: dict[str, unit.Quantity] = field(
        default_factory=lambda: {"P": 1.0 * unit.amu, "N": (95.0 / 105.0) ** 3 * unit.amu})
    radii: dict[str, unit.Quantity] = field(
        default_factory=lambda: {"P": 105.0 * (unit.nano * unit.meter), "N": 95.0 * (unit.nano * unit.meter)})
    surface_potentials: dict[str, unit.Quantity] = field(
        default_factory=lambda: {"P": 44.0 * (unit.milli * unit.volt), "N": -54.0 * (unit.milli * unit.volt)})
    psi_scale: float = 1.0
    platform_name: str = "Reference"
    potential_temperature: unit.Quantity = field(default_factory=lambda: 298.0 * unit.kelvin)
    integrator: str = "LangevinIntegrator"
    integrator_parameters: dict[str, Any] = field(
        default_factory=lambda: {
            "temperature": 298.0 * unit.kelvin,
            "stepSize": 0.00317647015905543 * (unit.pico * unit.second),
            "frictionCoeff": 0.001574074286750681 / (unit.pico * unit.second),
            "randomNumberSeed": None
        })
    brush_density: unit.Quantity = field(default_factory=lambda: 0.09 / ((unit.nano * unit.meter) ** 2))
    brush_length: unit.Quantity = field(default_factory=lambda: 10.6 * (unit.nano * unit.meter))
    debye_length: unit.Quantity = field(default_factory=lambda: 5.726968 * (unit.nano * unit.meter))
    dielectric_constant: float = 80.0
    cutoff_factor: float = 21.0
    use_log: bool = False
    use_tabulated: bool = False
    velocity_seed: Optional[int] = None
    run_steps: int = 100
    state_data_interval: int = 100
    state_data_filename: str = "state_data.csv"
    trajectory_interval: int = 100
    trajectory_filename: str = "trajectory.gsd"
    checkpoint_interval: int = 100
    checkpoint_filename: str = "checkpoint.chk"
    minimize_energy_initially: bool = False
    final_configuration_gsd_filename: Optional[str] = "final_frame.gsd"
    final_configuration_xyz_filename: Optional[str] = "final_frame.xyz"
    epsilon: Optional[unit.Quantity] = None
    alpha: Optional[float] = None
    wall_directions: list[bool] = field(default_factory=lambda: [False, False, False])
    use_depletion: bool = False
    depletion_phi: Optional[float] = None
    depletant_radius: Optional[unit.Quantity] = None
    use_gravity: bool = False
    gravitational_acceleration: Optional[unit.Quantity] = None
    water_density: Optional[unit.Quantity] = None
    particle_density: Optional[unit.Quantity] = None
    update_reporter: Optional[str] = None
    update_reporter_parameters: Optional[dict[str, Any]] = None
    use_substrate: bool = False
    substrate_type: Optional[str] = None
    use_snowman: bool = False
    snowman_seed: Optional[int] = None
    snowman_bond_types: Optional[dict[str, str]] = None
    snowman_distances: Optional[dict[str, unit.Quantity]] = None

    def __post_init__(self) -> None:
        """Check if the parameters are valid after initialization."""
        if not self.initial_configuration.endswith(".xyz"):
            raise ValueError("The filename of the initial configuration must end with '.xyz'")
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
        if self.platform_name not in ["Reference", "CPU", "CUDA", "OpenCL"]:
            raise ValueError("The platform name must be 'Reference', 'CPU', 'CUDA', or 'OpenCL'.")
        possible_integrators = [name for name, _ in inspect.getmembers(integrators, inspect.isfunction)]
        if self.integrator not in possible_integrators:
            raise ValueError(f"Integrator {self.integrator} not available, the integrator must be one of the "
                             f"following: {', '.join(possible_integrators)}.")
        integrator_getter = getattr(integrators, self.integrator)
        try:
            integrator_getter(**self.integrator_parameters)
        except TypeError:
            raise TypeError(f"Integrator {self.integrator} does not accept the given arguments "
                            f"{self.integrator_parameters}. The expected signature is "
                            f"{inspect.signature(integrator_getter)}")
        if not self.potential_temperature.unit.is_compatible(unit.kelvin):
            raise TypeError("The temperature must have a unit compatible with kelvin.")
        if self.potential_temperature <= 0.0 * unit.kelvin:
            raise ValueError("The temperature must be greater than zero.")
        if not self.brush_density.unit.is_compatible((unit.nano * unit.meter) ** (-2)):
            raise TypeError("The brush density must have a unit compatible with 1/nanometer^2.")
        if self.brush_density <= 0.0 * ((unit.nano * unit.meter) ** (-2)):
            raise ValueError("The brush density must be greater than zero.")
        if not self.brush_length.unit.is_compatible(unit.nano * unit.meter):
            raise TypeError("The brush length must have a unit compatible with nanometers.")
        if self.brush_length <= 0.0 * (unit.nano * unit.meter):
            raise ValueError("The brush length must be greater than zero.")
        if not self.debye_length.unit.is_compatible(unit.nano * unit.meter):
            raise TypeError("The Debye length must have a unit compatible with nanometers.")
        if self.debye_length <= 0.0 * (unit.nano * unit.meter):
            raise ValueError("The Debye length must be greater than zero.")
        if self.dielectric_constant <= 0.0:
            raise ValueError("The dielectric constant must be greater than zero.")
        if self.run_steps == 0:
            warnings.warn("The number of time steps is zero.")
        if self.run_steps < 0:
            raise ValueError("The number of time steps must be greater than or equal to zero.")
        if self.state_data_interval <= 0:
            raise ValueError("The state data interval must be greater than zero.")
        if not self.state_data_filename.endswith(".csv"):
            raise ValueError("The filename of the state data must end with '.csv'.")
        if self.trajectory_interval <= 0:
            raise ValueError("The trajectory interval must be greater than zero.")
        if not self.trajectory_filename.endswith(".gsd"):
            raise ValueError("The filename of the trajectory must end with '.gsd'.")
        if self.checkpoint_interval <= 0:
            raise ValueError("The checkpoint interval must be greater than zero.")
        if not self.checkpoint_filename.endswith(".chk"):
            raise ValueError("The filename of the checkpoint must end with '.chk'.")
        if (self.final_configuration_gsd_filename is not None
                and not self.final_configuration_gsd_filename.endswith(".gsd")):
            raise ValueError("The filename of the final configuration must end with '.gsd'.")
        if (self.final_configuration_xyz_filename is not None
                and not self.final_configuration_xyz_filename.endswith(".xyz")):
            raise ValueError("The filename of the final configuration must end with '.xyz'.")
        if isinstance(self.wall_directions, str):
            raise ValueError("Wall directions was parsed as a string although it should be a list of bools. "
                             "Make sure that the yaml file is correctly formatted and that there is space after each "
                             "dash in the list of wall directions.")
        if len(self.wall_directions) != 3:
            raise ValueError("Wall directions must be specified for three dimensions.")
        if any(self.wall_directions):
            if self.epsilon is None:
                raise ValueError("Epsilon must be specified if walls are active.")
            if not self.epsilon.unit.is_compatible(unit.kilojoule_per_mole):
                raise TypeError("Epsilon must have a unit compatible with kilojoules per mole.")
            if self.epsilon <= 0.0 * unit.kilojoule_per_mole:
                raise ValueError("epsilon must be greater than zero.")
            if self.alpha is None:
                raise ValueError("Alpha must be specified if walls are active.")
            if not 0.0 <= self.alpha <= 1.0:
                raise ValueError("Alpha must be between zero and one.")
        else:
            if self.epsilon is not None:
                raise ValueError("Epsilon must not be specified if walls are not active.")
            if self.alpha is not None:
                raise ValueError("Alpha must not be specified if walls are not active.")
        if self.use_depletion:
            if self.depletion_phi is None:
                raise ValueError("Depletion phi must be specified if depletion is on.")
            if not 0.0 <= self.depletion_phi <= 1.0:
                raise ValueError("Depletion phi must be between zero and one.")
            if self.depletant_radius is None:
                raise ValueError("Depletant radius must be specified if depletion is on.")
            if not self.depletant_radius.unit.is_compatible(
                    unit.nano * unit.meter):
                raise TypeError("Depletant radius must have a unit compatible with nanometers.")
            if self.depletant_radius <= 0.0 * (unit.nano * unit.meter):
                raise ValueError("Depletant radius must be greater than zero.")
            for t in self.radii:
                if self.depletant_radius / self.radii[t] > 0.1547:
                    warnings.warn("Size ratio of depletant to colloid particles is too large. "
                                  "Analytical computation of depletion potential may be invalid."
                                  "See Dijkstra et. al., Journal of Physics: Condensed Matter, 1999, Volume 11, "
                                  "pp 10079 - 10106.")
        else:
            if self.depletion_phi is not None:
                raise ValueError("Depletion phi must not be specified if depletion potential is not on.")
            if self.depletant_radius is not None:
                raise ValueError("Depletant radius must not be specified if depletion potential is not on.")
        if self.use_gravity:
            if self.gravitational_acceleration is None:
                raise ValueError("Gravitational acceleration must be specified if gravity is on.")
            if not self.gravitational_acceleration.unit.is_compatible(unit.meter / unit.second ** 2):
                raise TypeError(
                    "The gravitational acceleration must have a unit compatible with meters per second squared.")
            if self.gravitational_acceleration <= 0.0 * (unit.meter / unit.second ** 2):
                raise ValueError("The gravitational acceleration must be greater than zero.")
            if self.water_density is None:
                raise ValueError("Density of water must be specified if gravity is on.")
            if not self.water_density.unit.is_compatible(unit.gram / (unit.centi * unit.meter)**3):
                raise TypeError("The water density must have a unit compatible with grams per centimeter cubed.")
            if self.water_density <= 0.0 * (unit.gram / (unit.centi * unit.meter)**3):
                raise ValueError("The water density must be greater than zero.")
            if self.particle_density is None:
                raise ValueError("Density of particle must be specified if gravity is on.")
            if not self.particle_density.unit.is_compatible(unit.gram / (unit.centi * unit.meter)**3):
                raise TypeError("The particle density must have a unit compatible with grams per centimeter cubed.")
            if self.particle_density <= 0.0 * (unit.gram / (unit.centi * unit.meter)**3):
                raise ValueError("The particle density must be greater than zero.")
            if not all(self.wall_directions):
                raise ValueError("Gravity can only be turned on if all walls are active and, hence, no periodic "
                                 "boundary conditions are present.")
        else:
            if self.gravitational_acceleration is not None:
                raise ValueError("Gravitational acceleration must not be specified if gravity is not on.")
            if self.water_density is not None:
                raise ValueError("Density of water must not be specified if gravity is not on.")
            if self.particle_density is not None:
                raise ValueError("Density of particle must not be specified if gravity is not on.")
        if self.update_reporter is not None:
            possible_update_reporters = [name for name, _ in inspect.getmembers(update_reporters, inspect.isclass)
                                         if name != "ABC" and "Abstract" not in name]
            if self.update_reporter not in possible_update_reporters:
                raise ValueError(f"Update reporter {self.update_reporter} not available, the update reporter must be one of the following:",
                                 f"{', '.join(possible_update_reporters)}.")
            if self.update_reporter_parameters is None:
                raise ValueError("Update-reporter parameters must be specified if the update reporter is on.")
            if "simulation" in self.update_reporter_parameters or "append_file" in self.update_reporter_parameters:
                raise ValueError("Update-reporter parameters should not contain simulation and append_file keys.")
        else:
            if self.update_reporter_parameters is not None:
                raise ValueError("Update-reporter parameters must not be specified if the update reporter is not on.")
        if self.use_substrate:
            if not all(self.wall_directions):
                raise ValueError("A substrate can only be used if all walls are active.")
            if self.substrate_type is None:
                raise ValueError("The substrate type must be specified if a substrate is used.")
            if not self.substrate_type == "wall":
                if self.substrate_type not in self.radii:
                    raise ValueError("The substrate type must be in the radii dictionary.")
                if self.substrate_type not in self.masses:
                    raise ValueError("The substrate type must be in the masses dictionary.")
                if self.masses[self.substrate_type] != 0.0 * unit.amu:
                    warnings.warn("The mass of the substrate type is not zero. Substrate will move during the simulation.")
            if self.substrate_type not in self.surface_potentials:
                raise ValueError("The substrate type must be in the surface potentials dictionary.")
            if self.use_tabulated:
                raise ValueError("A substrate can only be used with the algebraic colloid potentials.")
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

    def check_types_of_initial_configuration(self):
        """
        Check if the types of the initial configuration are consistent with the masses, radii, and surface-potentials
        dictionaries.

        :raises ValueError:
            If the types of the initial configuration are not consistent with the masses, radii, and surface-potentials
            dictionaries.
        """
        types_from_file, _, _ = read_xyz_file(self.initial_configuration)
        types = list(dict.fromkeys(types_from_file))
        for t in types:
            if t not in self.masses:
                raise ValueError(f"Type {t} of the initial configuration is not in masses dictionary.")
            if t not in self.radii:
                raise ValueError(f"Type {t} of the initial configuration is not in radii dictionary.")
            if t not in self.surface_potentials:
                raise ValueError(f"Type {t} of the initial configuration is not in surface potentials dictionary.")
            if t == self.substrate_type:
                raise ValueError(f"Type {t} of the initial configuration cannot be the substrate type. Use "
                                 f"checkpoints to restart simulations with a substrate.")
            if self.snowman_bond_types is not None and t in self.snowman_bond_types.values():
                raise ValueError(f"Type {t} of the initial configuration cannot be a snowman head type. Use "
                                 f"checkpoints to restart simulations with snowman colloids.")
        for t in self.masses:
            if t not in types:
                if t == self.substrate_type:
                    continue
                if self.snowman_bond_types is not None and t in self.snowman_bond_types.values():
                    continue
                raise ValueError(f"Type {t} of the masses dictionary is not in the initial configuration.")
        for t in self.radii:
            if t not in types:
                if t == self.substrate_type:
                    continue
                if self.snowman_bond_types is not None and t in self.snowman_bond_types.values():
                    continue
                raise ValueError(f"Type {t} of the radii dictionary is not in the initial configuration.")
        for t in self.surface_potentials:
            if t not in types:
                if t == self.substrate_type:
                    continue
                if self.snowman_bond_types is not None and t in self.snowman_bond_types.values():
                    continue
                raise ValueError(f"Type {t} of the surface potentials dictionary is not in the initial configuration.")


if __name__ == '__main__':
    RunParameters(initial_configuration="tests/first_frame.xyz").to_yaml("example.yaml")
    parameters = RunParameters.from_yaml("example.yaml")
    print(parameters)