from dataclasses import dataclass, field
from typing import Optional
from openmm import unit
from colloids.abstracts import Parameters
from colloids.units import energy_unit, length_unit


@dataclass(order=True, frozen=True)
class TIParameters(Parameters):
    """
    Data class for the parameters of a thermodynamic-integration (Frenkel-Ladd / Einstein-crystal)
    run of PACSim.

    These parameters are supplied in a *separate* YAML file that is passed to ``pacsim-run`` with
    the ``-t/--ti_file`` option, in addition to the usual run-parameter YAML file. When a TI file is
    present, a harmonic (Einstein-crystal) restraint is added to the system that ties every mobile
    particle to a fixed reference position with the potential

        u(r) = coupling * spring_constant * |r - r0|^2

    (see ``colloids.harmonic_restraint.HarmonicRestraint``). One value of ``coupling`` (the
    dimensionless Einstein coupling parameter lambda_ein in [0, 1]) is simulated per ``pacsim-run``
    invocation; a full free-energy calculation runs one window per coupling value and combines the
    per-window energies afterwards (see the develop-crystal-TI analysis scripts).

    The restraint energy and the PACS component energies are recorded per frame in the GSD
    trajectory log by the GSDReporter, because the restraint is placed in its own force group with a
    distinct name. No additional reporter is needed.

    :param spring_constant:
        The Einstein spring constant Lambda_E of the harmonic restraint.
        The unit must be compatible with kilojoules per mole per squared nanometer and the value
        must be greater than zero. There is no default; it must be specified.
    :type spring_constant: Optional[unit.Quantity]
    :param coupling:
        The dimensionless Einstein coupling parameter lambda_ein for this window. It scales the
        strength of the whole restraint and must satisfy 0 <= coupling <= 1.
        Defaults to 1.0.
    :type coupling: float
    :param reference_configuration:
        The path to a GSD file whose frame supplies the reference (lattice) positions r0 that the
        particles are restrained to. If None, the initial configuration of the run (the
        ``initial_configuration`` of the run-parameter file) is used as the reference.
        The filename must end with ".gsd".
        Defaults to None.
    :type reference_configuration: Optional[str]
    :param reference_frame_index:
        The index of the frame in ``reference_configuration`` used for the reference positions.
        Defaults to -1 (the last frame).
    :type reference_frame_index: int
    :param fix_center_of_mass:
        If True, a CMMotionRemover is added to the system so that the center of mass stays fixed, as
        required by the Einstein-crystal method (this removes the quasi-divergence of the
        thermodynamic-integration integrand at small coupling). If the run already includes immobile
        substrate particles or walls that break translational invariance, this may be set to False.
        Defaults to True.
    :type fix_center_of_mass: bool
    :param restrain_types:
        A list of particle-type names that should be restrained. If None, every mobile particle
        (mass greater than zero) is restrained; immobile substrate particles (mass zero) are never
        restrained. If given, only particles of these types are restrained.
        Defaults to None.
    :type restrain_types: Optional[list[str]]
    :param use_virtual_particles:
        If True (the default), the Einstein restraint is applied with the "virtual particle" method:
        a fixed (mass-zero) virtual particle is added at each reference position and the real particle
        is tied to it by a harmonic bond. This is numerically stable on the CUDA and OpenCL platforms.
        If False, the restraint is a single CustomExternalForce; this is simpler and gives an identical
        free energy, but the combination of a CustomExternalForce (with per-particle parameters) and a
        periodic nonbonded force diverges to NaN on both GPU platforms (all precisions), so it must
        only be used on the CPU or Reference platform (see develop-crystal-TI/openmm_opencl_bug).
        Because virtual particles have mass zero, the virtual-particle method cannot be combined with
        an explicit (mass-zero) substrate; an error is raised if the initial configuration contains
        immobile particles.
        Defaults to True.
    :type use_virtual_particles: bool

    :raises TypeError:
        If the spring constant has an incompatible unit.
    :raises ValueError:
        If the spring constant is not specified or is not greater than zero.
        If the coupling is not in the interval [0, 1].
        If the reference configuration filename does not end with ".gsd".
        If restrain_types is given but empty.
    """

    spring_constant: Optional[unit.Quantity] = None
    coupling: float = 1.0
    reference_configuration: Optional[str] = None
    reference_frame_index: int = -1
    fix_center_of_mass: bool = True
    restrain_types: Optional[list[str]] = None
    use_virtual_particles: bool = True

    def __post_init__(self) -> None:
        """Check if the parameters are valid after initialization."""
        spring_constant_unit = energy_unit / (length_unit ** 2)
        if self.spring_constant is None:
            raise ValueError("The spring constant must be specified in the TI parameter file.")
        if not self.spring_constant.unit.is_compatible(spring_constant_unit):
            raise TypeError("The spring constant must have a unit compatible with kilojoules per "
                            "mole per squared nanometer.")
        if self.spring_constant <= 0.0 * spring_constant_unit:
            raise ValueError("The spring constant must be greater than zero.")
        if not 0.0 <= self.coupling <= 1.0:
            raise ValueError("The coupling must be between zero and one.")
        if (self.reference_configuration is not None
                and not self.reference_configuration.endswith(".gsd")):
            raise ValueError("The filename of the reference configuration must end with '.gsd'.")
        if self.restrain_types is not None:
            if isinstance(self.restrain_types, str):
                raise ValueError("restrain_types was parsed as a string although it should be a "
                                 "list of type names. Make sure the yaml list is correctly "
                                 "formatted (a space after each dash).")
            if len(self.restrain_types) == 0:
                raise ValueError("restrain_types must not be empty if specified.")


if __name__ == '__main__':
    TIParameters(spring_constant=1000.0 * (energy_unit / (length_unit ** 2))).to_yaml("example_ti.yaml")
    print(TIParameters.from_yaml("example_ti.yaml"))
