import argparse
import math
from typing import Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
import openmm
from openmm import unit
from scipy.optimize import minimize, root_scalar
from colloids import ColloidPotentialsAlgebraic, ColloidPotentialsParameters
from colloids.run_parameters import RunParameters
from colloids.colloids_tune.tune_parameters import TuneParameters


class ExampleAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        default_parameters = TuneParameters()
        default_parameters.to_yaml("example_tune.yaml")
        parser.exit()


def tune_surface_potential(colloid_potentials: ColloidPotentialsAlgebraic, other_radius: unit.Quantity,
                           other_surface_potential: unit.Quantity, tuned_radius: unit.Quantity,
                           tuned_potential_depth: unit.Quantity, plot_filename: Optional[str]) -> unit.Quantity:
    """
    Tune the surface potential of a colloid with a given radius so that the potential depth of the combined steric
    and electrostatic potentials with another colloid is equal to the given potential depth.

    :param colloid_potentials:
        The algebraic colloid potentials containing the steric and electrostatic potentials.
    :type colloid_potentials: ColloidPotentialsAlgebraic
    :param other_radius:
        The radius of the other colloid.
        The unit of the radius must be compatible with nanometers and the value must be greater than zero.
    :type other_radius: unit.Quantity
    :param other_surface_potential:
        The surface potential of the other colloid.
        The unit of the surface_potential must be compatible with millivolts.
    :type other_surface_potential: unit.Quantity
    :param tuned_radius:
        The radius of the colloid whose surface potential will be tuned.
        The unit of the radius must be compatible with nanometers and the value must be greater than zero.
    :type tuned_radius: unit.Quantity
    :param tuned_potential_depth:
        The desired potential depth of the combined steric and electrostatic potential with the other colloid.
        The unit of the potential_depth must be compatible with kilojoules per mole and the value must be smaller
        than zero.
    :type tuned_potential_depth: unit.Quantity
    :param plot_filename:
        If not None, the filename of the plot for the potential energy of the colloid with the tuned potential.
    :type plot_filename: Optional[str]

    :return:
        The tuned surface potential of the colloid in millivolts.
    :rtype: unit.Quantity

    :raises TypeError:
        If other_radius, other_radius, or tuned_potential_depth is not a Quantity with a proper unit (via the
        abstract base class).
    :raises ValueError:
        If other_radius or tuned_radius is not greater than zero.
        If the tuned_potential_depth is not smaller than zero.
    """
    if not other_radius.unit.is_compatible(unit.nano * unit.meter):
        raise TypeError("The radius must be a Quantity with a unit compatible with nanometers.")
    if not other_radius.value_in_unit(unit.nano * unit.meter) > 0.0:
        raise ValueError("The radius must be greater than zero.")
    if not other_surface_potential.unit.is_compatible(unit.milli * unit.volt):
        raise TypeError("The positive_surface_potential must be a Quantity with a unit compatible with millivolts.")
    if not tuned_radius.unit.is_compatible(unit.nano * unit.meter):
        raise TypeError("The tune_radius must be a Quantity with a unit compatible with nanometers.")
    if not tuned_radius.value_in_unit(unit.nano * unit.meter) > 0.0:
        raise ValueError("The tune_radius must be greater than zero.")
    if not tuned_potential_depth.unit.is_compatible(unit.kilojoule_per_mole):
        raise TypeError("The potential_depth must be a Quantity with a unit compatible with kilojoule per mole.")
    if not tuned_potential_depth.value_in_unit(unit.kilojoule_per_mole) < 0.0:
        raise ValueError("The potential_depth must be less than zero.")

    system = openmm.System()
    platform = openmm.Platform.getPlatformByName("Reference")
    dummy_integrator = openmm.LangevinIntegrator(0.0, 0.0, 0.0)
    electrostatic_potential = colloid_potentials.set_up_electrostatic_potential()
    steric_potential = colloid_potentials.set_up_steric_potential()
    electrostatic_potential.setNonbondedMethod(electrostatic_potential.NoCutoff)
    steric_potential.setNonbondedMethod(steric_potential.NoCutoff)
    system.addParticle(1.0 * unit.amu)
    system.addParticle(1.0 * unit.amu)
    # Force the surface potential of the other colloid to be positive.
    electrostatic_potential.addParticle([other_radius.value_in_unit(unit.nano * unit.meter),
                                         abs(other_surface_potential.value_in_unit(unit.milli * unit.volt)),
                                         False])
    # We use the global electrostatic prefactor to tune the (negative) surface potential.
    electrostatic_potential.addParticle([tuned_radius.value_in_unit(unit.nano * unit.meter), 1.0, False])
    steric_potential.addParticle([other_radius.value_in_unit(unit.nano * unit.meter), False])
    steric_potential.addParticle([tuned_radius.value_in_unit(unit.nano * unit.meter), False])
    system.addForce(electrostatic_potential)
    system.addForce(steric_potential)
    context = openmm.Context(system, dummy_integrator, platform)
    original_electrostatic_prefactor = context.getParameter("electrostatic_prefactor")
    radius_sum = (other_radius + tuned_radius).value_in_unit(unit.nano * unit.meter)

    def potential_energy(surface_separation: Sequence[float], surface_potential: float) -> float:
        # Surface separation must be a numpy array for the minimize function.
        assert len(surface_separation) == 1
        assert surface_potential <= 0.0
        # We use the global electrostatic prefactor to tune the negative surface potential.
        context.setParameter("electrostatic_prefactor", original_electrostatic_prefactor * surface_potential)
        context.setPositions([openmm.Vec3(0.0, 0.0, 0.0),
                              openmm.Vec3(radius_sum + surface_separation[0], 0.0, 0.0)])
        return context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)

    def deviation_potential_energy(surface_potential: float) -> float:
        minimum_energy_result = minimize(potential_energy, np.array([10.0]), args=(surface_potential,), tol=1.0e-3)
        if not minimum_energy_result.success:
            raise RuntimeError(minimum_energy_result.message + " Minimization failed.")
        assert len(minimum_energy_result.x) == 1
        return (potential_energy(minimum_energy_result.x, surface_potential)
                - tuned_potential_depth.value_in_unit(unit.kilojoule_per_mole))

    result = root_scalar(
        deviation_potential_energy,
        bracket=[-10.0 * abs(other_surface_potential.value_in_unit(unit.milli * unit.volt)), 0.0],
        method="brentq")
    if not result.converged:
        raise RuntimeError(result.flag)

    # Choose the opposite sign of the surface potential.
    tuned_surface_potential = -math.copysign(
        result.root, other_surface_potential.value_in_unit(unit.milli * unit.volt)) * (unit.milli * unit.volt)

    if plot_filename is not None:
        plt.figure()
        surface_separations = np.linspace(0.0, 30.0, 1000)
        potential_energies = np.zeros(len(surface_separations))
        for index, surface_sep in enumerate(surface_separations):
            # noinspection PyTypeChecker
            potential_energies[index] = potential_energy([surface_sep],
                                                         tuned_surface_potential.value_in_unit(unit.milli * unit.volt))
        plt.plot(surface_separations, potential_energies)
        plt.xlabel("Surface separation (nm)")
        plt.ylabel("Potential energy (kJ/mol)")
        plt.axhline(tuned_potential_depth.value_in_unit(unit.kilojoule_per_mole), color="black", linestyle="--")
        plt.ylim(1.1 * tuned_potential_depth.value_in_unit(unit.kilojoule_per_mole), 0.0)
        plt.savefig(plot_filename)
        plt.close()

    return tuned_surface_potential


def main():
    parser = argparse.ArgumentParser(description="Tune the surface potential of a colloid with a given radius so that "
                                                 "the potential depth of the combined steric and electrostatic "
                                                 "potentials is equal to a given potential depth.")
    parser.add_argument("simulation_parameters", help="YAML file with simulation parameters", type=str)
    parser.add_argument("tune_parameters", help="YAML file with tune parameters", type=str)
    parser.add_argument("--example", help="write an example analysis YAML file and exit",
                        action=ExampleAction)
    args = parser.parse_args()

    if not args.simulation_parameters.endswith(".yaml"):
        raise ValueError("The YAML file for the simulation parameters must have the .yaml extension.")
    if not args.tune_parameters.endswith(".yaml"):
        raise ValueError("The YAML file for the tune parameters must have the .yaml extension.")

    parameters = RunParameters.from_yaml(args.simulation_parameters)
    tune_parameters = TuneParameters.from_yaml(args.tune_parameters)
    if tune_parameters.tuned_type not in parameters.masses:
        raise ValueError(f"The type of the tuned colloid {tune_parameters.tuned_type} is not present in the "
                         "masses dictionary in the simulation parameters.")
    if tune_parameters.tuned_type not in parameters.radii:
        raise ValueError(f"The type of the tuned colloid {tune_parameters.tuned_type} is not present in the "
                         "radii dictionary in the simulation parameters.")
    if tune_parameters.tuned_type not in parameters.surface_potentials:
        raise ValueError(f"The type of the tuned colloid {tune_parameters.tuned_type} is not present in the "
                         "surface_potentials dictionary in the simulation parameters.")
    if tune_parameters.other_type not in parameters.masses:
        raise ValueError(f"The type of the other colloid {tune_parameters.other_type} is not present in the "
                         "masses dictionary in the simulation parameters.")
    if tune_parameters.other_type not in parameters.radii:
        raise ValueError(f"The type of the other colloid {tune_parameters.other_type} is not present in the "
                         "radii dictionary in the simulation parameters.")
    if tune_parameters.other_type not in parameters.surface_potentials:
        raise ValueError(f"The type of the other colloid {tune_parameters.other_type} is not present in the "
                         "surface_potentials dictionary in the simulation parameters.")

    potentials_parameters = ColloidPotentialsParameters(
        brush_density=parameters.brush_density, brush_length=parameters.brush_length,
        debye_length=parameters.debye_length, temperature=parameters.potential_temperature,
        dielectric_constant=parameters.dielectric_constant
    )
    colloid_potentials = ColloidPotentialsAlgebraic(
        colloid_potentials_parameters=potentials_parameters, use_log=parameters.use_log,
        cutoff_factor=parameters.cutoff_factor, periodic_boundary_conditions=False)

    tuned_surface_potential = tune_surface_potential(colloid_potentials,
                                                     parameters.radii[tune_parameters.other_type],
                                                     parameters.surface_potentials[tune_parameters.other_type],
                                                     parameters.radii[tune_parameters.tuned_type],
                                                     tune_parameters.tuned_potential_depth,
                                                     tune_parameters.plot_filename)

    print(f"The tuned surface potential is {tuned_surface_potential}.")


if __name__ == '__main__':
    main()
