import time
import numpy as np
import openmm
from openmm import app
from openmm import unit
from colloids import (ColloidPotentialsParameters, ColloidPotentialsAbstract, ColloidPotentialsAlgebraic,
                      ColloidPotentialsTabulated)


def load_initxyz(filename):
    with open(filename, "r") as f:
        line = next(f)
        ls = line.split()
        number_atoms = int(ls[0])
        _ = next(f)
        types = np.zeros(number_atoms, dtype=str)
        pos = np.zeros((number_atoms, 3), dtype=unit.Quantity)
        for index_atom in range(number_atoms):
            line = next(f)
            ls = line.split()
            types[index_atom] = ls[0]
            pos[index_atom, 0] = float(ls[1]) * unit.nanometer
            pos[index_atom, 1] = float(ls[2]) * unit.nanometer
            pos[index_atom, 2] = float(ls[3]) * unit.nanometer
    return types, pos


def main(platform_name: str = "Reference", potentials: str = "algebraic", use_log: bool = False,
         number_steps: int = 100) -> None:
    radius_positive = 105.0 * unit.nanometer
    radius_negative = 95.0 * unit.nanometer
    surface_potential_positive = 44.0 * (unit.milli * unit.volt)
    surface_potential_negative = -54.0 * (unit.milli * unit.volt)
    parameters = ColloidPotentialsParameters(brush_density=0.09 / (unit.nanometer ** 2),
                                             brush_length=10.6 * unit.nanometer,
                                             debye_length=5.726968 * unit.nanometer,
                                             temperature=298.0 * unit.kelvin,
                                             dielectric_constant=80.0)
    # noinspection PyUnresolvedReferences
    collision_rate = 0.01 / unit.picosecond
    # noinspection PyUnresolvedReferences
    timestep = 0.05 * unit.picosecond
    mass_positive = 1.0 * unit.amu
    mass_negative = (radius_negative / radius_positive) ** 3 * mass_positive
    side_length = 12328.05 * unit.nanometer

    types, positions = load_initxyz("first_frame.xyz")

    topology = app.topology.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("res1", chain)
    try:
        app.element.Element(0, "P1", "P1", mass_positive)
        app.element.Element(1, "N1", "N1", mass_negative)
    except ValueError:
        # ValueError is raised if the element already exists.
        pass
    for type, position in zip(types, positions):
        if type == "P":
            topology.addAtom("Cs", app.element.Element.getBySymbol("P1"), residue)
        else:
            assert type == "N"
            topology.addAtom("Cl", app.element.Element.getBySymbol("N1"), residue)
    topology.setPeriodicBoxVectors(np.array([[side_length.value_in_unit(unit.nanometer), 0.0, 0.0],
                                             [0.0, side_length.value_in_unit(unit.nanometer), 0.0],
                                             [0.0, 0.0, side_length.value_in_unit(unit.nanometer)]]))

    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(openmm.Vec3(side_length.value_in_unit(unit.nanometer), 0.0, 0.0),
                                        openmm.Vec3(0.0, side_length.value_in_unit(unit.nanometer), 0.0),
                                        openmm.Vec3(0.0, 0.0, side_length.value_in_unit(unit.nanometer)))
    platform = openmm.Platform.getPlatformByName(platform_name)
    integrator = openmm.LangevinIntegrator(parameters.temperature, collision_rate, timestep)

    if potentials == "algebraic":
        colloid_potentials: ColloidPotentialsAbstract = ColloidPotentialsAlgebraic(
            colloid_potentials_parameters=parameters, use_log=use_log)
    else:
        assert potentials == "tabulated"
        colloid_potentials: ColloidPotentialsAbstract = ColloidPotentialsTabulated(
            radius_one=radius_positive, radius_two=radius_negative,
            surface_potential_one=surface_potential_positive, surface_potential_two=surface_potential_negative,
            colloid_potentials_parameters=parameters, use_log=use_log)

    for type, position in zip(types, positions):
        if type == "P":
            system.addParticle(mass_positive)
            colloid_potentials.add_particle(radius=radius_positive, surface_potential=surface_potential_positive)
        else:
            assert type == "N"
            system.addParticle(mass_negative)
            colloid_potentials.add_particle(radius=radius_negative, surface_potential=surface_potential_negative)
    for force in colloid_potentials.yield_potentials():
        system.addForce(force)

    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(parameters.temperature, 1)

    start_time = time.perf_counter_ns()
    simulation.step(number_steps)
    end_time = time.perf_counter_ns()

    print(f"Time per time step: {(end_time - start_time) * 1e-9 / number_steps} s / step (platform: {platform_name}, "
          f"potentials: {potentials}, use_log: {use_log}, number_steps: {number_steps})")


if __name__ == '__main__':
    for use_log in (False, True):
        for platform, number_steps in zip(("Reference", "CPU", "OpenCL", "CUDA"), (10, 1000, 1000, 1000)):
            for potentials in ("algebraic", "tabulated"):
                try:
                    main(platform_name=platform, potentials=potentials, use_log=use_log, number_steps=number_steps)
                except openmm.OpenMMException:
                    print(f"Platform {platform} not available.")
        print()
