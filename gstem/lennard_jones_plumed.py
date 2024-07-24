import numpy as np
import openmm
from openmm import app
from openmm import unit
from openmmplumed import PlumedForce
from colloids.gsd_reporter import GSDReporter
from colloids.status_reporter import StatusReporter
from initial_particle_positions import subrandom_particle_positions, lattice_particle_positions


def main():
    number_particles = 500
    reduced_density = 0.5
    mass = 39.9 * unit.amu  # argon
    sigma = 3.4 * unit.angstrom  # argon
    epsilon = 0.238 * unit.kilocalories_per_mole  # argon
    cutoff = 4.0 * sigma
    switch_width = 3.4 * unit.angstrom  # argon
    temperature = epsilon / unit.BOLTZMANN_CONSTANT_kB / unit.AVOGADRO_CONSTANT_NA  # kT/epsilon = 1
    reduced_time_step = 0.001
    number_equilibration_steps = 0
    number_production_steps = 10
    platform = "CPU"  # "Reference", "CPU", "CUDA", or "OpenCL"
    trajectory_filename = "trajectory.gsd"
    trajectory_interval = 1000
    state_data_filename = "state_data.csv"
    state_data_interval = 100
    initial = "lattice"  # "lattice" or "random"
    use_plumed = True

    # Create topology with Argon atoms.
    topology = app.Topology()
    element = app.Element.getBySymbol('Ar')
    chain = topology.addChain()
    for _ in range(number_particles):
        residue = topology.addResidue('Ar', chain)
        topology.addAtom('Ar', element, residue)

    # Create OpenMM system.
    system = openmm.System()
    number_density = reduced_density / sigma ** 3
    volume = number_particles * (number_density ** -1)
    box_edge = volume ** (1.0 / 3.0)
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_edge, 0.0 * unit.angstrom, 0.0 * unit.angstrom),
        openmm.Vec3(0.0 * unit.angstrom, box_edge, 0.0 * unit.angstrom),
        openmm.Vec3(0.0 * unit.angstrom, 0.0 * unit.angstrom, box_edge)
    )

    # Create Lennard-Jones force with periodic boundary conditions.
    lennard_jones_force = openmm.CustomNonbondedForce(
        "4.0 * epsilon / (alpha * alpha) "
        "* (1.0 / (r * r / (sigma * sigma) - 1.0 )^6 - alpha / (r * r / (sigma * sigma) - 1.0)^3 )")
    lennard_jones_force.addGlobalParameter("sigma", sigma)
    lennard_jones_force.addGlobalParameter("epsilon", epsilon)
    lennard_jones_force.addGlobalParameter("alpha", 50)
    lennard_jones_force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    lennard_jones_force.setCutoffDistance(cutoff)
    lennard_jones_force.setUseSwitchingFunction(True)
    lennard_jones_force.setSwitchingDistance(cutoff - switch_width)

    # Add particles to the system and the Lennard-Jones force.
    for particle_index in range(number_particles):
        system.addParticle(mass)
        # Set charge to zero to switch off electrostatics.
        lennard_jones_force.addParticle()

    # Add the Lennard-Jones force to the system.
    system.addForce(lennard_jones_force)

    if use_plumed:
        # See https://www.plumed-nest.org/eggs/19/049/data/plumed_GeTe.dat.html
        distance_threshold_first_coordination_sphere = 0.4  # TODO: SET ACCORDING TO SIMULATIONS!
        switch_width = 0.01
        script = f"""
        # Calculate the Steinhardt Q6 vector for each of the atoms in the system
        q6: Q6 ...
            SPECIES=1-{number_particles} 
            SWITCH={{GAUSSIAN D_0={distance_threshold_first_coordination_sphere} R_0={switch_width} D_MAX={distance_threshold_first_coordination_sphere + switch_width}}} 
            MEAN 
            HISTOGRAM={{GAUSSIAN LOWER=0.0 UPPER=1.0 NBINS=20 SMEAR=0.1}}
        ...
        PRINT ARG=q6.* FILE=q6
        # Calculate the local Steinhardt parameter for each of the atoms in the system 
        # in the manner described by ten Wolde and Frenkel.
        lq6: LOCAL_Q6 ...
            SPECIES=q6 
            SWITCH={{GAUSSIAN D_0={distance_threshold_first_coordination_sphere} R_0={switch_width} D_MAX={distance_threshold_first_coordination_sphere + switch_width}}} 
            MEAN 
            HISTOGRAM={{GAUSSIAN LOWER=0.0 UPPER=1.0 NBINS=20 SMEAR=0.1}} 
        ...
        PRINT ARG=lq6.* FILE=lq6
        """
        system.addForce(PlumedForce(script))

    # Create integrator.
    timestep = reduced_time_step * (mass * sigma ** 2 / epsilon).sqrt()
    print(f"Time step: {timestep}")
    integrator = openmm.LangevinIntegrator(temperature, 1.0 / unit.picosecond, timestep)

    # Create simulation.
    platform_object = openmm.Platform.getPlatformByName(platform)
    simulation = app.Simulation(topology, system, integrator, platform_object)

    # Set initial positions and velocities, and minimize energy of initial configuration.
    if initial == "random":
        positions = subrandom_particle_positions(number_particles, system.getDefaultPeriodicBoxVectors())
    else:
        positions = lattice_particle_positions(number_particles, system.getDefaultPeriodicBoxVectors())

    min_distance = float("inf")
    for index_one in range(len(positions)):
        for index_two in range(index_one + 1, len(positions)):
            if index_one != index_two:
                position_one = positions[index_one]
                position_two = positions[index_two]
                distance = np.linalg.norm(position_one - position_two)
                min_distance = min(min_distance, distance)
    assert min_distance > 0.0
    if min_distance <= sigma.value_in_unit(unit.nano * unit.meter):
        raise RuntimeError("Particles are too close together.")

    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature)

    if number_equilibration_steps > 0:
        # Equilibration run.
        simulation.reporters.append(StatusReporter(max(1, number_equilibration_steps // 100), number_equilibration_steps))
        print("Equilibrating...")
        simulation.step(number_equilibration_steps)
        simulation.reporters = []

    simulation.reporters.append(StatusReporter(max(1, number_production_steps // 100), number_production_steps))
    simulation.reporters.append(GSDReporter(trajectory_filename, trajectory_interval,
                                            {"Ar": sigma}, {"Ar": 0.0 * unit.volt},
                                            simulation))
    simulation.reporters.append(app.StateDataReporter(state_data_filename, state_data_interval, step=True,
                                                      time=True, kineticEnergy=True, potentialEnergy=True,
                                                      temperature=True, speed=True))
    print("Production...")
    simulation.step(number_production_steps)


if __name__ == '__main__':
    main()

