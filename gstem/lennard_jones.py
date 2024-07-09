import openmm
from openmm import app
from openmm import unit
from colloids.gsd_reporter import GSDReporter
from colloids.status_reporter import StatusReporter
from initial_particle_positions import subrandom_particle_positions


def main():
    number_particles = 10000
    reduced_density = 0.05
    mass = 39.9 * unit.amu  # argon
    sigma = 3.4 * unit.angstrom  # argon
    epsilon = 0.238 * unit.kilocalories_per_mole  # argon
    cutoff = 3.0 * sigma
    switch_width = 3.4 * unit.angstrom  # argon
    temperature = epsilon / unit.BOLTZMANN_CONSTANT_kB / unit.AVOGADRO_CONSTANT_NA  # kT/epsilon = 1
    reduced_time_step = 0.005
    number_equilibration_steps = 100000
    number_production_steps = 1000000
    platform = "OpenCL"  # "Reference", "CPU", "CUDA", or "OpenCL"
    trajectory_filename = "trajectory.gsd"
    trajectory_interval = 1000
    state_data_filename = "state_data.csv"
    state_data_interval = 100

    # Create topology with Argon atoms.
    topology = app.Topology()
    element = app.Element.getBySymbol('Ar')
    chain = topology.addChain()
    residue = topology.addResidue('Ar', chain)
    for _ in range(number_particles):
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
    lennard_jones_force = openmm.NonbondedForce()
    lennard_jones_force.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    lennard_jones_force.setCutoffDistance(cutoff)
    lennard_jones_force.setUseSwitchingFunction(True)
    lennard_jones_force.setUseDispersionCorrection(True)
    lennard_jones_force.setSwitchingDistance(cutoff - switch_width)

    # Add particles to the system and the Lennard-Jones force.
    for particle_index in range(number_particles):
        system.addParticle(mass)
        # Set charge to zero to switch off electrostatics.
        lennard_jones_force.addParticle(0.0, sigma, epsilon)

    # Add the Lennard-Jones force to the system.
    system.addForce(lennard_jones_force)

    # Create integrator.
    timestep = reduced_time_step * (mass * sigma ** 2 / epsilon).sqrt()
    integrator = openmm.LangevinIntegrator(temperature, 1.0 / unit.picosecond, timestep)

    # Create simulation.
    platform_object = openmm.Platform.getPlatformByName(platform)
    simulation = app.Simulation(topology, system, integrator, platform_object)

    # Set initial positions and velocities, and minimize energy of initial configuration.
    positions = subrandom_particle_positions(number_particles, system.getDefaultPeriodicBoxVectors())
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temperature)

    # Equilibration run.
    simulation.reporters.append(StatusReporter(max(1, number_equilibration_steps // 100), number_equilibration_steps))
    print("Equilibrating...")
    simulation.step(number_equilibration_steps)

    simulation.reporters = []
    simulation.reporters.append(StatusReporter(max(1, number_production_steps // 100), number_production_steps))
    simulation.reporters.append(GSDReporter(trajectory_filename, trajectory_interval,
                                            {"Ar" : sigma}, {"Ar": 0.0 * unit.volt},
                                            simulation))
    simulation.reporters.append(app.StateDataReporter(state_data_filename, state_data_interval, step=True,
                                                      time=True, kineticEnergy=True, potentialEnergy=True,
                                                      temperature=True, speed=True))
    print("Production...")
    simulation.step(number_production_steps)


if __name__ == '__main__':
    main()

