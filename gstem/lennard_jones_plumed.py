import gsd.hoomd
import numpy as np
import openmm
from openmm import app
from openmm import unit
from openmmplumed import PlumedForce
from colloids.gsd_reporter import GSDReporter
from colloids.shifted_lennard_jones_walls import ShiftedLennardJonesWalls
from colloids.status_reporter import StatusReporter
from initial_particle_positions import subrandom_particle_positions, lattice_particle_positions


def main():
    number_particles = 500
    reduced_density = 0.05
    mass = 39.9 * unit.amu / 50.0  # argon
    sigma = 3.4 * unit.angstrom  # argon
    epsilon = 0.238 * unit.kilocalories_per_mole  # argon
    cutoff = 4.0 * sigma
    switch_width = 3.4 * unit.angstrom  # argon
    temperature = 1.0 / 3.0 * epsilon / unit.BOLTZMANN_CONSTANT_kB / unit.AVOGADRO_CONSTANT_NA  # kT/epsilon = 1
    reduced_time_step = 0.001
    number_equilibration_steps = 0
    number_production_steps = 1000000
    platform = "CPU"  # "Reference", "CPU", "CUDA", or "OpenCL"
    trajectory_filename = "trajectory.gsd"
    trajectory_interval = 1000
    state_data_filename = "state_data.csv"
    state_data_interval = 100
    initial = "lattice"  # "lattice" or "random"
    use_plumed = True
    # Only relevant if use_plumed is True.
    distance_threshold_first_coordination_sphere = 1.5 * sigma.value_in_unit(unit.nano * unit.meter)
    lq6_threshold = 0.5
    contact_distance_threshold = 1.5 * sigma.value_in_unit(unit.nano * unit.meter)
    switch_width_plumed = 0.01  # Only relevant if use_plumed is True.

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
    box_vectors = np.array([
        [box_edge, 0.0 * unit.angstrom, 0.0 * unit.angstrom],
        [0.0 * unit.angstrom, box_edge, 0.0 * unit.angstrom],
        [0.0 * unit.angstrom, 0.0 * unit.angstrom, box_edge]])

    # On my MAC, PLUMED throws a segmentation fault if one switches off periodic boundaries by not setting these
    # box vectors and by choosing CutoffNonPeriodic for the Lennard-Jones force. Therefore, we simply choose a big
    # enough periodic box here so that periodic boundary conditions should not matter with the walls.
    enlarged_box_edge = box_edge + cutoff / 2.0
    system.setDefaultPeriodicBoxVectors(openmm.Vec3(enlarged_box_edge, 0.0 * unit.angstrom, 0.0 * unit.angstrom),
                                        openmm.Vec3(0.0 * unit.angstrom, enlarged_box_edge, 0.0 * unit.angstrom),
                                        openmm.Vec3(0.0 * unit.angstrom, 0.0 * unit.angstrom, enlarged_box_edge))

    # Create Lennard-Jones force with periodic boundary conditions.
    lennard_jones_force = openmm.CustomNonbondedForce(
        "4.0 * epsilon_lj / (alpha_lj * alpha_lj) "
        "* (1.0 / (r * r / (sigma * sigma) - 1.0 )^6 - alpha_lj / (r * r / (sigma * sigma) - 1.0)^3 )")
    lennard_jones_force.addGlobalParameter("sigma", sigma)
    lennard_jones_force.addGlobalParameter("epsilon_lj", epsilon)
    lennard_jones_force.addGlobalParameter("alpha_lj", 50)
    lennard_jones_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    lennard_jones_force.setCutoffDistance(cutoff)
    lennard_jones_force.setUseSwitchingFunction(True)
    lennard_jones_force.setSwitchingDistance(cutoff - switch_width)

    # Add particles to the system and the Lennard-Jones force.
    for particle_index in range(number_particles):
        system.addParticle(mass)
        lennard_jones_force.addParticle()

    # Add the Lennard-Jones force to the system.
    system.addForce(lennard_jones_force)

    # Add walls to the system.
    # The shifted Lennard-Jones walls have a divergence at a distance of sigma / 2.0 - 1.0 nm to the walls.
    # Here, sigma is the parameter of the generalized LJ potential, that is, the hard-core diameter of the particles.
    # Since sigma / 2.0 is smaller than 1 nm, this means that the divergence is actually outside the expected walls.
    # We choose the wall_distances so that the shifted Lennard-Jones walls diverge at a distance of sigma / 2.0 - 0.2 nm
    # to the walls which looks reasonable in the movies.
    walls = ShiftedLennardJonesWalls(wall_distances=[box_edge - 2.0 * (unit.nano * unit.meter) * (1.0 - 0.2),
                                                     box_edge - 2.0 * (unit.nano * unit.meter) * (1.0 - 0.2),
                                                     box_edge - 2.0 * (unit.nano * unit.meter) * (1.0 - 0.2)],
                                     epsilon=temperature * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA,
                                     alpha=1.0, wall_directions=[True, True, True])
    for particle_index in range(number_particles):
        walls.add_particle(particle_index, radius=sigma / 2.0)
    for potential in walls.yield_potentials():
        system.addForce(potential)

    # See https://www.plumed-nest.org/eggs/19/049/data/plumed_GeTe.dat.html
    script = f"""
    # Calculate the Steinhardt Q6 vector for each of the atoms in the system
    q6: Q6 ...
        SPECIES=1-{number_particles} 
        SWITCH={{GAUSSIAN D_0={distance_threshold_first_coordination_sphere} R_0={switch_width_plumed} D_MAX={distance_threshold_first_coordination_sphere + switch_width_plumed}}} 
        NOPBC
    ...
    # Calculate the local Steinhardt parameter for each of the atoms in the system
    lq6: LOCAL_Q6 ...
        SPECIES=q6
        SWITCH={{GAUSSIAN D_0={distance_threshold_first_coordination_sphere} R_0={switch_width_plumed} D_MAX={distance_threshold_first_coordination_sphere + switch_width_plumed}}} 
        MEAN 
        HISTOGRAM={{GAUSSIAN LOWER=-1.0 UPPER=1.0 NBINS=40 SMEAR=0.1}}
        LOWMEM
        NOPBC
    ...
    PRINT ARG=lq6.* FILE=lq6 STRIDE={state_data_interval}
    DUMPMULTICOLVAR DATA=lq6 FILE=LQ6MULTICOLVAR.xyz STRIDE={trajectory_interval}
    # Now select only those atoms that have a local q6 parameter that is larger than a certain threshold
    flq6: MFILTER_MORE ...
        DATA=lq6 
        SWITCH={{GAUSSIAN D_0={lq6_threshold} R_0={switch_width_plumed} D_MAX={lq6_threshold + switch_width_plumed}}}
        LOWMEM
        NOPBC
    ...
    # Calculate the contact matrix for those atoms that have a local q6 parameter that is larger than a threshold
    cc_cmat: CONTACT_MATRIX ...
        ATOMS=flq6 
        SWITCH={{GAUSSIAN D_0={contact_distance_threshold} R_0={switch_width_plumed} D_MAX={contact_distance_threshold + switch_width_plumed}}}
        NOPBC
    ...
    # Use depth first clustering to identify the sizes of the clusters
    dfs: DFSCLUSTERING MATRIX=cc_cmat LOWMEM NOPBC
    # Compute the sum of the coordination numbers for the atoms in the largest cluster                                                         
    clust1: CLUSTER_PROPERTIES CLUSTERS=dfs CLUSTER=1 SUM LOWMEM NOPBC
    PRINT ARG=clust1.sum FILE=clust1 STRIDE={state_data_interval}
    # Do the same but without the filter on lq6.
    cc_cmat_all: CONTACT_MATRIX ...
        ATOMS=1-{number_particles} 
        SWITCH={{GAUSSIAN D_0={contact_distance_threshold} R_0={switch_width_plumed} D_MAX={contact_distance_threshold + switch_width_plumed}}}
        NOPBC
    ...
    dfs_all: DFSCLUSTERING MATRIX=cc_cmat_all LOWMEM NOPBC
    clust1_all: CLUSTER_PROPERTIES CLUSTERS=dfs_all CLUSTER=1 SUM LOWMEM NOPBC
    PRINT ARG=clust1_all.sum FILE=clust1_all STRIDE={state_data_interval}
    """
    with open("plumed.dat", "w") as file:
        print(script, file=file)
    if use_plumed:
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
        positions = subrandom_particle_positions(number_particles, box_vectors)
    else:
        positions = lattice_particle_positions(number_particles, box_vectors)

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
        simulation.reporters.append(StatusReporter(max(1, number_equilibration_steps // 100),
                                                   number_equilibration_steps))
        print("Equilibrating...")
        simulation.step(number_equilibration_steps)
        simulation.reporters = []

    simulation.reporters.append(StatusReporter(max(1, number_production_steps // 100), number_production_steps))
    simulation.reporters.append(GSDReporter(trajectory_filename, trajectory_interval,
                                            {"Ar": sigma / 2.0}, {"Ar": 0.0 * unit.volt},
                                            simulation, cell=box_vectors))
    simulation.reporters.append(app.StateDataReporter(state_data_filename, state_data_interval, step=True,
                                                      time=True, kineticEnergy=True, potentialEnergy=True,
                                                      temperature=True, speed=True))
    print("Production...")
    simulation.step(number_production_steps)

    if use_plumed:
        with (gsd.hoomd.open("trajectory.gsd", "r") as file_read,
              gsd.hoomd.open("trajectory_lq6.gsd", "w") as file_write):
            for i, frame in enumerate(file_read):
                if i == 0:
                    frame.particles.charge = np.zeros(frame.particles.N)
                else:
                    lq6 = np.loadtxt("LQ6MULTICOLVAR.xyz", skiprows=i * 2 + (i - 1) * number_particles,
                                     max_rows=number_particles, usecols=(4,))
                    assert len(lq6) > 0
                    frame.particles.charge = lq6
                file_write.append(frame)
                file_write.flush()


if __name__ == '__main__':
    main()
