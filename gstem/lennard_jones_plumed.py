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
    platform = "CUDA"  # "Reference", "CPU", "CUDA", or "OpenCL"
    trajectory_filename = "trajectory.gsd"
    trajectory_interval = 1000
    state_data_filename = "state_data.csv"
    state_data_interval = 100
    initial = "lattice"  # "lattice" or "random"
    use_plumed = True
    distance_threshold_first_coordination_sphere = 0.55  # Only relevant if use_plumed is True.
    lq6_threshold = 0.19
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
        script = f"""
        # Calculate the Steinhardt Q6 vector for each of the atoms in the system
        q6: Q6 ...
            SPECIES=1-{number_particles} 
            SWITCH={{GAUSSIAN D_0={distance_threshold_first_coordination_sphere} R_0={switch_width_plumed} D_MAX={distance_threshold_first_coordination_sphere + switch_width_plumed}}} 
            MEAN 
            HISTOGRAM={{GAUSSIAN LOWER=0.0 UPPER=1.0 NBINS=20 SMEAR=0.1}}
        ...
        PRINT ARG=q6.* FILE=q6 STRIDE={state_data_interval}
        # Calculate the local Steinhardt parameter for each of the atoms in the system 
        # in the manner described by ten Wolde and Frenkel.
        lq6: LOCAL_Q6 ...
            SPECIES=q6 
            SWITCH={{GAUSSIAN D_0={distance_threshold_first_coordination_sphere} R_0={switch_width_plumed} D_MAX={distance_threshold_first_coordination_sphere + switch_width_plumed}}} 
            MEAN 
            HISTOGRAM={{GAUSSIAN LOWER=0.0 UPPER=1.0 NBINS=20 SMEAR=0.1}} 
        ...
        PRINT ARG=lq6.* FILE=lq6
        # Now select only those atoms that have a local q6 parameter that is larger than a certain threshold
        flq6: MFILTER_MORE DATA=lq6 SWITCH={{GAUSSIAN D_0={lq6_threshold} R_0={switch_width_plumed} D_MAX={lq6_threshold + switch_width_plumed}}}
        # Calculate the coordination number for those atoms that have a local q6 parameter that is larger than a certain threshold
        cc_cmat: CONTACT_MATRIX ATOMS=flq6 SWITCH={{GAUSSIAN D_0={contact_distance_threshold} R_0={switch_width_plumed} D_MAX={contact_distance_threshold + switch_width_plumed}}}
        # Use depth first clustering to identify the sizes of the clusters
        dfs: DFSCLUSTERING MATRIX=cc_cmat
        # Compute the sum of the coordination numbers for the atoms in the largest cluster                                                         
        clust1: CLUSTER_PROPERTIES CLUSTERS=dfs CLUSTER=1 SUM  
        PRINT ARG=clust1.* FILE=clust1 STRIDE={state_data_interval}
        # Do the same but without the filter on lq6.
        cc_cmat_all: CONTACT_MATRIX ATOMS=q6 SWITCH={{GAUSSIAN D_0={contact_distance_threshold} R_0={switch_width_plumed} D_MAX={contact_distance_threshold + switch_width_plumed}}}
        dfs_all: DFSCLUSTERING MATRIX=cc_cmat_all
        clust1_all: CLUSTER_PROPERTIES CLUSTERS=dfs_all CLUSTER=1 SUM
        PRINT ARG=clust1_all.* FILE=clust1_all STRIDE={state_data_interval}
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
                                            {"Ar": sigma / 2.0}, {"Ar": 0.0 * unit.volt},
                                            simulation))
    simulation.reporters.append(app.StateDataReporter(state_data_filename, state_data_interval, step=True,
                                                      time=True, kineticEnergy=True, potentialEnergy=True,
                                                      temperature=True, speed=True))
    print("Production...")
    simulation.step(number_production_steps)


if __name__ == '__main__':
    main()

