import argparse
import time
import subprocess
import sys
import numpy as np
import openmm
from openmm import app
from openmm import unit
from colloids import (ColloidPotentialsParameters, ColloidPotentialsAbstract, ColloidPotentialsAlgebraic,
                      ColloidPotentialsTabulated)
from colloids.helper_functions import read_xyz_file


def benchmark_parameters():
    # noinspection PyUnresolvedReferences
    parameters = {
        "radius_positive": 105.0 * unit.nanometer,
        "radius_negative": 95.0 * unit.nanometer,
        "surface_potential_positive": 44.0 * (unit.milli * unit.volt),
        "surface_potential_negative": -54.0 * (unit.milli * unit.volt),
        "colloid_potentials_parameters": ColloidPotentialsParameters(brush_density=0.09 / (unit.nanometer ** 2),
                                                                     brush_length=10.6 * unit.nanometer,
                                                                     debye_length=5.726968 * unit.nanometer,
                                                                     temperature=298.0 * unit.kelvin,
                                                                     dielectric_constant=80.0),
        "collision_rate": 0.01 / (unit.pico * unit.second),
        "timestep": 0.05 * (unit.pico * unit.second),
        "mass_positive": 1.0 * unit.amu,
        "side_length": 12328.05 * unit.nanometer,
    }
    parameters["mass_negative"] = (
            (parameters["radius_negative"] / parameters["radius_positive"]) ** 3 * parameters["mass_positive"])
    return parameters


def benchmark_openmm(platform_name: str = "Reference", potentials: str = "algebraic", use_log: bool = False,
                     number_steps: int = 100) -> None:
    parameters = benchmark_parameters()

    types, positions, cell = read_xyz_file("../colloids/tests/first_frame.xyz")

    topology = app.topology.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("res1", chain)
    for t, position in zip(types, positions):
        topology.addAtom(t, None, residue)
    topology.setPeriodicBoxVectors(cell)

    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(openmm.Vec3(*cell[0]), openmm.Vec3(*cell[1]), openmm.Vec3(*cell[2]))
    # Prevent printing the traceback when the platform is not existing.
    try:
        platform = openmm.Platform.getPlatformByName(platform_name)
    except openmm.OpenMMException as err:
        print(err, file=sys.stderr)
        exit(1)
    integrator = openmm.LangevinIntegrator(parameters["colloid_potentials_parameters"].temperature,
                                           parameters["collision_rate"],
                                           parameters["timestep"])

    if potentials == "algebraic":
        colloid_potentials: ColloidPotentialsAbstract = ColloidPotentialsAlgebraic(
            colloid_potentials_parameters=parameters["colloid_potentials_parameters"], use_log=use_log)
    else:
        assert potentials == "tabulated"
        colloid_potentials: ColloidPotentialsAbstract = ColloidPotentialsTabulated(
            radius_one=parameters["radius_positive"], radius_two=parameters["radius_negative"],
            surface_potential_one=parameters["surface_potential_positive"],
            surface_potential_two=parameters["surface_potential_negative"],
            colloid_potentials_parameters=parameters["colloid_potentials_parameters"], use_log=use_log)

    for t, position in zip(types, positions):
        if t == "P":
            system.addParticle(parameters["mass_positive"])
            colloid_potentials.add_particle(radius=parameters["radius_positive"],
                                            surface_potential=parameters["surface_potential_positive"])
        else:
            assert t == "N"
            system.addParticle(parameters["mass_negative"])
            colloid_potentials.add_particle(radius=parameters["radius_negative"],
                                            surface_potential=parameters["surface_potential_negative"])
    for force in colloid_potentials.yield_potentials():
        system.addForce(force)

    if platform_name == "CUDA":
        simulation = app.Simulation(topology, system, integrator, platform, 
                                    platformProperties={"Precision": "mixed"})
    else:
        simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(parameters["colloid_potentials_parameters"].temperature, 1)

    start_time = time.perf_counter_ns()
    simulation.step(number_steps)
    end_time = time.perf_counter_ns()

    print(f"Time per time step: {(end_time - start_time) * 1e-9 / number_steps} s / step (platform: {platform_name}, "
          f"potentials: {potentials}, use_log: {use_log}, number_steps: {number_steps})")


# See https://stackoverflow.com/questions/60979532/argparse-ignore-positional-arguments-if-a-flag-is-set
# Run each benchmark in a new subprocess so that openmm is properly reset between runs.
class BenchmarkAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        for use_log in ("false", "true"):
            for platform, number_steps in zip(("Reference", "CPU", "OpenCL", "CUDA"), (10, 100, 1000, 10000)):
                for potentials in ("algebraic", "tabulated"):
                    try:
                        subprocess.run(f"python {__file__} {platform} {potentials} {use_log} {number_steps}",
                                       shell=True, check=True, stderr=subprocess.PIPE, text=True)
                    except subprocess.CalledProcessError as err:
                        print(err.stderr.strip())
            print()
        parser.exit()


def main():
    parser = argparse.ArgumentParser(description="Benchmark OpenMM for a colloids system.")
    parser.add_argument("--all", action=BenchmarkAction,
                        help="run benchmark with various parameters")
    parser.add_argument("platform", help="OpenMM platform to use", type=str,
                        choices=("Reference", "CPU", "OpenCL", "CUDA"))
    parser.add_argument("potentials", help="colloid potentials to use", type=str,
                        choices=("algebraic", "tabulated"))
    parser.add_argument("use_log", help="use logarithmic potentials", type=str, choices=("false", "true"))
    parser.add_argument("number_steps", help="number of time steps to run", type=int)
    args = parser.parse_args()
    assert args.number_steps > 0
    benchmark_openmm(platform_name=args.platform, potentials=args.potentials,
                     use_log=(args.use_log == "true"), number_steps=args.number_steps)


if __name__ == '__main__':
    main()
