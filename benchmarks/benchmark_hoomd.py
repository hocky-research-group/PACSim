import argparse
import subprocess
import sys
import time
import hoomd
import hoomd.md
from openmm import unit
from benchmark import benchmark_parameters
from colloids.colloid_potentials_tabulated_hoomd import ColloidPotentialsTabulatedHoomd


def benchmark_hoomd(device: str = "CPU", number_steps: int = 100, shift: bool = True):
    parameters = benchmark_parameters()

    radius_positive = parameters["radius_positive"].value_in_unit(unit.nano * unit.meter)
    radius_negative = parameters["radius_negative"].value_in_unit(unit.nano * unit.meter)
    surface_potential_positive = parameters["surface_potential_positive"].value_in_unit(unit.milli * unit.volt)
    surface_potential_negative = parameters["surface_potential_negative"].value_in_unit(unit.milli * unit.volt)
    k_temperature = (parameters["colloid_potentials_parameters"].temperature * unit.BOLTZMANN_CONSTANT_kB
                     * unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.kilojoule_per_mole)

    # noinspection PyUnresolvedReferences
    collision_rate = parameters["collision_rate"].value_in_unit((unit.pico * unit.second) ** -1)
    # noinspection PyUnresolvedReferences
    timestep = parameters["timestep"].value_in_unit(unit.pico * unit.second)
    mass_positive = parameters["mass_positive"].value_in_unit(unit.amu)
    mass_negative = parameters["mass_negative"].value_in_unit(unit.amu)

    # Prevent printing the traceback when the platform is not existing.
    try:
        hoomd.context.initialize("--mode=cpu --notice-level=0" if device == "CPU" else "--mode=gpu --notice-level=0")
    except RuntimeError as err:
        print(err, file=sys.stderr)
        exit(1)
    snapshot = hoomd.data.gsd_snapshot("../colloids/tests/first_frame.gsd")
    hoomd.init.read_snapshot(snapshot)
    nl = hoomd.md.nlist.cell()
    ColloidPotentialsTabulatedHoomd(
        radius_one=radius_positive, radius_two=radius_negative,
        surface_potential_one=surface_potential_positive, surface_potential_two=surface_potential_negative,
        type_one="P", type_two="N",
        colloid_potentials_parameters=parameters["colloid_potentials_parameters"], neighbor_list=nl, shift=shift)

    hoomd.md.integrate.mode_standard(dt=timestep)
    langevin = hoomd.md.integrate.langevin(group=hoomd.group.all(), kT=k_temperature, seed=1)
    langevin.set_gamma("P", mass_positive * collision_rate)
    langevin.set_gamma("N", mass_negative * collision_rate)

    start_time = time.perf_counter_ns()
    hoomd.run(number_steps)
    end_time = time.perf_counter_ns()

    print(f"Time per time step: {(end_time - start_time) * 1e-9 / number_steps} s / step (device: {device}, "
          f"number_steps: {number_steps}, shift: {shift})")


# See https://stackoverflow.com/questions/60979532/argparse-ignore-positional-arguments-if-a-flag-is-set
# Run each benchmark in a new subprocess so that openmm is properly reset between runs.
class BenchmarkAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        for device, number_steps in zip(("CPU", "GPU"), (10000, 10000)):
            for shift in ("false", "true"):
                try:
                    subprocess.run(f"python {__file__} {device} {number_steps} {shift}", shell=True,
                                   check=True, stderr=subprocess.PIPE, text=True)
                except subprocess.CalledProcessError as err:
                    print(err.stderr.strip())
        parser.exit()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Hoomd for a colloids system.")
    parser.add_argument("--all", action=BenchmarkAction,
                        help="run benchmark with various parameters")
    parser.add_argument("device", help="Hoomd device to use", type=str,
                        choices=("CPU", "GPU"))
    parser.add_argument("number_steps", help="number of time steps to run", type=int)
    parser.add_argument("shift", help="use shifted potentials", type=str, choices=("false", "true"))
    args = parser.parse_args()
    assert args.number_steps > 0
    benchmark_hoomd(device=args.device, number_steps=args.number_steps, shift=(args.shift == "true"))


if __name__ == '__main__':
    main()
