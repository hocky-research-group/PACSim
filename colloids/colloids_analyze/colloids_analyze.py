import argparse
import inspect
import pathlib
from colloids.run_parameters import RunParameters
from colloids.colloids_analyze import LabeledRunParametersWithPath
from colloids.colloids_analyze.analysis_parameters import AnalysisParameters
from colloids.colloids_analyze.coordination_numbers_plotter import CoordinationNumbersPlotter
from colloids.colloids_analyze.rdf_plotter import RDFPlotter
from colloids.colloids_analyze.sdf_plotter import SDFPlotter
from colloids.colloids_analyze.snowman_orientation_distribution_plotter import SnowmanOrientationDistributionPlotter
from colloids.colloids_analyze.state_data_plotter import StateDataPlotter


class ExampleAction(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        default_parameters = AnalysisParameters()
        default_parameters.to_yaml("example_analysis.yaml")
        parser.exit()


def main():
    parser = argparse.ArgumentParser(description="Create an initial configuration for an OpenMM simulation of a "
                                                 "colloids system.")
    parser.add_argument("simulation_parameters", help="YAML file with simulation parameters", type=str,
                        nargs="+")
    parser.add_argument("analysis_parameters", help="YAML file with analysis parameters",
                        type=str)
    parser.add_argument("--example", help="write an example analysis YAML file and exit",
                        action=ExampleAction)
    args = parser.parse_args()

    for simulation_parameters in args.simulation_parameters:
        if not simulation_parameters.endswith(".yaml"):
            raise ValueError("The YAML file for the simulation parameters must have the .yaml extension.")
    if not args.analysis_parameters.endswith(".yaml"):
        raise ValueError("The YAML file for the analysis parameters must have the .yaml extension.")

    analysis_parameters = AnalysisParameters.from_yaml(args.analysis_parameters)

    if analysis_parameters.labels is not None:
        if not len(analysis_parameters.labels) == len(args.simulation_parameters):
            raise ValueError("The number of labels must match the number of simulation parameters files.")

    labels = (analysis_parameters.labels if analysis_parameters.labels is not None
              else [sp for sp in args.simulation_parameters])

    run_parameters = [LabeledRunParametersWithPath(path=pathlib.Path(simulation_parameters).parent, label=label,
                                                   run_parameters=RunParameters.from_yaml(simulation_parameters))
                      for label, simulation_parameters in zip(labels, args.simulation_parameters)]

    if analysis_parameters.plot_state_data:
        plotter = StateDataPlotter(analysis_parameters.working_directory, run_parameters)
        plotter.plot()

    if analysis_parameters.plot_rdf:
        if analysis_parameters.rdf_parameters is not None:
            try:
                plotter = RDFPlotter(analysis_parameters.working_directory, run_parameters,
                                     **analysis_parameters.rdf_parameters)
            except TypeError:
                raise TypeError(
                    f"RDFPlotter does not accept the given arguments {analysis_parameters.rdf_parameters}. "
                    f"The expected signature is {inspect.signature(RDFPlotter)} (the working_directory and "
                    f"run_parameters arguments need not be specified).")
        else:
            plotter = RDFPlotter(analysis_parameters.working_directory, run_parameters)
        plotter.plot()
    else:
        if analysis_parameters.rdf_parameters is not None:
            raise ValueError("The RDF plotter parameters are only valid if the RDF plot is to be plotted.")

    if analysis_parameters.plot_sdf:
        if analysis_parameters.sdf_parameters is not None:
            try:
                plotter = SDFPlotter(analysis_parameters.working_directory, run_parameters,
                                     **analysis_parameters.sdf_parameters)
            except TypeError:
                raise TypeError(
                    f"SDFPlotter does not accept the given arguments {analysis_parameters.sdf_parameters}. "
                    f"The expected signature is {inspect.signature(SDFPlotter)} (the working_directory and "
                    f"run_parameters arguments need not be specified).")
        else:
            plotter = SDFPlotter(analysis_parameters.working_directory, run_parameters)
        plotter.plot()

    if analysis_parameters.plot_coordination_numbers:
        if analysis_parameters.coordination_numbers_parameters is not None:
            try:
                plotter = CoordinationNumbersPlotter(analysis_parameters.working_directory, run_parameters,
                                                     **analysis_parameters.coordination_numbers_parameters)
            except TypeError:
                raise TypeError(
                    f"CoordinationNumbersPlotter does not accept the given arguments "
                    f"{analysis_parameters.coordination_numbers_parameters}. The expected signature is "
                    f"{inspect.signature(CoordinationNumbersPlotter)} (the working_directory and run_parameters "
                    f"arguments need not be specified).")
        else:
            plotter = CoordinationNumbersPlotter(analysis_parameters.working_directory, run_parameters)
        plotter.plot()

    if analysis_parameters.plot_snowman_orientation_distribution:
        if analysis_parameters.snowman_orientation_distribution_parameters is not None:
            try:
                plotter = SnowmanOrientationDistributionPlotter(
                    analysis_parameters.working_directory, run_parameters,
                    **analysis_parameters.snowman_orientation_distribution_parameters)
            except TypeError:
                raise TypeError(
                    f"SnowmanOrientationDistributionPlotter does not accept the given arguments "
                    f"{analysis_parameters.snowman_orientation_distribution_parameters}. The expected signature is "
                    f"{inspect.signature(SnowmanOrientationDistributionPlotter)} (the working_directory and "
                    f"run_parameters arguments need not be specified).")
        else:
            plotter = SnowmanOrientationDistributionPlotter(analysis_parameters.working_directory, run_parameters)
        plotter.plot()


if __name__ == '__main__':
    main()
