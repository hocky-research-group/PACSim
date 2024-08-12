import argparse
import pathlib
from colloids.run_parameters import RunParameters
from colloids.colloids_analyze.analysis_parameters import AnalysisParameters
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

    run_parameters = [{
        "path": pathlib.Path(simulation_parameters).parent,
        "parameters": RunParameters.from_yaml(simulation_parameters)}
        for simulation_parameters in args.simulation_parameters]
    analysis_parameters = AnalysisParameters.from_yaml(args.analysis_parameters)

    if analysis_parameters.plot_state_data:
        plotter = StateDataPlotter(analysis_parameters.working_directory,
                                   run_parameters, analysis_parameters.state_data_labels)
        plotter.plot()


if __name__ == '__main__':
    main()
