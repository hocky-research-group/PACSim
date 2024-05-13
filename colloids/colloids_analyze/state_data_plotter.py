from typing import Collection, Optional, Union
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from openmm import unit
import pandas as pd
from colloids.run_parameters import RunParameters
from colloids.colloids_analyze import Plotter

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "text.latex.preamble": r"\usepackage{siunitx}"
})


class StateDataPlotter(Plotter):
    def __init__(self, working_directory: str,
                 run_parameters: Collection[dict[str, Union[pathlib.Path, RunParameters]]],
                 labels: Optional[Collection[str]]):
        super().__init__(working_directory)
        self._temperatures = [rp["parameters"].temperature for rp in run_parameters]
        # If rp["parameters"].state_data_filename is a complete path, the division operator of paths only returns that
        # complete path. Otherwise, this combines base_path and path.
        self._state_data_paths = [
            rp["path"] / rp["parameters"].state_data_filename for rp in run_parameters]
        for state_data_path in self._state_data_paths:
            if not state_data_path.exists() and state_data_path.is_file():
                raise ValueError(f"The state data file {state_data_path} does not exist.")
            if not state_data_path.suffix == ".csv":
                raise ValueError(f"The state data file {state_data_path} does not have the .csv extension.")
        if labels is not None and len(labels) != len(self._state_data_paths):
            raise ValueError("The number of labels must match the number of state data files.")
        self._labels = labels if labels is not None else [path for path in self._state_data_paths]

    def plot(self) -> None:
        potential_energy_figure = plt.figure()
        potential_energy_axes = potential_energy_figure.subplots()
        kinetic_energy_figure = plt.figure()
        kinetic_energy_axes = kinetic_energy_figure.subplots()
        temperature_figure = plt.figure()
        temperature_axes = temperature_figure.subplots()
        for index, (temperature, state_data_path, label) in enumerate(
                zip(self._temperatures, self._state_data_paths, self._labels)):
            kt = (unit.BOLTZMANN_CONSTANT_kB * temperature * unit.AVOGADRO_CONSTANT_NA).value_in_unit(
                unit.kilojoules_per_mole)
            state_data = pd.read_csv(state_data_path)
            # Remove quotes and hash symbols from the column names.
            state_data.rename(columns=lambda c: c.strip("\"\"#"), inplace=True)
            potential_energy_axes.plot(state_data["Time (ps)"] / 1000, state_data["Potential Energy (kJ/mole)"] / kt,
                                       label=label)
            kinetic_energy_axes.plot(state_data["Time (ps)"] / 1000, state_data["Kinetic Energy (kJ/mole)"] / kt,
                                     label=label)
            temperature_axes.plot(state_data["Time (ps)"] / 1000, state_data["Temperature (K)"], label=label,
                                  alpha=0.5)
            temperature_axes.axhline(temperature.value_in_unit(unit.kelvin), linestyle="dashed", color=f"C{index}")
            temperature_axes.axhline(np.mean(state_data["Temperature (K)"]), linestyle="dotted", color=f"C{index}")
        potential_energy_axes.set_xlabel(r"time $t / \unit{\nano\second}$")
        potential_energy_axes.set_ylabel(r"potential energy $U / k_\mathrm{B} T$")
        kinetic_energy_axes.set_xlabel(r"time $t / \unit{\nano\second}$")
        kinetic_energy_axes.set_ylabel(r"kinetic energy $K / k_\mathrm{B} T$")
        temperature_axes.set_xlabel(r"time $t / \unit{\nano\second}$")
        temperature_axes.set_ylabel(r"temperature $T / \unit{\kelvin}$")
        potential_energy_axes.legend()
        kinetic_energy_axes.legend()
        temperature_axes.legend()
        potential_energy_figure.savefig(self._working_directory / "potential_energy.pdf", bbox_inches="tight")
        kinetic_energy_figure.savefig(self._working_directory / "kinetic_energy.pdf", bbox_inches="tight")
        temperature_figure.savefig(self._working_directory / "temperature.pdf", bbox_inches="tight")
        potential_energy_figure.clear()
        kinetic_energy_figure.clear()
        temperature_figure.clear()
