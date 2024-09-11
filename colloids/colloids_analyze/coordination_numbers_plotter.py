from typing import Optional, Sequence
import matplotlib.pyplot as plt
import MDAnalysis
import MDAnalysis.analysis.rdf
import numpy as np
from colloids.colloids_analyze import LabeledRunParametersWithPath, Plotter


class CoordinationNumbersPlotter(Plotter):
    def __init__(self, working_directory: str, run_parameters: Sequence[LabeledRunParametersWithPath],
                 coordination_number_range: Sequence[int] = (0, 1500), frame_start: Optional[int] = -1,
                 frame_stop: Optional[int] = None, frame_step: Optional[int] = None, types: Sequence[str] = ("P", "N")):
        super().__init__(working_directory, run_parameters)
        if not len(coordination_number_range) == 2:
            raise ValueError("The range for the coordination number must have exactly two elements.")
        if not all(r >= 0 for r in coordination_number_range):
            raise ValueError("The range for the coordination number must be non-negative.")
        if not coordination_number_range[0] < coordination_number_range[1]:
            raise ValueError("The lower limit of the range for the coordination number must be less than the upper "
                             "limit.")
        if not len(types) == 2:
            raise ValueError("The types sequence must have exactly two elements.")
        self._coordination_number_range = list(coordination_number_range)
        self._frame_start = frame_start
        self._frame_stop = frame_stop
        self._frame_step = frame_step
        self._types = list(types)

    def plot(self) -> None:
        figure = plt.figure()
        axes = figure.subplots()
        legend_lines = []
        for index, rp in enumerate(self._run_parameters):
            # If run_parameters["parameters"].trajectory_filename is a complete path, the division operator of paths
            # only returns that complete path. Otherwise, this combines base_path and path.
            trajectory_path = rp.path / rp.run_parameters.trajectory_filename
            if not trajectory_path.exists() and trajectory_path.is_file():
                raise ValueError(f"The trajectory file {trajectory_path} does not exist.")
            if not trajectory_path.suffix == ".gsd":
                raise ValueError(f"The trajectory file {trajectory_path} does not have the .gsd extension.")
            universe = MDAnalysis.Universe(trajectory_path, in_memory=True)
            if not self._types[0] in rp.run_parameters.masses:
                raise ValueError(f"The atom type {self._types[0]} is not in the masses of the run parameters.")
            if not self._types[1] in rp.run_parameters.masses:
                raise ValueError(f"The atom type {self._types[1]} is not in the masses of the run parameters.")
            atoms = universe.select_atoms(" or ".join(f"name {t}" for t in self._types))
            first_atom_group = universe.select_atoms(f"name {self._types[0]}")
            second_atom_group = universe.select_atoms(f"name {self._types[1]}")

            rdf = MDAnalysis.analysis.rdf.InterRDF_s(universe, [[atoms, atoms]],
                                                     range=self._coordination_number_range, norm="none")
            rdf.run(start=self._frame_start, stop=self._frame_stop, step=self._frame_step)
            cdf = rdf.get_cdf()[0]
            # cdf is of shape (len(atoms), len(atoms), len(rdf.results.bins))
            # For every atom from the two groups, it contains the number of nearby particles up until the radius
            # corresponding to the current bin.
            # Use the last bin to get the total number of nearby particles within the given range.
            n_atoms = cdf[:, :, -1]
            # Coordination numbers include the particle itself, so the diagonal of the matrix should be 1.0.
            assert np.all(np.diagonal(n_atoms) == 1.0)
            np.fill_diagonal(n_atoms, 0.0)
            n_atoms_sum = np.sum(n_atoms, axis=1)
            # Coordination numbers should be symmetric if the same atom group is used.
            assert np.all(n_atoms_sum == np.sum(n_atoms, axis=0))
            bins = [-0.5 + 1.0 * i for i in range(int(max(n_atoms_sum)) + 2)]
            _, _, l = axes.hist(n_atoms_sum, bins=bins, density=True, histtype="step", color=f"C{index}",
                                linestyle="solid", label=rp.label)
            legend_lines.append(l[0])

            rdf = MDAnalysis.analysis.rdf.InterRDF_s(universe, [[first_atom_group, second_atom_group]],
                                                     range=self._coordination_number_range, norm="none")
            rdf.run(start=self._frame_start, stop=self._frame_stop, step=self._frame_step)
            cdf = rdf.get_cdf()[0]
            n_atoms = cdf[:, :, -1]
            # Sum over the atoms of the second group to get the coordination number for every atom of the first group.
            n_atoms_sum = np.sum(n_atoms, axis=1)
            bins = [-0.5 + 1.0 * i for i in range(int(max(n_atoms_sum)) + 2)]
            axes.hist(n_atoms_sum, bins=bins, density=True, histtype="step", color=f"C{index}", linestyle="dashed")
            # Sum over the atoms of the first group to get the coordination number for every atom of the second group.
            n_atoms_sum = np.sum(n_atoms, axis=0)
            bins = [-0.5 + 1.0 * i for i in range(int(max(n_atoms_sum)) + 2)]
            axes.hist(n_atoms_sum, bins=bins, density=True, histtype="step", color=f"C{index}", linestyle="dotted")

        # Add two legends according to
        # https://matplotlib.org/3.8.2/users/explain/axes/legend_guide.html#multiple-legends-on-the-same-axes
        first_legend = axes.legend(handles=legend_lines, loc="upper right")
        axes.add_artist(first_legend)
        l1, = axes.plot([], [], color="k", linestyle="solid", label=f"{self._types} neighbors of {self._types}")
        l2, = axes.plot([], [], color="k", linestyle="dashed", label=f"{self._types[1]} neighbors of {self._types[0]}")
        l3, = axes.plot([], [], color="k", linestyle="dotted", label=f"{self._types[0]} neighbors of {self._types[1]}")
        axes.legend(handles=[l1, l2, l3], loc="upper left")
        axes.set_xlabel(f"number of neighbors within range {self._coordination_number_range}")
        axes.set_ylabel(r"density")
        figure.savefig(self._working_directory / "coordination_numbers.pdf")
        figure.clear()
