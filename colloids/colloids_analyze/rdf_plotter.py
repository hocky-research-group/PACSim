from typing import Optional, Sequence
import matplotlib.pyplot as plt
import MDAnalysis
import MDAnalysis.analysis.rdf
from colloids.colloids_analyze import LabeledRunParametersWithPath, Plotter


class RDFPlotter(Plotter):
    def __init__(self, working_directory: str, run_parameters: Sequence[LabeledRunParametersWithPath],
                 rdf_range: Sequence[int] = (0, 1500), frame_start: Optional[int] = -1,
                 frame_stop: Optional[int] = None, frame_step: Optional[int] = None, types: Sequence[str] = ("P", "N"),
                 vertical_line: Optional[float] = None):
        super().__init__(working_directory, run_parameters)
        if not len(rdf_range) == 2:
            raise ValueError("The range for the coordination number must have exactly two elements.")
        if not all(r >= 0 for r in rdf_range):
            raise ValueError("The range of the radial distribution function must be non-negative.")
        if not rdf_range[0] < rdf_range[1]:
            raise ValueError("The lower limit of the radial distribution function must be less than the upper limit.")
        if not len(types) == 2:
            raise ValueError("The types sequence must have exactly two elements.")
        if vertical_line is not None:
            if not rdf_range[0] <= vertical_line <= rdf_range[1]:
                raise ValueError("The coordinate of the vertical line must be within the range of the radial "
                                 "distribution function.")
        self._rdf_range = list(rdf_range)
        self._frame_start = frame_start
        self._frame_stop = frame_stop
        self._frame_step = frame_step
        self._types = list(types)
        self._vertical_line = vertical_line

    def plot(self) -> None:
        rdf_figure_all = plt.figure()
        rdf_axes_all = rdf_figure_all.subplots()
        rdf_figure_types = plt.figure()
        rdf_axes_types = rdf_figure_types.subplots()
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

            rdf = MDAnalysis.analysis.rdf.InterRDF(atoms, atoms, exclusion_block=(1, 1), range=self._rdf_range)
            rdf.run(start=self._frame_start, stop=self._frame_stop, step=self._frame_step)
            rdf_axes_all.plot(rdf.results.bins, rdf.results.rdf, label=rp.label, color=f"C{index}", linestyle="solid")

            rdf = MDAnalysis.analysis.rdf.InterRDF(first_atom_group, second_atom_group, range=self._rdf_range)
            rdf.run(start=self._frame_start, stop=self._frame_stop, step=self._frame_step)
            rdf_axes_types.plot(rdf.results.bins, rdf.results.rdf, label=rp.label, color=f"C{index}", linestyle="solid")

            if self._vertical_line is not None:
                rdf_axes_all.axvline(self._vertical_line, color="k", linestyle="dashed")
                rdf_axes_types.axvline(self._vertical_line, color="k", linestyle="dashed")

        rdf_axes_all.set_xlabel(r"distance $r$ / nm")
        rdf_axes_all.set_ylabel(r"radial distribution function $g(r)$")
        rdf_axes_types.set_xlabel(r"distance $r$ / nm")
        rdf_axes_types.set_ylabel(r"radial distribution function $g(r)$")
        rdf_axes_all.legend()
        rdf_axes_types.legend()
        rdf_axes_all.set_title(f"RDF for types {self._types}")
        rdf_axes_types.set_title(f"RDF between {self._types[0]}--{self._types[1]}")
        rdf_figure_all.savefig(self._working_directory / "rdf_all.pdf")
        rdf_figure_types.savefig(self._working_directory / "rdf_types.pdf")
        rdf_figure_all.clear()
