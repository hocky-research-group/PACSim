from typing import Optional, Sequence
import matplotlib.pyplot as plt
import MDAnalysis
import MDAnalysis.analysis.rdf
from colloids.colloids_analyze import LabeledRunParametersWithPath, Plotter


class RDFPlotter(Plotter):
    def __init__(self, working_directory: str, run_parameters: Sequence[LabeledRunParametersWithPath],
                 rdf_range: tuple[int, int] = (0, 1500), frame_start: Optional[int] = -1,
                 frame_stop: Optional[int] = None, frame_step: Optional[int] = None):
        super().__init__(working_directory, run_parameters)
        if not all(r >= 0 for r in rdf_range):
            raise ValueError("The range of the radial distribution function must be non-negative.")
        if not rdf_range[0] < rdf_range[1]:
            raise ValueError("The lower limit of the radial distribution function must be less than the upper limit.")
        self._rdf_range = rdf_range
        self._frame_start = frame_start
        self._frame_stop = frame_stop
        self._frame_step = frame_step

    def plot(self) -> None:
        rdf_figure = plt.figure()
        rdf_axes = rdf_figure.subplots()
        for index, rp in enumerate(self._run_parameters):
            # If run_parameters["parameters"].trajectory_filename is a complete path, the division operator of paths
            # only returns that complete path. Otherwise, this combines base_path and path.
            trajectory_path = rp.path / rp.run_parameters.trajectory_filename
            if not trajectory_path.exists() and trajectory_path.is_file():
                raise ValueError(f"The trajectory file {trajectory_path} does not exist.")
            if not trajectory_path.suffix == ".gsd":
                raise ValueError(f"The trajectory file {trajectory_path} does not have the .gsd extension.")
            universe = MDAnalysis.Universe(trajectory_path, in_memory=True)
            types = set(k for k in rp.run_parameters.masses.keys())
            if rp.run_parameters.substrate_type is not None:
                assert rp.run_parameters.substrate_type in types
                types.remove(rp.run_parameters.substrate_type)
            if rp.run_parameters.snowman_bond_types is not None:
                for t in rp.run_parameters.snowman_bond_types.values():
                    assert t in types
                    types.remove(t)
            atoms = universe.select_atoms(" or ".join(f"name {t}" for t in types))
            rdf = MDAnalysis.analysis.rdf.InterRDF(atoms, atoms, exclusion_block=(1, 1), range=self._rdf_range)
            rdf.run(start=self._frame_start, stop=self._frame_stop, step=self._frame_step)
            rdf_axes.plot(rdf.results.bins, rdf.results.rdf, label=rp.label, color=f"C{index}")
        rdf_axes.set_xlabel(r"distance $r$ / nm")
        rdf_axes.set_ylabel(r"radial distribution function $g(r)$")
        rdf_axes.legend()
        rdf_figure.savefig(self._working_directory / "rdf.pdf")
        rdf_figure.clear()
