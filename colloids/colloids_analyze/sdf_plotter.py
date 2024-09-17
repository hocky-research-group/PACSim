from typing import Optional, Sequence
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import MDAnalysis
import MDAnalysis.analysis.distances
import MDAnalysis.analysis.rdf
import numpy as np
from openmm import unit
import scipy.spatial.transform
from colloids.colloids_analyze import LabeledRunParametersWithPath, Plotter


class SDFPlotter(Plotter):
    _nanometer = unit.nano * unit.meter

    def __init__(self, working_directory: str, run_parameters: Sequence[LabeledRunParametersWithPath],
                 sdf_ranges: Sequence[Sequence[int]] = ((-1500, 1500), (-1500, 1500), (-1500, 1500)),
                 n_bins: Sequence[int] = (150, 150, 150), frame_index: int = -1, types: Sequence[str] = ("N", "P"),
                 align_snowman: bool = False, sdf_threshold: float = 0.0, radius_cap: Optional[float] = None,
                 open_interactive: bool = False):
        super().__init__(working_directory, run_parameters)
        if not len(sdf_ranges) == 3:
            raise ValueError("The ranges for the spatial distribution function must have exactly three elements for "
                             "the three cartesian directions.")
        if not all(len(r) == 2 for r in sdf_ranges):
            raise ValueError("Each range for the spatial distribution function must have exactly two elements.")
        if not all(r[0] < r[1] for r in sdf_ranges):
            raise ValueError("The lower limit of each range for the spatial distribution function must be less than "
                             "the upper limit.")
        if not len(n_bins) == 3:
            raise ValueError("The number of bins for the spatial distribution function must have exactly three "
                             "elements.")
        if not all(n > 0 for n in n_bins):
            raise ValueError("The number of bins for the spatial distribution function must be positive.")
        if not len(types) == 2:
            raise ValueError("The types sequence must have exactly two elements.")
        if not sdf_threshold >= 0.0:
            raise ValueError("The threshold for the spatial distribution function must be non-negative.")
        if radius_cap is not None:
            if not radius_cap > 0.0:
                raise ValueError("The radius cap must be positive.")
        self._sdf_ranges = list(list(r) for r in sdf_ranges)
        self._n_bins = list(n_bins)
        self._frame_index = frame_index
        self._types = list(types)
        self._align_snowman = align_snowman
        self._sdf_threshold = sdf_threshold
        self._radius_cap = radius_cap
        self._open_interactive = open_interactive

    def plot(self) -> None:
        with PdfPages(self._working_directory / "sdf.pdf") as pdf:
            for index, rp in enumerate(self._run_parameters):
                # If run_parameters["parameters"].trajectory_filename is a complete path, the division operator of paths
                # only returns that complete path. Otherwise, this combines base_path and path.
                trajectory_path = rp.path / rp.run_parameters.trajectory_filename
                if not trajectory_path.exists() and trajectory_path.is_file():
                    raise ValueError(f"The trajectory file {trajectory_path} does not exist.")
                if not trajectory_path.suffix == ".gsd":
                    raise ValueError(f"The trajectory file {trajectory_path} does not have the .gsd extension.")
                if not self._types[0] in rp.run_parameters.masses:
                    raise ValueError(f"The atom type {self._types[0]} is not in the masses of the run parameters.")
                if not self._types[1] in rp.run_parameters.masses:
                    raise ValueError(f"The atom type {self._types[1]} is not in the masses of the run parameters.")
                universe = MDAnalysis.Universe(trajectory_path, in_memory=True)
                # Advance to correct frame index.
                _ = universe.trajectory[self._frame_index]
                first_atom_group = universe.select_atoms(f"name {self._types[0]}")
                second_atom_group = universe.select_atoms(f"name {self._types[1]}")
                if self._align_snowman:
                    if not self._types[0] in rp.run_parameters.snowman_bond_types:
                        raise ValueError(f"The atom type {self._types[0]} is not in the snowman bond types of the run "
                                         f"parameters.")
                    snowman_type = rp.run_parameters.snowman_bond_types[self._types[0]]
                    snowman_atom_group = universe.select_atoms(f"name {snowman_type}")
                    snowman_distance = rp.run_parameters.snowman_distances[self._types[0]].value_in_unit(
                        self._nanometer)
                    distances = MDAnalysis.analysis.distances.distance_array(first_atom_group.positions,
                                                                             snowman_atom_group.positions)
                    snowman_indices = np.zeros(len(first_atom_group), dtype=int)
                    for first_index, first_atom in enumerate(first_atom_group):
                        relevant_distances = distances[first_index]
                        relevant_snowman_indices = np.nonzero(np.abs(relevant_distances - snowman_distance) < 1.0e-1)[0]
                        assert len(relevant_snowman_indices) == 1
                        snowman_indices[first_index] = relevant_snowman_indices[0]
                    assert np.all(np.sort(snowman_indices) == np.arange(len(first_atom_group)))
                else:
                    snowman_atom_group = None
                    snowman_indices = None
                x_bin_edges = np.linspace(start=self._sdf_ranges[0][0], stop=self._sdf_ranges[0][1],
                                          num=self._n_bins[0] + 1, endpoint=True)
                y_bin_edges = np.linspace(start=self._sdf_ranges[1][0], stop=self._sdf_ranges[1][1],
                                          num=self._n_bins[1] + 1, endpoint=True)
                z_bin_edges = np.linspace(start=self._sdf_ranges[2][0], stop=self._sdf_ranges[2][1],
                                          num=self._n_bins[2] + 1, endpoint=True)
                sdf = np.zeros((self._n_bins[0], self._n_bins[1], self._n_bins[2]))
                for first_index, first_atom in enumerate(first_atom_group):
                    if self._align_snowman:
                        snowman_atom = snowman_atom_group[snowman_indices[first_index]]
                        snowman_distance_vector = snowman_atom.position - first_atom.position
                        assert np.abs(np.linalg.norm(snowman_distance_vector) - snowman_distance) < 1.0e-1
                        # noinspection PyUnresolvedReferences
                        rot = scipy.spatial.transform.Rotation.align_vectors(np.array([0.0, 0.0, 1.0]).reshape(1, 3),
                                                                             snowman_distance_vector.reshape(1, 3))[0]
                    else:
                        # noinspection PyUnresolvedReferences
                        rot = scipy.spatial.transform.Rotation.identity()
                    for second_index, second_atom in enumerate(second_atom_group):
                        if self._types[0] == self._types[1] and first_index == second_index:
                            continue
                        distance_vector = second_atom.position - first_atom.position
                        distance_vector = rot.apply(distance_vector)
                        x_bin_index = np.argmax(x_bin_edges > distance_vector[0]) - 1
                        assert 0 <= x_bin_index < self._n_bins[0] or x_bin_index == -1
                        y_bin_index = np.argmax(y_bin_edges > distance_vector[1]) - 1
                        assert 0 <= y_bin_index < self._n_bins[1] or y_bin_index == -1
                        z_bin_index = np.argmax(z_bin_edges > distance_vector[2]) - 1
                        assert 0 <= z_bin_index < self._n_bins[2] or z_bin_index == -1
                        if x_bin_index != -1 and y_bin_index != -1 and z_bin_index != -1:
                            sdf[x_bin_index, y_bin_index, z_bin_index] += 1
                x_coords = []
                y_coords = []
                z_coords = []
                for x_index in range(self._n_bins[0]):
                    for y_index in range(self._n_bins[1]):
                        for z_index in range(self._n_bins[2]):
                            if sdf[x_index, y_index, z_index] > self._sdf_threshold:
                                x_coord = (x_bin_edges[x_index] + x_bin_edges[x_index + 1]) / 2.0
                                y_coord = (y_bin_edges[y_index] + y_bin_edges[y_index + 1]) / 2.0
                                z_coord = (z_bin_edges[z_index] + z_bin_edges[z_index + 1]) / 2.0
                                if (self._radius_cap is None
                                    or np.linalg.norm([x_coord, y_coord, z_coord]) <= self._radius_cap):
                                    x_coords.append(x_coord)
                                    y_coords.append(y_coord)
                                    z_coords.append(z_coord)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.scatter(x_coords, y_coords, z_coords, marker="o", alpha=0.1)
                ax.set_xlabel("x / nm")
                ax.set_ylabel("y / nm")
                ax.set_zlabel("z / nm")
                ax.set_title(f"{rp.label}, SDF between {self._types[0]}--{self._types[1]}, "
                             f"align snowman {self._align_snowman}, \n"
                             f"threshold {self._sdf_threshold} (max value {np.max(sdf)}), "
                             f"radius cap {self._radius_cap}")
                ax.set_aspect("equal")
                if self._open_interactive:
                    plt.show()
                pdf.savefig(fig)
                fig.clear()
                plt.close(fig)
