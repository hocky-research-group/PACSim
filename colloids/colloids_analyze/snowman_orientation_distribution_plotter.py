from typing import Optional, Sequence
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import MDAnalysis
import MDAnalysis.analysis.distances
import numpy as np
from openmm import unit
from colloids.colloids_analyze import LabeledRunParametersWithPath, PlotterWithClusterIndex


class SnowmanOrientationDistributionPlotter(PlotterWithClusterIndex):
    _nanometer = unit.nano * unit.meter

    def __init__(self, working_directory: str, run_parameters: Sequence[LabeledRunParametersWithPath],
                 frame_index: int = -1, open_interactive: bool = False, cluster_index: Optional[int] = None):
        super().__init__(working_directory, run_parameters, cluster_index)
        self._frame_index = frame_index
        self._open_interactive = open_interactive

    def plot(self) -> None:
        with PdfPages(self._working_directory / "snowman_orientation_distribution.pdf") as pdf:
            for index, rp in enumerate(self._run_parameters):
                # If run_parameters["parameters"].trajectory_filename is a complete path, the division operator of paths
                # only returns that complete path. Otherwise, this combines base_path and path.
                trajectory_path = rp.path / rp.run_parameters.trajectory_filename
                if not trajectory_path.exists() and trajectory_path.is_file():
                    raise ValueError(f"The trajectory file {trajectory_path} does not exist.")
                if not trajectory_path.suffix == ".gsd":
                    raise ValueError(f"The trajectory file {trajectory_path} does not have the .gsd extension.")
                universe = MDAnalysis.Universe(trajectory_path, in_memory=True)
                # Advance to correct frame index.
                _ = universe.trajectory[self._frame_index]
                cluster_map = self._get_cluster_map(trajectory_path, len(universe.atoms))

                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                for i, (snowman_body_type, snowman_head_type) in enumerate(
                        rp.run_parameters.snowman_bond_types.items()):
                    snowman_distance = rp.run_parameters.snowman_distances[snowman_body_type].value_in_unit(
                        self._nanometer)
                    snowman_body_group = universe.select_atoms(f"name {snowman_body_type}")
                    snowman_head_group = universe.select_atoms(f"name {snowman_head_type}")
                    distances = MDAnalysis.analysis.distances.distance_array(
                        snowman_body_group.positions, snowman_head_group.positions)
                    snowman_indices = np.zeros(len(snowman_body_group), dtype=int)
                    for body_index, snowman_body in enumerate(snowman_body_group):
                        relevant_distances = distances[body_index]
                        relevant_head_indices = np.nonzero(np.abs(relevant_distances - snowman_distance) < 1.0e-1)[0]
                        assert len(relevant_head_indices) == 1
                        snowman_indices[body_index] = relevant_head_indices[0]
                    assert np.all(np.sort(snowman_indices) == np.arange(len(snowman_body_group)))
                    x_coords = []
                    y_coords = []
                    z_coords = []
                    for body_index, snowman_body in enumerate(snowman_body_group):
                        snowman_head = snowman_head_group[snowman_indices[body_index]]

                        # It's enough if either the head or the body are part of the cluster.
                        if (cluster_map[snowman_body.id] != self._cluster_index
                                and cluster_map[snowman_head.id] != self._cluster_index):
                            continue

                        snowman_distance_vector = snowman_head.position - snowman_body.position
                        assert np.abs(np.linalg.norm(snowman_distance_vector) - snowman_distance) < 1.0e-1
                        snowman_distance_vector /= np.linalg.norm(snowman_distance_vector)
                        assert np.abs(np.linalg.norm(snowman_distance_vector) - 1.0) < 1.0e-6
                        x_coords.append(snowman_distance_vector[0])
                        y_coords.append(snowman_distance_vector[1])
                        z_coords.append(snowman_distance_vector[2])
                    ax.scatter(x_coords, y_coords, z_coords, marker="o", alpha=0.1, color=f"C{i}")
                ax.set_title(f"{rp.label}, orientation of snowmen, cluster {self._cluster_index}")
                ax.set_aspect("equal")
                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(-1.1, 1.1)
                ax.set_zlim(-1.1, 1.1)
                if self._open_interactive:
                    plt.show()
                pdf.savefig(fig)
                fig.clear()
                plt.close(fig)
