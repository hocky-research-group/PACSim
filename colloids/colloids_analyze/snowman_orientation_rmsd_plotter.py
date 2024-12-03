from typing import Sequence
import matplotlib.pyplot as plt
import MDAnalysis
import MDAnalysis.analysis
import numpy as np
from openmm import unit
from colloids.colloids_analyze import LabeledRunParametersWithPath, Plotter


class SnowmanOrientationRMSDPlotter(Plotter):
    _nanometer = unit.nano * unit.meter

    def __init__(self, working_directory: str, run_parameters: Sequence[LabeledRunParametersWithPath],
                 start_frame: int = 0) -> None:
        super().__init__(working_directory, run_parameters)
        self._start_frame = start_frame

    def plot(self) -> None:
        for index, rp in enumerate(self._run_parameters):
            # If run_parameters["parameters"].trajectory_filename is a complete path, the division operator of paths
            # only returns that complete path. Otherwise, this combines base_path and path.
            trajectory_path = rp.path / rp.run_parameters.trajectory_filename
            if not trajectory_path.exists() or not trajectory_path.is_file():
                raise ValueError(f"The trajectory file {trajectory_path} does not exist.")
            if not trajectory_path.suffix == ".gsd":
                raise ValueError(f"The trajectory file {trajectory_path} does not have the .gsd extension.")
            universe = MDAnalysis.Universe(trajectory_path, in_memory=True)
            # Advance to correct frame index.
            _ = universe.trajectory[self._start_frame]

            for i, (snowman_body_type, snowman_head_type) in enumerate(
                    rp.run_parameters.snowman_bond_types.items()):
                # Find the snomwan mapping in the first frame.
                snowman_distance = rp.run_parameters.snowman_distances[snowman_body_type].value_in_unit(self._nanometer)
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

                coords = np.empty((len(snowman_body_group), len(universe.trajectory) - self._start_frame, 3),
                                    dtype=float)
                unwrapped_phis = np.empty((len(snowman_body_group), len(universe.trajectory) - self._start_frame),
                                          dtype=float)
                unwrapped_thetas = np.empty((len(snowman_body_group), len(universe.trajectory) - self._start_frame),
                                            dtype=float)
                angular_distances = np.empty((len(snowman_body_group), len(universe.trajectory) - self._start_frame),
                                                dtype=float)

                for frame_index, _ in enumerate(universe.trajectory[self._start_frame:]):
                    for body_index, snowman_body in enumerate(snowman_body_group):
                        snowman_head = snowman_head_group[snowman_indices[body_index]]
                        snowman_distance_vector = snowman_head.position - snowman_body.position
                        snowman_distance_vector /= np.linalg.norm(snowman_distance_vector)
                        assert np.abs(np.linalg.norm(snowman_distance_vector) - 1.0) < 1.0e-6
                        coords[body_index, frame_index] = snowman_distance_vector.copy()
                        theta = np.arccos(snowman_distance_vector[2])  # Polar angle in [0, pi).
                        phi = np.arctan2(snowman_distance_vector[1], snowman_distance_vector[0])  # Azimuthal angle in [-pi, pi).
                        # Choose the shortest path to the previous frame.
                        # Use reference for zeroth frame so that initial value stays unchanged.
                        ref_theta = np.pi / 2.0 if frame_index == 0 else unwrapped_thetas[body_index, frame_index - 1]
                        ref_phi = 0.0 if frame_index == 0 else unwrapped_phis[body_index, frame_index - 1]
                        # Shortest separation for theta ranges from -pi/2 to pi/2.
                        shortest_diff_theta = (theta - ref_theta + np.pi / 2.0) % np.pi - np.pi / 2.0
                        # Shortest separation for phi ranges from -pi to pi.
                        shortest_diff_phi = (phi - ref_phi + np.pi) % (2.0 * np.pi) - np.pi
                        unwrapped_phis[body_index, frame_index] = ref_phi + shortest_diff_phi
                        unwrapped_thetas[body_index, frame_index] = ref_theta + shortest_diff_theta
                        if frame_index == 0:
                            angular_distances[body_index, frame_index] = 0.0
                        else:
                            dot_product = np.dot(coords[body_index, frame_index], coords[body_index, 0])
                            assert dot_product < 1.0
                            angular_distances[body_index, frame_index] = np.arccos(dot_product)
                # Find time index where angular distance exceeds 3.0 and gets close to critical area around pi.
                critical_time_index = np.min(np.nonzero(angular_distances > 3.0)[1])

                # I have to average here over Delta t :(

                plt.figure()
                for body_index, snowman_body in enumerate(snowman_body_group):
                    plt.plot(angular_distances[body_index, :critical_time_index], color="C0", alpha=0.01)
                plt.xlabel("Frame index")
                plt.ylabel("Angular distance")
                plt.show()
                plt.close()
