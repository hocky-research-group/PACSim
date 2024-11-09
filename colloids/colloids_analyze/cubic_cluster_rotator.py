from functools import partial
from typing import Sequence
import MDAnalysis.analysis.distances
import numpy as np
import ovito.data
import ovito.io
import ovito.modifiers
import scipy.spatial.transform
from colloids.colloids_analyze import LabeledRunParametersWithPath, PlotterWithClusterIndex


class CubicClusterRotator(PlotterWithClusterIndex):
    def __init__(self, working_directory: str, run_parameters: Sequence[LabeledRunParametersWithPath],
                 cluster_index: int = 0, cubic_type: str = "P") -> None:
        super().__init__(working_directory, run_parameters, cluster_index)
        self._cubic_type = cubic_type

    @staticmethod
    def _modify(_: int, data: ovito.data.DataCollection, cubic_type: str) -> None:
        data_types = np.array([data.particles["Particle Type"].type_by_id(identifier).name
                                   for identifier in data.particles["Particle Type"]])
        cubic_type_selection = (data_types == cubic_type)
        cubic_type_positions = data.particles["Position"][cubic_type_selection]
        assert len(cubic_type_positions) > 0
        distances = MDAnalysis.analysis.distances.distance_array(cubic_type_positions, cubic_type_positions)
        assert np.all(np.diagonal(distances) == 0.0)

        for tolerance in range(10, 110, 10):
            for index, first_position in enumerate(cubic_type_positions):
                relevant_distances = distances[index]
                # Find seven closest surrounding particles because particle itself is included.
                closest_indices = np.argsort(relevant_distances)[:7]
                closest_distances = relevant_distances[closest_indices]
                assert closest_distances[0] == 0.0  # Distance of particle to itself.
                # Check if particle well within cluster.
                if np.allclose(closest_distances[1:], closest_distances[1], rtol=0.0, atol=tolerance):
                    # Find the closest index with the smallest difference along the z-axis.
                    close_positions = cubic_type_positions[closest_indices]
                    close_z_values = close_positions[:, 2]
                    ref_z_value = first_position[2]
                    z_diff = np.abs(close_z_values - ref_z_value)
                    assert z_diff[0] == 0.0
                    min_indices = np.argsort(z_diff)
                    assert min_indices[0] == 0
                    # This position will be rotated on the x-axis.
                    second_position = cubic_type_positions[closest_indices[min_indices[1]]]

                    diff_vector = second_position - first_position
                    # Find and exclude the position on that will be rotated on the x-axis together with the second position.
                    second_position_prime = first_position - diff_vector
                    distances = np.array([np.linalg.norm(second_position_prime - close_position)
                                          for close_position in close_positions])
                    excluded_index = np.argmin(distances)
                    # Average the two vectors pointing in opposite directions.
                    other_diff_vector = -(close_positions[excluded_index] - first_position)
                    new_x_vector = (diff_vector + other_diff_vector) / 2.0

                    third_index = min_indices[2] if excluded_index != min_indices[2] else min_indices[3]
                    third_position = cubic_type_positions[closest_indices[third_index]]
                    # Repeat averaging for the y-axis.
                    diff_vector = third_position - first_position
                    third_position_prime = first_position - diff_vector
                    distances = np.array([np.linalg.norm(third_position_prime - close_position)
                                            for close_position in close_positions])
                    excluded_index = np.argmin(distances)
                    other_diff_vector = -(close_positions[excluded_index] - first_position)
                    new_y_vector = (diff_vector + other_diff_vector) / 2.0

                    # Apply the rotation.
                    rot = scipy.spatial.transform.Rotation.align_vectors(
                        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                        np.row_stack((new_x_vector, new_y_vector)),
                        weights=[np.inf, 1])[0]
                    rot_matrix = rot.as_matrix()
                    translation_vector = -first_position
                    full_matrix = np.column_stack((rot_matrix, translation_vector))
                    data.apply(ovito.modifiers.AffineTransformationModifier(transformation=full_matrix))

                    # If z-axis is pointing downwards, rotate the system by 180 degrees around the y-axis.
                    if data.cell[2, 2] <= 0.0:
                        new_rot_matrix = scipy.spatial.transform.Rotation.from_euler("y", np.pi).as_matrix()
                        new_translation_vector = np.array([0.0, 0.0, 0.0])
                        new_full_matrix = np.column_stack((new_rot_matrix, new_translation_vector))
                        data.apply(ovito.modifiers.AffineTransformationModifier(transformation=new_full_matrix))
                    return
        raise RuntimeError("No suitable cubic particle found for rotation")

    def plot(self) -> None:
        for index, rp in enumerate(self._run_parameters):
            # If run_parameters["parameters"].trajectory_filename is a complete path, the division operator of paths
            # only returns that complete path. Otherwise, this combines base_path and path.
            trajectory_path = rp.path / rp.run_parameters.trajectory_filename
            cluster_xyz_path = trajectory_path.with_stem(trajectory_path.stem + "_cluster")
            cluster_xyz_path = cluster_xyz_path.with_suffix(".xyz")
            if not cluster_xyz_path.exists() and cluster_xyz_path.is_file():
                raise ValueError(f"The cluster xyz file {cluster_xyz_path} does not exist, run the ClusterAnalyzer "
                                 f"first.")
            if self._cubic_type not in rp.run_parameters.masses:
                raise ValueError(f"The cubic type {self._cubic_type} is not present the run parameters.")

            pipeline = ovito.io.import_file(cluster_xyz_path)
            # Filter out non-cluster particles.
            pipeline.modifiers.append(ovito.modifiers.ExpressionSelectionModifier(
                operate_on="particles", expression=f"Cluster == {self._cluster_index}"))
            pipeline.modifiers.append(ovito.modifiers.InvertSelectionModifier(operate_on="particles"))
            pipeline.modifiers.append(ovito.modifiers.DeleteSelectedModifier())
            pipeline.modifiers.append(partial(self._modify, cubic_type=self._cubic_type))
            assert pipeline.num_frames == 1
            xyz_path = trajectory_path.with_stem(trajectory_path.stem + f"_cluster_{self._cluster_index}_rotated")
            xyz_path = xyz_path.with_suffix(".xyz")
            try:
                ovito.io.export_file(
                    pipeline, xyz_path, "xyz", frame=0,
                    columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z",
                             "Radius", "Charge"])
            except RuntimeError as error:
                print(f"Rotating cluster in file {cluster_xyz_path} failed with error: {error}.")