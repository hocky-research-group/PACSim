from functools import partial
from typing import Sequence
import MDAnalysis.analysis.distances
import numpy as np
from openmm import unit
import ovito.data
import ovito.io
import ovito.modifiers
from colloids.colloids_analyze import LabeledRunParametersWithPath, Plotter
from colloids.colloids_analyze.ovito_modifier import modify


class ClusterAnalyzer(Plotter):
    _nanometer = unit.nano * unit.meter

    def __init__(self, working_directory: str, run_parameters: Sequence[LabeledRunParametersWithPath],
                 frame_index: int = -1, particle_types: Sequence[str] = ("P", "N"), l: int = 6,
                 q_threshold: float = 0.7, solid_threshold: int = 4, normalize_q: bool = True,
                 r_max: float = 250.0, add_snowman_heads: bool = True) -> None:
        super().__init__(working_directory, run_parameters)
        self._frame_index = frame_index
        self._particle_types = particle_types
        self._l = l
        self._q_threshold = q_threshold
        self._solid_threshold = solid_threshold
        self._normalize_q = normalize_q
        self._r_max = r_max
        self._add_snowman_heads = add_snowman_heads
        self._modify_function = partial(modify, l=l, q_threshold=q_threshold, solid_threshold=solid_threshold,
                                        normalize_q=normalize_q, r_max=r_max)

    @staticmethod
    def _add_snowman_heads(_: int, data: ovito.data.DataCollection, snowman_bond_types: dict[str, str],
                           snowman_distances: dict[str, unit.Quantity]) -> None:
        for snowman_body_type, snowman_head_type in snowman_bond_types.items():
            assert snowman_body_type in snowman_distances
            snowman_distance = snowman_distances[snowman_body_type].value_in_unit(ClusterAnalyzer._nanometer)
            data_types = np.array([data.particles["Particle Type"].type_by_id(identifier).name
                                   for identifier in data.particles["Particle Type"]])

            snowman_body_selection = (data_types == snowman_body_type)
            snowman_head_selection = (data_types == snowman_head_type)
            snowman_body_positions = data.particles["Position"][snowman_body_selection]
            snowman_head_positions = data.particles["Position"][snowman_head_selection]
            assert len(snowman_body_positions) == len(snowman_head_positions)
            snowman_body_clusters = data.particles["Cluster"][snowman_body_selection]
            assert np.all(data.particles["Cluster"][snowman_head_selection] == -1)
            snowman_body_colors = data.particles["Color"][snowman_body_selection]
            assert len(snowman_body_positions) == len(snowman_head_positions)
            snowman_head_cluster_idx = np.full(len(snowman_head_positions), -1, dtype=int)
            snowman_head_colors = np.full((len(snowman_head_positions), 3), 0.0)

            distances = MDAnalysis.analysis.distances.distance_array(snowman_body_positions, snowman_head_positions)
            for body_index, body_position in enumerate(snowman_body_positions):
                relevant_distances = distances[body_index]
                relevant_head_indices = np.nonzero(np.abs(relevant_distances - snowman_distance) < 1.0e-1)[0]
                assert len(relevant_head_indices) == 1
                head_index = relevant_head_indices[0]
                snowman_head_cluster_idx[head_index] = snowman_body_clusters[body_index]
                snowman_head_colors[head_index] = snowman_body_colors[body_index]

            data.particles_["Cluster_"][snowman_head_selection] = snowman_head_cluster_idx
            data.particles_["Color_"][snowman_head_selection] = snowman_head_colors

    def plot(self) -> None:
        for index, rp in enumerate(self._run_parameters):
            # If run_parameters["parameters"].trajectory_filename is a complete path, the division operator of paths
            # only returns that complete path. Otherwise, this combines base_path and path.
            trajectory_path = rp.path / rp.run_parameters.trajectory_filename
            if not trajectory_path.exists() or not trajectory_path.is_file():
                raise ValueError(f"The trajectory file {trajectory_path} does not exist.")
            if not trajectory_path.suffix == ".gsd":
                raise ValueError(f"The trajectory file {trajectory_path} does not have the .gsd extension.")
            xyz_path = trajectory_path.with_stem(trajectory_path.stem + "_cluster")
            xyz_path = xyz_path.with_suffix(".xyz")

            pipeline = ovito.io.import_file(trajectory_path)
            pipeline.modifiers.append(ovito.modifiers.SelectTypeModifier(
                operate_on="particles", property="Particle Type", types=set(self._particle_types)))
            pipeline.modifiers.append(self._modify_function)
            if self._add_snowman_heads:
                pipeline.modifiers.append(partial(ClusterAnalyzer._add_snowman_heads,
                                                  snowman_bond_types=rp.run_parameters.snowman_bond_types,
                                                  snowman_distances=rp.run_parameters.snowman_distances))

            ovito.io.export_file(
                pipeline, xyz_path, "xyz",
                frame=self._frame_index if self._frame_index >= 0 else pipeline.num_frames + self._frame_index,
                columns=["Particle Type", "Position.X", "Position.Y", "Position.Z", "Particle Identifier",
                         "Color.R", "Color.G", "Color.B", "Radius", "Charge", "Cluster"])
