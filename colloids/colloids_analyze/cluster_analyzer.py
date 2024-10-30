from functools import partial
from typing import Sequence
import ovito.io
import ovito.modifiers
from colloids.colloids_analyze import LabeledRunParametersWithPath, Plotter
from colloids.colloids_analyze.ovito_modifier import modify


class ClusterAnalyzer(Plotter):
    def __init__(self, working_directory: str, run_parameters: Sequence[LabeledRunParametersWithPath],
                 frame_index: int = -1, particle_types: Sequence[str] = ("P", "N"), l: int = 6,
                 q_threshold: float = 0.7, solid_threshold: int = 4, normalize_q: bool = True,
                 r_max: float = 250.0) -> None:
        super().__init__(working_directory, run_parameters)
        self._frame_index = frame_index
        self._particle_types = particle_types
        self._l = l
        self._q_threshold = q_threshold
        self._solid_threshold = solid_threshold
        self._normalize_q = normalize_q
        self._r_max = r_max
        self._modify_function = partial(modify, l=l, q_threshold=q_threshold, solid_threshold=solid_threshold,
                                        normalize_q=normalize_q, r_max=r_max)

    def plot(self) -> None:
        for index, rp in enumerate(self._run_parameters):
            # If run_parameters["parameters"].trajectory_filename is a complete path, the division operator of paths
            # only returns that complete path. Otherwise, this combines base_path and path.
            trajectory_path = rp.path / rp.run_parameters.trajectory_filename
            if not trajectory_path.exists() and trajectory_path.is_file():
                raise ValueError(f"The trajectory file {trajectory_path} does not exist.")
            if not trajectory_path.suffix == ".gsd":
                raise ValueError(f"The trajectory file {trajectory_path} does not have the .gsd extension.")
            xyz_path = trajectory_path.with_stem(trajectory_path.stem + "_cluster")
            xyz_path = xyz_path.with_suffix(".xyz")

            pipeline = ovito.io.import_file(trajectory_path)
            pipeline.modifiers.append(ovito.modifiers.SelectTypeModifier(
                operate_on="particles", property="Particle Type", types=set(self._particle_types)))
            pipeline.modifiers.append(self._modify_function)

            ovito.io.export_file(
                pipeline, xyz_path, "xyz",
                frame=self._frame_index if self._frame_index >= 0 else pipeline.num_frames + self._frame_index,
                columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z",
                         "Color.R", "Color.G", "Color.B", "Radius", "Charge", "Cluster"])
