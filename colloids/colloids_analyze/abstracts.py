from abc import ABC, abstractmethod
from dataclasses import dataclass
import pathlib
from typing import Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
from colloids.run_parameters import RunParameters

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "text.latex.preamble": r"\usepackage{siunitx}"
})


@dataclass(order=True, frozen=True)
class LabeledRunParametersWithPath(object):
    path: pathlib.Path
    label: str
    run_parameters: RunParameters


class Plotter(ABC):
    def __init__(self, working_directory: str, run_parameters: Sequence[LabeledRunParametersWithPath]):
        self._working_directory = pathlib.Path(working_directory)
        if not self._working_directory.exists() and self._working_directory.is_dir():
            raise ValueError("The working directory does not exist.")
        self._run_parameters = run_parameters

    @abstractmethod
    def plot(self) -> None:
        raise NotImplementedError


class PlotterWithClusterIndex(Plotter, ABC):
    def __init__(self, working_directory: str, run_parameters: Sequence[LabeledRunParametersWithPath],
                 cluster_index: Optional[int]) -> None:
        super().__init__(working_directory, run_parameters)
        if cluster_index is not None and cluster_index < 0:
            raise ValueError("The cluster index must be greater than or equal to zero.")
        # ClusterAnalyzer sets the cluster index to -1 for ignored particles.
        # We use -2 to avoid confusion.
        self._cluster_index = cluster_index if cluster_index is not None else -2

    def _get_cluster_map(self, trajectory_path: pathlib.Path, total_number_atoms: int) -> dict[int, int]:
        if self._cluster_index != -2:
            xyz_path = trajectory_path.with_stem(trajectory_path.stem + "_cluster")
            xyz_path = xyz_path.with_suffix(".xyz")
            if not xyz_path.exists() and xyz_path.is_file():
                raise ValueError(f"The cluster file {xyz_path} does not exist. Run the ClusterAnalyzer first or set "
                                 f"cluster_index to None.")

            xyz_total_number_atoms = None
            with open(xyz_path, "r") as file:
                for index, line in enumerate(file):
                    if index == 0:
                        assert xyz_total_number_atoms is None
                        xyz_total_number_atoms = int(line)
                        continue
                    if index == 1:
                        if not "Properties=id:I:1:species:S:1:pos:R:3:color:R:3:radius:R:1:charge:R:1:cluster:I:1" in line:
                            raise ValueError("Unexpected format of the extended XYZ file, use the ClusterAnalyzer "
                                             "first or set cluster_index to None.")
                    break
            assert xyz_total_number_atoms is not None and xyz_total_number_atoms == total_number_atoms

            data = np.loadtxt(xyz_path, skiprows=2, usecols=(0, 10), dtype=int)
            assert data.shape[0] == total_number_atoms
            data = data[data[:, 0].argsort()]  # Sort by first column.
            # Particle ids start at 1.
            assert np.all(data[:, 0] == np.arange(start=1, stop=total_number_atoms + 1, step=1))
            return {particle_index: cluster_index for particle_index, cluster_index in data}
        else:
            return {particle_index: self._cluster_index for particle_index in range(1, total_number_atoms + 1)}
