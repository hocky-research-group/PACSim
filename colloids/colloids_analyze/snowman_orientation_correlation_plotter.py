from typing import Sequence
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import MDAnalysis
import MDAnalysis.analysis
import numpy as np
from openmm import unit
from colloids.colloids_analyze import LabeledRunParametersWithPath, Plotter


class SnowmanOrientationCorrelationPlotter(Plotter):
    _nanometer = unit.nano * unit.meter

    def __init__(self, working_directory: str, run_parameters: Sequence[LabeledRunParametersWithPath],
                 start_frame: int = 0) -> None:
        super().__init__(working_directory, run_parameters)
        self._start_frame = start_frame

    @staticmethod
    def autocorr_fft(x):
        assert len(x.shape) == 2
        n, d = x.shape
        # pad 0s to 2n-1
        ext_size = 2 * n - 1
        # nearest power of 2
        fsize = 2 ** np.ceil(np.log2(ext_size)).astype('int')

        xp = x - np.mean(x, axis=0)
        autocorr = np.zeros(n)

        for i in range(d):
            cf = np.fft.fft(xp[:, i], fsize)
            sf = cf.conjugate() * cf
            corr = np.fft.ifft(sf).real
            corr = corr[:n] / np.arange(n, 0, -1)
            autocorr += corr

        return autocorr / autocorr[0]

    def plot(self) -> None:
        with PdfPages(self._working_directory / "orientation_autocorrelation.pdf") as pdf:
            full_mean_autocorrelation_functions = {}
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

                    for frame_index, _ in enumerate(universe.trajectory[self._start_frame:]):
                        for body_index, snowman_body in enumerate(snowman_body_group):
                            snowman_head = snowman_head_group[snowman_indices[body_index]]
                            snowman_distance_vector = snowman_head.position - snowman_body.position
                            # Use coords because it has higher precision than snowman_distance_vector.
                            coords[body_index, frame_index] = snowman_distance_vector.copy()
                            coords[body_index, frame_index] /= np.linalg.norm(coords[body_index, frame_index])
                            assert np.linalg.norm(coords[body_index, frame_index]) - 1.0 < 1.0e-12

                    autocorrelation_functions = np.empty(coords.shape[:2], dtype=float)
                    fig = plt.figure()
                    for body_index in range(len(snowman_body_group)):
                        autocorrelation_functions[body_index] = self.autocorr_fft(coords[body_index])
                        plt.plot(autocorrelation_functions[body_index], color="C0", alpha=0.1)
                    full_mean_autocorrelation_functions[snowman_body_type + ", " + rp.label] = np.mean(
                        autocorrelation_functions, axis=0)
                    plt.plot(full_mean_autocorrelation_functions[snowman_body_type + ", " + rp.label], color="C1",
                             marker=".")
                    plt.title(f"{rp.label}, Snowman body type: {snowman_body_type}")
                    plt.xlabel(r"$\Delta t$")
                    plt.ylabel(r"$\langle \hat{\mathbf{r}}_i(t)\cdot\hat{\mathbf{r}}_i(t + \Delta t) \rangle$")
                    plt.yscale("log")
                    #plt.xlim(0, np.where(full_mean_autocorrelation_functions[snowman_body_type + ", " + rp.label] < 1.0e-2)[0][0])
                    #plt.ylim(1.0e-2, 1.0)
                    pdf.savefig(fig)
                    fig.clear()
                    plt.close(fig)

            fig = plt.figure()
            for label, autocorr in full_mean_autocorrelation_functions.items():
                plt.plot(autocorr, label=label, marker=".")
            plt.xlabel(r"$\Delta t$")
            plt.ylabel(r"$\langle \hat{\mathbf{r}}_i(t)\cdot\hat{\mathbf{r}}_i(t + \Delta t) \rangle$")
            plt.yscale("log")
            plt.legend()
            pdf.savefig(fig)
            fig.clear()
            plt.close(fig)
