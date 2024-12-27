from typing import Sequence
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import MDAnalysis
import MDAnalysis.analysis
import numpy as np
from openmm import unit
from scipy.optimize import curve_fit
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
        print(autocorr[0])
        return autocorr / autocorr[0]

    @staticmethod
    def decay(t, a, b):
        return np.exp(-a * (t - b + b * np.exp(-t / b)))

    def plot(self) -> None:
        with PdfPages(self._working_directory / "orientation_autocorrelation.pdf") as pdf:
            full_mean_autocorrelation_functions = {}
            max_index = None
            fits = {}
            x_datas = {}
            for index, rp in enumerate(self._run_parameters):
                print(f"Processing: {rp.run_parameters.trajectory_filename}")
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

                    autocorrelation_functions = np.zeros(coords.shape[:2], dtype=float)
                    fig = plt.figure()
                    for body_index in range(len(snowman_body_group)):
                        autocorrelation_functions[body_index] = self.autocorr_fft(coords[body_index])
                        plt.plot(autocorrelation_functions[body_index], color="C0", alpha=0.1)
                    full_mean_autocorrelation_functions[snowman_body_type + ", " + rp.label] = np.mean(
                        autocorrelation_functions, axis=0)

                    # TODO: REMOVE LATEST COMMIT AGAIN
                    x_data = np.arange(300)  # Free: 36, R10: 80, R30: 300, R40: 600, R60: 600
                    fit = curve_fit(self.decay, x_data,
                                    full_mean_autocorrelation_functions[snowman_body_type + ", " + rp.label][:len(x_data)])
                    fits[snowman_body_type + ", " + rp.label] = fit
                    x_datas[snowman_body_type + ", " + rp.label] = x_data

                    plt.plot(full_mean_autocorrelation_functions[snowman_body_type + ", " + rp.label], color="C1",
                             marker=".")
                    plt.plot(x_data, self.decay(x_data, *fit[0]), color="C2", label=f"Fit: {fit[0]}")
                    plt.title(f"{rp.label}, Snowman body type: {snowman_body_type}, "
                              f"Trajectory length: {len(universe.trajectory) - self._start_frame}")
                    plt.xlabel(r"$\Delta t$")
                    plt.ylabel(r"$\langle \hat{\mathbf{r}}_i(t)\cdot\hat{\mathbf{r}}_i(t + \Delta t) \rangle$")
                    plt.yscale("log")
                    if max_index is None:
                        max_index = (len(universe.trajectory) - self._start_frame) // 10
                    else:
                        max_index = max(max_index, (len(universe.trajectory) - self._start_frame) // 10)
                    plt.xlim(0, (len(universe.trajectory) - self._start_frame) // 10)
                    plt.ylim(1.0e-3, 1.0)
                    pdf.savefig(fig)
                    fig.clear()
                    plt.close(fig)

            fig = plt.figure()

            for i, (label, autocorr) in enumerate(full_mean_autocorrelation_functions.items()):
                plt.plot(autocorr, label=label, marker=".", linestyle="dashed", color=f"C{i}")
                plt.plot(x_datas[label], self.decay(x_datas[label], *fits[label][0]),
                         label=f"{label} fit: {fits[label][0]}", color=f"C{i}")
            plt.xlabel(r"$\Delta t$")
            plt.ylabel(r"$\langle \hat{\mathbf{r}}_i(t)\cdot\hat{\mathbf{r}}_i(t + \Delta t) \rangle$")
            plt.yscale("log")
            plt.xlim(0, max_index)
            plt.ylim(1.0e-3, 1.0)
            plt.legend()
            pdf.savefig(fig)
            fig.clear()
            plt.close(fig)
