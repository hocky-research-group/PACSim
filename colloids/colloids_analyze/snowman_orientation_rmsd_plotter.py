from typing import Sequence
from matplotlib.backends.backend_pdf import PdfPages
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
        with PdfPages(self._working_directory / "rmsd.pdf") as pdf:
            full_mean_square_rotations = {}
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
                    # angular_distances = np.empty((len(snowman_body_group), len(universe.trajectory) - self._start_frame),
                    #                                dtype=float)

                    for frame_index, _ in enumerate(universe.trajectory[self._start_frame:]):
                        for body_index, snowman_body in enumerate(snowman_body_group):
                            snowman_head = snowman_head_group[snowman_indices[body_index]]
                            snowman_distance_vector = snowman_head.position - snowman_body.position
                            # Use coords because it has higher precision than snowman_distance_vector.
                            coords[body_index, frame_index] = snowman_distance_vector.copy()
                            coords[body_index, frame_index] /= np.linalg.norm(coords[body_index, frame_index])
                            assert np.linalg.norm(coords[body_index, frame_index]) - 1.0 < 1.0e-12
                            theta = np.arccos(coords[body_index, frame_index][2])  # Polar angle in [0, pi).
                            phi = np.arctan2(coords[body_index, frame_index][1], coords[body_index, frame_index][0])  # Azimuthal angle in [-pi, pi).
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
                            #if frame_index == 0:
                            #    angular_distances[body_index, frame_index] = 0.0
                            #else:
                            #    dot_product = np.dot(coords[body_index, frame_index], coords[body_index, 0])
                            #    assert dot_product < 1.0
                            #    angular_distances[body_index, frame_index] = np.arccos(dot_product)
                    # Find time index where angular distance exceeds 3.0 and gets close to critical area around pi.
                    # critical_time_index = np.min(np.nonzero(angular_distances > 3.0)[1])

                    # plt.figure()
                    # for body_index, snowman_body in enumerate(snowman_body_group):
                    #     plt.plot(angular_distances[body_index, :critical_time_index], color="C0", alpha=0.01)
                    # plt.xlabel("Frame index")
                    # plt.ylabel("Angular distance")
                    # plt.show()
                    # plt.close()

                    mean_square_rotations = [[] for _ in range(len(snowman_body_group))]
                    delta_frame = 0
                    critical_frame_delta_reached = False
                    while True:
                        square_rotations = [[] for _ in range(len(snowman_body_group))]
                        for body_index, snowman_body in enumerate(snowman_body_group):
                            for frame_one in range(coords.shape[1]):
                                frame_two = frame_one + delta_frame
                                if frame_two >= coords.shape[1]:
                                    break
                                if frame_one == frame_two:
                                    square_rotations[body_index].append(0.0)
                                else:
                                    dot_product = np.dot(coords[body_index, frame_one], coords[body_index, frame_two])
                                    assert dot_product <= 1.0 + 1.0e-12
                                    rotation = np.arccos(min(dot_product, 1.0))
                                    if rotation >= 3.0:
                                        critical_frame_delta_reached = True
                                        break  # Break frame_two loop.
                                    square_rotations[body_index].append(rotation * rotation)
                                if critical_frame_delta_reached:
                                    break  # Break frame_one loop.
                            if critical_frame_delta_reached:
                                break  # Break body_index loop.
                        if critical_frame_delta_reached:
                            print("Critical frame delta: ", delta_frame)
                            break  # Break while loop.
                        else:
                            assert all(len(square_rotations[body_index]) == coords.shape[1] - delta_frame
                                       for body_index in range(len(snowman_body_group)))
                            for body_index in range(len(snowman_body_group)):
                                mean_square_rotations[body_index].append(np.mean(square_rotations[body_index]))
                            delta_frame += 1
                    assert all(len(mean_square_rotations[body_index]) == delta_frame
                               for body_index in range(len(snowman_body_group)))

                    fig = plt.figure()
                    for body_index, snowman_body in enumerate(snowman_body_group):
                        plt.plot(mean_square_rotations[body_index], color="C0", alpha=0.01)
                    full_mean_square_rotations[snowman_body_type + ", " + rp.label] = np.mean(
                        np.array(mean_square_rotations), axis=0)
                    plt.plot(full_mean_square_rotations[snowman_body_type + ", " + rp.label], color="C1")
                    plt.title(f"{rp.label}, Snowman body type: {snowman_body_type}")
                    plt.xlabel(r"$\Delta t$")
                    plt.ylabel(r"$\langle \theta^2 \rangle$")
                    pdf.savefig(fig)
                    fig.clear()
                    plt.close(fig)

            fig = plt.figure()
            for label, msr in full_mean_square_rotations.items():
                plt.plot(msr, label=label)
            plt.xlabel(r"$\Delta t$")
            plt.ylabel(r"$\langle \theta^2 \rangle$")
            plt.legend()
            pdf.savefig(fig)
            fig.clear()
            plt.close(fig)

            fig = plt.figure()
            for label, msr in full_mean_square_rotations.items():
                plt.plot(msr, label=label)
            plt.xlabel(r"$\Delta t$")
            plt.ylabel(r"$\langle \theta^2 \rangle$")
            plt.xscale("log")
            plt.yscale("log")
            plt.legend()
            pdf.savefig(fig)
            fig.clear()
            plt.close(fig)
