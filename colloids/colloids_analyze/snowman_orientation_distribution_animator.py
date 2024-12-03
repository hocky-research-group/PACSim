from typing import Optional, Sequence
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.widgets
import mpl_toolkits.axes_grid1
import MDAnalysis
import MDAnalysis.analysis
import numpy as np
from openmm import unit
from colloids.colloids_analyze import LabeledRunParametersWithPath, PlotterWithClusterIndex


# See https://stackoverflow.com/questions/44985966/managing-dynamic-plotting-in-matplotlib-animation-module
class Player(FuncAnimation):
    def __init__(self, fig, func, init_func=None, fargs=None, save_count=None, min_frame=0, max_frame=100,
                 pos=(0.125, 0.92), **kwargs):
        self.current_frame = min_frame
        self.min_frame = min_frame
        self.max_frame = max_frame
        self.runs = True
        self.forwards = True
        self.fig = fig
        self.func = func
        self.time_test = None
        self.setup(pos)
        super().__init__(self.fig, self.func, frames=self.play(), init_func=init_func, fargs=fargs,
                         save_count=save_count, **kwargs)

    def play(self):
        while self.runs:
            # Increases or decreases the frame number depending on self._forwards.
            self.current_frame = self.current_frame + self.forwards - (not self.forwards)
            if self.min_frame < self.current_frame < self.max_frame:
                self.time_text.set_text(f"frame={self.current_frame}")
                yield self.current_frame
            else:
                self.stop()
                self.time_text.set_text(f"frame={self.current_frame}")
                yield self.current_frame

    def start(self):
        self.runs = True
        self.event_source.start()

    def stop(self, *_, **__):
        self.runs = False
        self.event_source.stop()

    def forward(self, *_, **__):
        self.forwards = True
        self.start()

    def backward(self, *_, **__):
        self.forwards = False
        self.start()

    def one_forward(self, *_, **__):
        self.forwards = True
        self.one_step()

    def one_backward(self, *_, **__):
        self.forwards = False
        self.one_step()

    def one_step(self):
        if self.min_frame < self.current_frame < self.max_frame:
            self.current_frame = self.current_frame + self.forwards - (not self.forwards)
        elif self.current_frame == self.min_frame and self.forwards:
            self.current_frame += 1
        elif self.current_frame == self.max_frame and not self.forwards:
            self.current_frame -= 1
        self.func(self.current_frame)
        self.time_text.set_text(f"frame={self.current_frame}")
        self.fig.canvas.draw_idle()

    def setup(self, pos):
        playerax = self.fig.add_axes([pos[0], pos[1], 0.22, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        tax = divider.append_axes("right", size="100%", pad=0.05)
        tax.axis("off")
        self.button_oneback = matplotlib.widgets.Button(playerax, label=u'\u29CF')
        self.button_back = matplotlib.widgets.Button(bax, label=u'\u25C0')
        self.button_stop = matplotlib.widgets.Button(sax, label=u'\u25A0')
        self.button_forward = matplotlib.widgets.Button(fax, label=u'\u25B6')
        self.button_oneforward = matplotlib.widgets.Button(ofax, label=u'\u29D0')
        self.time_text = tax.text(0.5, 0.5, f"frame={self.min_frame}", verticalalignment="center",
                                  horizontalalignment="left")
        self.button_oneback.on_clicked(self.one_backward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.one_forward)


class SnowmanOrientationDistributionAnimator(PlotterWithClusterIndex):
    _nanometer = unit.nano * unit.meter

    def __init__(self, working_directory: str, run_parameters: Sequence[LabeledRunParametersWithPath],
                 cluster_index: Optional[int] = None, highlight_ids=Optional[Sequence[int]],
                 start_frame: int = 0) -> None:
        super().__init__(working_directory, run_parameters, cluster_index)
        self._highlight_ids = highlight_ids
        self._start_frame = start_frame

    def plot(self) -> None:
        old_rc_params = plt.rcParams.copy()
        # Switch of LaTeX rendering for the plot labels in the animation (switched on in colloids.colloids_analyze.abstracts).
        plt.rcParams.update({"text.usetex": False})

        for index, rp in enumerate(self._run_parameters):
            if not len(rp.run_parameters.snowman_bond_types) == 1:
                raise ValueError("The number of snowman bond types must be exactly one.")

            # If run_parameters["parameters"].trajectory_filename is a complete path, the division operator of paths
            # only returns that complete path. Otherwise, this combines base_path and path.
            trajectory_path = rp.path / rp.run_parameters.trajectory_filename
            if not trajectory_path.exists() or not trajectory_path.is_file():
                raise ValueError(f"The trajectory file {trajectory_path} does not exist.")
            if not trajectory_path.suffix == ".gsd":
                raise ValueError(f"The trajectory file {trajectory_path} does not have the .gsd extension.")
            universe = MDAnalysis.Universe(trajectory_path, in_memory=True)
            cluster_map = self._get_cluster_map(trajectory_path, len(universe.atoms))

            # Find the snowman mapping in the first frame
            snowman_body_type = list(rp.run_parameters.snowman_bond_types.keys())[0]
            snowman_head_type = list(rp.run_parameters.snowman_bond_types.values())[0]
            snowman_distance = rp.run_parameters.snowman_distances[snowman_body_type].value_in_unit(self._nanometer)
            snowman_body_group = universe.select_atoms(f"name {snowman_body_type}")
            snowman_head_group = universe.select_atoms(f"name {snowman_head_type}")
            if self._highlight_ids is not None:
                highlight_body_indices = [-1 for _ in self._highlight_ids]
                for i, atom in enumerate(snowman_body_group):
                    if atom.index in self._highlight_ids:
                        highlight_body_indices[self._highlight_ids.index(atom.index)] = i
                        print(self._highlight_ids[self._highlight_ids.index(atom.index)])
                        print(atom.position)
                for i, found_index in enumerate(highlight_body_indices):
                    if found_index == -1:
                        print(f"[WARNING] Could not find the atom index {self._highlight_ids[i]} in the system.")
                    assert snowman_body_group[found_index].index == self._highlight_ids[i]
            distances = MDAnalysis.analysis.distances.distance_array(
                snowman_body_group.positions, snowman_head_group.positions)
            snowman_indices = np.zeros(len(snowman_body_group), dtype=int)
            for body_index, snowman_body in enumerate(snowman_body_group):
                relevant_distances = distances[body_index]
                relevant_head_indices = np.nonzero(np.abs(relevant_distances - snowman_distance) < 1.0e-1)[0]
                assert len(relevant_head_indices) == 1
                snowman_indices[body_index] = relevant_head_indices[0]
            assert np.all(np.sort(snowman_indices) == np.arange(len(snowman_body_group)))

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            number_snowman = len(universe.select_atoms(f"name {list(rp.run_parameters.snowman_bond_types.keys())[0]}"))
            x_coords, y_coords, z_coords = np.zeros(number_snowman), np.zeros(number_snowman), np.zeros(number_snowman)
            scatter = ax.scatter(x_coords, y_coords, z_coords, marker="o", alpha=0.1, color="C0")
            if self._highlight_ids is not None:
                highlight_x_coords = np.zeros(len(self._highlight_ids))
                highlight_y_coords = np.zeros(len(self._highlight_ids))
                highlight_z_coords = np.zeros(len(self._highlight_ids))
                highlight_scatter = ax.scatter(highlight_x_coords, highlight_y_coords, highlight_z_coords, marker="o",
                                               alpha=1.0, color="C3")
            else:
                highlight_scatter = None
            ax.set_aspect("equal")
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)
            ax.set_zlim(-1.1, 1.1)

            def update(frame_index):
                _ = universe.trajectory[frame_index]
                for body_index, snowman_body in enumerate(snowman_body_group):
                    snowman_head = snowman_head_group[snowman_indices[body_index]]
                    assert cluster_map[snowman_body.id] == cluster_map[snowman_head.id]
                    if cluster_map[snowman_body.id] != self._cluster_index:
                        continue
                    snowman_distance_vector = snowman_head.position - snowman_body.position
                    assert np.abs(np.linalg.norm(snowman_distance_vector) - snowman_distance) < 1.0e-1
                    snowman_distance_vector /= np.linalg.norm(snowman_distance_vector)
                    assert np.abs(np.linalg.norm(snowman_distance_vector) - 1.0) < 1.0e-6
                    x_coords[body_index] = snowman_distance_vector[0]
                    y_coords[body_index] = snowman_distance_vector[1]
                    z_coords[body_index] = snowman_distance_vector[2]
                    if highlight_scatter is not None and snowman_body.index in self._highlight_ids:
                        highlight_x_coords[self._highlight_ids.index(snowman_body.index)] = x_coords[body_index]
                        highlight_y_coords[self._highlight_ids.index(snowman_body.index)] = y_coords[body_index]
                        highlight_z_coords[self._highlight_ids.index(snowman_body.index)] = z_coords[body_index]
                        highlight_scatter._offsets3d = (highlight_x_coords, highlight_y_coords, highlight_z_coords)
                # See https://stackoverflow.com/questions/41602588/how-to-create-3d-scatter-animations
                scatter._offsets3d = (x_coords, y_coords, z_coords)

            # Call update once to set the initial frame.
            update(self._start_frame)
            ani = Player(fig, update, min_frame=self._start_frame, max_frame=len(universe.trajectory) - 1,
                         save_count=len(universe.trajectory))
            # TODO: Allow to save with option
            # ani.save("snowman_orientation_distribution.mp4")
            plt.show()
            plt.close()

        # Reset rcParams.
        plt.rcParams = old_rc_params
