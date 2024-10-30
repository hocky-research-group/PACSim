import freud
import matplotlib.pyplot as plt
import numpy as np
import ovito.data


def modify(_: int, data: ovito.data.DataCollection, l: int = 6, q_threshold: float = 0.7, solid_threshold: int = 7,
           normalize_q: bool = True, r_max: float = 100.0, color_map: str = "Set3") -> None:
    # See https://www.ovito.org/docs/current/python/introduction/custom_modifiers.html
    selection_mask = (data.particles["Selection"] == 1)
    # Reset selection so that selection coloring is removed.
    data.particles_.create_property("Selection", data=np.zeros(len(data.particles["Position"]), dtype=int))
    yield
    selected_positions = data.particles["Position"][selection_mask]
    yield
    solid = freud.order.SolidLiquid(l=l, q_threshold=q_threshold, solid_threshold=solid_threshold,
                                    normalize_q=normalize_q)
    yield
    # See https://www.ovito.org/manual/python/modules/ovito_data.html#ovito.data.SimulationCell
    cell = data.cell[:, 0:3]
    enlarge_cell_factor = 10.0  # Enlarge cell so that freud ignores periodic boundaries.
    box = freud.box.Box.from_matrix(cell * enlarge_cell_factor)
    yield
    solid.compute(system=(box, selected_positions), neighbors={"r_max": r_max, "exclude_ii": True})
    yield
    full_cluster_idx = np.full(len(data.particles["Position"]), -1, dtype=int)
    full_cluster_idx[selection_mask] = solid.cluster_idx
    cluster_idx = data.particles_.create_property("Cluster", data=full_cluster_idx)
    data.attributes["Cluster count"] = len(solid.cluster_sizes)
    data.attributes["Cluster sizes"] = solid.cluster_sizes
    data.attributes["Cluster largest"] = solid.cluster_sizes.max()
    yield
    # See https://matplotlib.org/stable/users/explain/colors/colormaps.html
    cmap = plt.get_cmap(color_map)
    output_colors = data.particles_.create_property("Color")
    # Mostly copied from docstring of ovito.modifiers.ColorByTypeModifier class.
    for index, type_id in enumerate(data.particles["Particle Type"]):
        element_type = data.particles["Particle Type"].type_by_id(type_id)
        if cluster_idx[index] < 0:
            output_colors[index] = element_type.color
        else:
            output_colors[index] = cmap(cluster_idx[index] % cmap.N)[:3]
