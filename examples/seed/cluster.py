import gdown
import gsd.hoomd
import numpy as np
import freud
from colloids.colloids_create.trajectory_wrapper import TrajectoryWrapper as trajectory_wrapper

if __name__ == "__main__":
    # Download the trajectory file
    url = 'https://drive.google.com/uc?id=1upqsXeeYmUBP1eGSGxVsRTNcGLQZgcUt'
    gdown.download(url, "crystal_trajectory.gsd", quiet=False)

    # Cluster analysis and saving the largest cluster
    # Open the trajectory file
    traj = gsd.hoomd.open('crystal_trajectory.gsd', 'r')
    last_frame: gsd.hoomd.Frame = traj[-1]

    # Get particle positions
    positions = last_frame.particles.position
    box = freud.box.Box.from_box(last_frame.configuration.box)

    # Use freud to find clusters (DBSCAN example)
    cluster = freud.cluster.Cluster()
    cluster.compute((box, positions), neighbors={'r_max': 230.0})
    cluster_indices = cluster.cluster_keys

    # Get biggest cluster
    cluster_sizes = [len(cluster_indices[i]) for i in range(len(cluster_indices))]
    largest_cluster_index = np.argmax(cluster_sizes)

    # Get the largest cluster's particle indices
    largest_cluster = cluster_indices[largest_cluster_index]

    # Load trajectory wrapper
    wrapper = trajectory_wrapper('crystal_trajectory.gsd', frame=-1)
    wrapper.subset_particles(largest_cluster)
    wrapper.save_current_frame('crystal_seed.gsd')