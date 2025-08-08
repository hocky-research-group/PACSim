import os
import numpy as np
import gsd.hoomd

class TrajectoryWrapper:
    """
    A wrapper class for handling GSD HOOMD trajectories, providing utilities for
    frame manipulation, particle selection, and trajectory saving.
    """

    def __init__(self, filename: str = None, trajectory: gsd.hoomd.HOOMDTrajectory = None, frame=None):
        """
        Initialize the TrajectoryWrapper.

        Args:
            filename (str, optional): Path to the GSD trajectory file.
            trajectory (gsd.hoomd.HOOMDTrajectory, optional): An existing trajectory object.
            frame (int, optional): The initial frame index to use.

        Raises:
            Exception: If neither filename nor trajectory is provided.
            FileNotFoundError: If the specified file does not exist.
        """
        if trajectory is None and filename is None:
            raise Exception("File name or trajectory must be specified")
        self.filename = filename
        if trajectory is not None:
            traj = trajectory
        else:
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"File {filename} does not exist")
            traj = gsd.hoomd.open(name=filename, mode='r')
        self.traj: list[gsd.hoomd.Frame] = [frame for frame in traj]
        self.current_frame = frame if frame is not None else 0

    def __getitem__(self, index):
        """
        Get a frame or a slice of frames from the trajectory.

        Args:
            index (int or slice): Frame index or slice.

        Returns:
            gsd.hoomd.Frame or list[gsd.hoomd.Frame]: The requested frame(s).

        Raises:
            IndexError: If the index is out of range.
        """
        if isinstance(index, slice):
            return [self.traj[i] for i in range(*index.indices(len(self.traj)))]
        elif isinstance(index, int):
            if index < 0:
                index += len(self.traj)
            if index < 0 or index >= len(self.traj):
                raise IndexError("Index out of range")
            return self.traj[index]
        else:
            return self.traj[index]
                
    def __len__(self):
        """
        Get the number of frames in the trajectory.

        Returns:
            int: Number of frames.
        """
        return len(self.traj)
    
    def set_current_frame(self, index: int):
        """
        Set the current frame index.

        Args:
            index (int): Frame index.

        Raises:
            IndexError: If the index is out of range.
        """
        if index < (1 - len(self.traj)) or index >= len(self.traj):
            raise IndexError("Index out of range")
        if index < 0:
            index += len(self.traj)
        self.current_frame = index

    def wrap_frame(self):
        """
        Wrap the positions of particles in the current frame to the simulation box.
        """
        frame: gsd.hoomd.Frame = self.traj[self.current_frame]
        box = frame.configuration.box[:3]
        frame.particles.position -= box / 2
        frame.particles.position %= box
        frame.particles.position -= box / 2
        self.traj[self.current_frame] = frame
    
    def subset_particles(self, indices: np.ndarray):
        """
        Create a new frame containing only the specified particle indices.

        Args:
            indices (np.ndarray): Indices of particles to keep.

        Modifies:
            self.traj[self.current_frame]: Replaces with the subsetted frame.
        """
        frame: gsd.hoomd.Frame = self.traj[self.current_frame]
        index_matching_array = -1 * np.ones(len(frame.particles.position), dtype=int)
        for new_idx, old_idx in enumerate(indices):
            index_matching_array[old_idx] = new_idx

        new_frame = gsd.hoomd.Frame()
        for attr in frame.particles.__dict__:
            if not attr.startswith('_'):
                data = getattr(frame.particles, attr)
                if data is not None:
                    if attr == 'N':
                        setattr(new_frame.particles, attr, len(indices))
                    elif attr == 'types':
                        setattr(new_frame.particles, attr, frame.particles.types)
                    elif hasattr(data, 'shape') and data.shape[0] == frame.particles.N:
                        setattr(new_frame.particles, attr, data[indices])
                else:
                    setattr(new_frame.particles, attr, data)
        new_frame.configuration.box = frame.configuration.box

        # Filter and remap constraints
        if hasattr(frame, 'constraints') and hasattr(frame.constraints, 'group') and frame.constraints.group is not None:
            constraints = frame.constraints.group
            mask = np.all(index_matching_array[constraints] != -1, axis=1)
            new_constraints = index_matching_array[constraints[mask]]
            new_values = frame.constraints.value[mask]
            new_frame.constraints.N = len(new_constraints)
            new_frame.constraints.group = new_constraints
            new_frame.constraints.value = new_values
        else:
            new_frame.constraints.N = 0
            new_frame.constraints.group = np.empty((0, 2), dtype=int)
            new_frame.constraints.value = np.empty((0,), dtype=int)

        self.traj[self.current_frame] = new_frame

    def superset_particles(self, other: 'TrajectoryWrapper'):
        """
        Add particles from another trajectory's current frame to this trajectory's current frame.

        Args:
            other (TrajectoryWrapper): The other trajectory to combine.

        Raises:
            TypeError: If other is not a TrajectoryWrapper.
            ValueError: If particle data shapes are unsupported.

        Modifies:
            self.traj[self.current_frame]: Replaces with the combined frame.
        """
        if not isinstance(other, TrajectoryWrapper):
            raise TypeError("Other must be a TrajectoryWrapper instance")

        other_frame: gsd.hoomd.Frame = other[self.current_frame]
        base_frame: gsd.hoomd.Frame = self.traj[self.current_frame]
        new_frame = gsd.hoomd.Frame()

        for attr in other_frame.particles.__dict__:
            if not attr.startswith('_'):
                other_data = getattr(other_frame.particles, attr)
                base_data = getattr(base_frame.particles, attr)

                if attr == 'N':
                    # Update the number of particles
                    data = base_frame.particles.N + other_frame.particles.N
                elif attr == 'types':
                    # Combine particle types
                    data = base_frame.particles.types
                    
                elif other_data is not None and base_data is not None:
                    if not isinstance(other_data, np.ndarray):
                        other_data = np.array(other_data)
                    if not isinstance(base_data, np.ndarray):
                        base_data = np.array(base_data)

                    if (hasattr(other_data, 'shape') and hasattr(base_data, 'shape')
                        and other_data.shape[0] == other_frame.particles.N
                        and base_data.shape[0] == base_frame.particles.N):

                        if other_data.ndim == 1:
                            # Append 1D data
                            data = np.concatenate((base_data, other_data))
                        elif other_data.ndim == 2 and base_data.ndim == 2:
                            # Append 2D data
                            data = np.vstack((base_data, other_data))
                        else:
                            raise ValueError(f"Unsupported data shape for attribute '{attr}'")
                        
                    else:
                        data = base_data
                else:
                    data = None

                setattr(new_frame.particles, attr, data)

        # Filter and remap constraints
        if (hasattr(other_frame, 'constraints') and hasattr(other_frame.constraints, 'group') and other_frame.constraints.group is not None
            and (hasattr(base_frame, 'constraints') and hasattr(base_frame.constraints, 'group') and base_frame.constraints.group is not None)):
            other_constraints = other_frame.constraints.group
            base_constraints = base_frame.constraints.group
            new_constraints = np.vstack((base_constraints, other_constraints + base_frame.particles.N))
            new_values = np.concatenate((base_frame.constraints.value, other_frame.constraints.value))
            new_frame.constraints.N = other_frame.constraints.N + base_frame.constraints.N
            new_frame.constraints.group = new_constraints
            new_frame.constraints.value = new_values
        else:
            new_frame.constraints.N = 0
            new_frame.constraints.group = np.empty((0, 2), dtype=int)
            new_frame.constraints.value = np.empty((0,), dtype=int)

        new_frame.configuration.box = base_frame.configuration.box
        self.traj[self.current_frame] = new_frame

    def seed_particles(self, seed: 'TrajectoryWrapper', epsilon: float = 0, seed_fractional_coords: np.ndarray = None):
        """
        Seed the current frame with particles from another trajectory's current frame.
        Optionally, place the seed at a specified fractional coordinate in the box.

        Args:
            seed (TrajectoryWrapper): The trajectory to seed from.
            epsilon (float, optional): Allowed overlap tolerance.
            seed_fractional_coords (np.ndarray, optional): Fractional coordinates for seed placement.

        Raises:
            TypeError: If seed is not a TrajectoryWrapper.

        Modifies:
            self.traj[self.current_frame]: May be replaced with seeded frame.
        """
        if not isinstance(seed, TrajectoryWrapper):
            raise TypeError("Seed must be a TrajectoryWrapper instance")
        
        seed.wrap_frame()
        self.wrap_frame()

        seed_frame: gsd.hoomd.Frame = seed[seed.current_frame]
        seed_positions = seed_frame.particles.position 
        seed_radii = seed_frame.particles.diameter / 2

        current_frame: gsd.hoomd.Frame = self.traj[self.current_frame]
        current_box = current_frame.configuration.box[:3]
        current_positions = current_frame.particles.position 
        current_radii = current_frame.particles.diameter / 2

        if seed_fractional_coords is not None:
            seed_location = current_box * seed_fractional_coords
        else:
            seed_location = current_box / 2

        seed_location -= current_box / 2  
        seed_positions += (seed_location - np.mean(seed_positions, axis=0))

        distance_matrix = np.linalg.norm(current_positions[:, None] - seed_positions[None, :], axis=2)
        overlap_matrix = (distance_matrix - current_radii[:, None] - seed_radii[None, :]) < epsilon
        overlapped_indices = np.where(~np.any(overlap_matrix, axis=1))[0]

        if np.any(~overlapped_indices):
            self.subset_particles(overlapped_indices)

        if len(seed_positions) > 0:
            self.superset_particles(seed)

    def save(self, filename: str):
        """
        Save the entire trajectory to a file.

        Args:
            filename (str): Output file path.
        """
        with gsd.hoomd.open(name=filename, mode='w') as f:
            for frame in self.traj:
                f.append(frame)

    def save_current_frame(self, filename: str):
        """
        Save only the current frame to a file.

        Args:
            filename (str): Output file path.
        """
        with gsd.hoomd.open(name=filename, mode='w') as f:
            f.append(self.traj[self.current_frame])


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python continue_seed.py <trajectory_file>")
        sys.exit(1)

    filename = sys.argv[1]
    traj = TrajectoryWrapper(filename=filename)
    traj2 = TrajectoryWrapper(filename=filename)
    print(traj[traj.current_frame].constraints.group)  # Print bond type IDs in current frame

    traj.subset_particles(np.arange(10))  # Example subset
    print(traj[traj.current_frame].constraints.group)  # Print particle types in current frame
    traj2.subset_particles(np.arange(10) + 200)  # Example subset
    traj.superset_particles(traj2)  # Combine with another trajectory
    print((traj[traj.current_frame].particles.N))  # Print number of particles after superset
