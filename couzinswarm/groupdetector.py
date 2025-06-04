import numpy as np
from collections import defaultdict
import sys
sys.path.append("/home/ggu7596/ML_collective_behaviors/softwares/couzinswarm")
from couzinswarm.objects import Fish
from couzinswarm.verletlist import VerletList

class GroupDetector:
    """
    Class to detect groups in a swarm based on the Couzin model.
    """

    def __init__(self, fish_list, verlet_list,box_lengths=None,reflect_at_boundary = None):
        """
        Initialize the GroupDetector with fish list and a Verlet list.

        Parameters
        ----------
        fish_list : list of Fish
            List of fish objects.
        verlet_list : VerletList
            The VerletList object to get neighbors.
        """
        self.fish_list = fish_list
        self.verlet_list = verlet_list
        self.box_lengths = np.array(box_lengths)
        self.reflect_at_boundary = reflect_at_boundary
    def detect_groups(self,f_list = None):
        if f_list is None:
            f_list = self.fish_list
        """
        Detect groups in the fish swarm.

        Returns
        -------
        groups : list of lists
            Each sublist contains the IDs of fish that belong to the same group.
        """
        visited = set()
        groups = []

        for fish in f_list:
            #print("hi!")
            if fish.ID in visited:
                continue

            # Perform a BFS to find all connected fish
            group = self._bfs_group_detection(fish)
            visited.update(group)
            groups.append(group)

        return groups

    def _bfs_group_detection(self, fish):
        """
        Perform a Breadth-First Search (BFS) to detect all fish in the same group.

        Parameters
        ----------
        fish : Fish
            The starting fish for BFS.

        Returns
        -------
        group : list
            List of fish IDs that are in the same group as the starting fish.
        """
        queue = [fish]
        group = [fish.ID]
        visited = set(group)

        while queue:
            current_fish = queue.pop(0)
            neighbors = self.verlet_list.get_neighbors(current_fish)

            for neighbor in neighbors:
                if neighbor.ID not in visited:
                    visited.add(neighbor.ID)
                    queue.append(neighbor)
                    group.append(neighbor.ID)

        return group
    
    # compute the unwrapped positions
    def detect_groups_v2(self, f_list=None):
        """
        Detect groups in the fish swarm, accounting for PBC, and store unwrapped positions and directions.

        Parameters
        ----------
        box_lengths : array-like
            The dimensions of the periodic box [Lx, Ly, Lz].
        f_list : list of Fish, optional
            List of fish to consider for group detection. If None, use all fish.

        Returns
        -------
        groups : list of lists
            Each sublist contains the IDs of fish that belong to the same group.
        """
        if f_list is None:
            f_list = self.fish_list

        visited = set()
        groups = []
        unwrapped_positions = {}
        unwrapped_directions = {}

        for fish in f_list:
            if fish.ID in visited:
                continue

            # Perform a BFS to find all connected fish, unwrapped positions, and directions
            group, unwrapped, dir_map = self._bfs_group_detection_v2(fish)
            visited.update(group)
            groups.append(group)

            # Store unwrapped positions and directions for each fish in the group
            unwrapped_positions.update(unwrapped)
            unwrapped_directions.update(dir_map)
        return unwrapped_positions,unwrapped_directions

    def _bfs_group_detection_v2(self, fish):
        """
        Perform a Breadth-First Search (BFS) to detect all fish in the same group, accounting for PBC.

        Parameters
        ----------
        fish : Fish
            The starting fish for BFS.

        Returns
        -------
        group : list
            List of fish IDs that are in the same group as the starting fish.
        unwrapped_positions : dict
            Dictionary mapping fish IDs to their unwrapped positions.
        """
        queue = [fish]
        group = [fish.ID]
        visited = set(group)

        # Initialize unwrapped positions
        unwrapped_positions = {fish.ID: np.array(fish.position)}
        unwrapped_directions =  {fish.ID: np.array(fish.direction)}
        while queue:
            current_fish = queue.pop(0)
            neighbors = self.verlet_list.get_neighbors(current_fish)

            # Retrieve the unwrapped position of the current fish
            current_position = unwrapped_positions[current_fish.ID]

            for neighbor in neighbors:
                if neighbor.ID not in visited:
                    visited.add(neighbor.ID)
                    queue.append(neighbor)
                    group.append(neighbor.ID)

                    # Unwrap neighbor's position relative to the current fish
                    unwrapped_position = np.array(neighbor.position)
                    for dim in range(3):  # Adjust for each dimension
                        if not self.reflect_at_boundary[dim]:
                            relative_position = neighbor.position[dim] - current_position[dim]
                            unwrapped_position[dim] -= self.box_lengths[dim] * np.round(relative_position / self.box_lengths[dim])

                    unwrapped_positions[neighbor.ID] = unwrapped_position
                    unwrapped_directions[neighbor.ID] = np.array(neighbor.direction)
        return group, unwrapped_positions,unwrapped_directions



# Example usage
if __name__ == "__main__":
    # Define parameters for the simulation
    num_fish = 20
    repulsion_radius = 1.0
    orientation_width = 10.0
    attraction_width = 10.0
    box_lengths = [50.0, 50.0, 50.0]
    reflect_at_boundary = [True, True, True]

    # Initialize the fish and verlet list
    fish_list = [Fish(position=np.random.rand(3) * box_lengths, ID=i) for i in range(num_fish)]
    verlet_list = VerletList(
        fish_list=fish_list,
        repulsion_radius=repulsion_radius,
        orientation_width=orientation_width,
        attraction_width=attraction_width,
        box_lengths=box_lengths,
        reflect_at_boundary=reflect_at_boundary
    )

    # Initialize the GroupDetector and detect groups
    group_detector = GroupDetector(fish_list, verlet_list)
    groups = group_detector.detect_groups()

    # Print the detected groups
    print("Detected groups:")
    for i, group in enumerate(groups):
        print(f"Group {i+1}: {group}")
