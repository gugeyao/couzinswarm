import numpy as np
import sys
sys.path.append("/home/ggu7596/ML_collective_behaviors/softwares/couzinswarm")
from couzinswarm.objects import Fish
from couzinswarm.verletlist import VerletList
from couzinswarm.groupdetector import GroupDetector  # Assuming GroupDetector is in a module named groupdetector
from scipy.spatial import ConvexHull

class GroupAnalyzer:
    """
    Class to analyze groups in a swarm based on Couzin model.
    """

    def __init__(self, fish_list, verlet_list, n_min_group_size,box_lengths=None,reflect_at_boundary = None):
        """
        Initialize the GroupAnalyzer with a fish list, Verlet list, group detector, and minimum group size.

        Parameters
        ----------
        fish_list : list of Fish
            List of fish objects.
        verlet_list : VerletList
            The VerletList object to get neighbors.
        group_detector : GroupDetector
            The GroupDetector object used to detect groups.
        n_min_group_size : int
            Minimum number of members required for the largest group.
        """
        self.fish_list = fish_list
        self.verlet_list = verlet_list
        self.group_detector = GroupDetector(fish_list, verlet_list,box_lengths,reflect_at_boundary)
        self.n_min_group_size = n_min_group_size
        self.box_lengths = np.array(box_lengths)
        self.reflect_at_boundary = reflect_at_boundary
    def detect_largest_group(self):
        groups = self.group_detector.detect_groups()
        #print("groups are "+str(groups))
        # Find the largest group
        largest_group = max(groups, key=len) if groups else []
        
        return largest_group
    def calculate_angular_momentum(self,pos,directions):
        center_of_group = np.mean(pos, axis=0)
        relative_positions = pos - center_of_group
        cross_products = np.cross(relative_positions, directions)
        mean_angular_momentum = np.mean(cross_products, axis=0)
        m = np.linalg.norm(mean_angular_momentum)  # Magnitude of the mean angular momentum vector
        average_distance = np.mean(np.linalg.norm(relative_positions, axis=1))
        normalized_m = m / average_distance if average_distance > 0 else m
        return normalized_m
                
    def compute_group_metrics(self):
        """
        Compute the mean directionality and mean angular momentum of the largest group.

        Returns
        -------
        P : float
            Mean directionality of the largest group, 0 if everything is segmented.
        m : float
            Mean angular momentum of the largest group, 0 if everything is segmented.
        """
        # Detect largest groups
        largest_group = self.detect_largest_group()
                # Check if the largest group has enough members
        if len(largest_group) < self.n_min_group_size:
            print("the number of members in the largest group: "+str(len(largest_group)))
            return 0, 0
        else:
            # Extract the fish that belong to the largest group
            largest_group_fish = [fish for fish in self.fish_list if fish.ID in largest_group]

            # Calculate mean directionality
            directions = np.array([fish.direction for fish in largest_group_fish])
            mean_direction = np.mean(directions, axis=0)
            P = np.linalg.norm(mean_direction)  # Magnitude of the mean direction vector
            #print("hi!")
            unwrapped_positions,unwrapped_directions = self.group_detector.detect_groups_v2(largest_group_fish)
            unwrapped_positions = np.array(list(unwrapped_positions.values()))
            unwrapped_directions = np.array(list(unwrapped_directions.values()))
            # print("unwrapped_positions: "+str(unwrapped_positions))
            # print("unwrapped_directions: "+str(unwrapped_directions))
            normalized_m = self.calculate_angular_momentum(unwrapped_positions,unwrapped_directions)

            # center_of_group = np.mean(positions, axis=0)
            # relative_positions = positions - center_of_group
            # cross_products = np.cross(relative_positions, directions)
            # mean_angular_momentum = np.mean(cross_products, axis=0)
            # m = np.linalg.norm(mean_angular_momentum)  # Magnitude of the mean angular momentum vector
            # average_distance = np.mean(np.linalg.norm(relative_positions, axis=1))
            # normalized_m = m / average_distance if average_distance > 0 else m
            # for dim in range(3):
            #     for i,box_choice in enumerate([-self.box_lengths[dim],0,self.box_lengths[dim]]):
            #         if not self.reflect_at_boundary[dim]:  #
            #             unwrapped_positions = np.array(positions)
            #             unwrapped_positions[:,dim] += box_choice
            #             norm_m = self.calculate_angular_momentum(unwrapped_positions,directions)
            #             if norm_m > norm_highest:
            #                 norm_highest = norm_m
            #             unwrapped_positions[:,dim] -= box_choice

                

        
        return P, normalized_m
    
    def compute_group_density(self,positions):
        """
        Compute group density in the Couzin model.
        :param positions: List of (x, y, z) positions of agents in the group.
        :return: Density of the group.
        """
        if len(positions) < 4:  # Not enough points for a convex hull in 3D
            return float('inf')  # Group is too small for a meaningful density

        # Compute the convex hull
        hull = ConvexHull(positions)
        volume = hull.volume  # Use hull.area for 2D

        # Calculate density
        num_agents = len(positions)
        density = num_agents / volume
        return density
        

# Usage Example
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


    # Set the minimum group size required to compute metrics
    n_min_group_size = 5

    # Create an instance of GroupAnalyzer
    analyzer = GroupAnalyzer(fish_list, verlet_list, n_min_group_size)

    # Compute the metrics
    mean_directionality, mean_angular_momentum = analyzer.compute_group_metrics()

    # Print the results
    print("Mean Directionality (P):", mean_directionality)
    print("Mean Angular Momentum (m):", mean_angular_momentum)
    
  # Define parameters for the simulation
    num_fish = 20
    radius = 10.0
    angle_increment = 2 * np.pi / num_fish
    box_lengths = [50.0, 50.0, 50.0]
    reflect_at_boundary = [True, True, True]

    # Generate fish positioned in a perfect circle with directions tangential to the circle
    fish_list = []
    for i in range(num_fish):
        angle = i * angle_increment
        position = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0]) + box_lengths[0] / 2.0
        direction = np.array([-np.sin(angle), np.cos(angle), 0.0])  # Tangential direction
        fish_list.append(Fish(position=position, direction=direction, ID=i))

    # Initialize Verlet list
    verlet_list = VerletList(
        fish_list=fish_list,
        repulsion_radius=1.0,
        orientation_width=10.0,
        attraction_width=10.0,
        box_lengths=box_lengths,
        reflect_at_boundary=reflect_at_boundary
    )

    # Set the minimum group size required to compute metrics
    n_min_group_size = 5

    # Create an instance of GroupAnalyzer
    analyzer = GroupAnalyzer(fish_list, verlet_list, n_min_group_size)
    # Compute the metrics
    mean_directionality, mean_angular_momentum = analyzer.compute_group_metrics()

    # Print the results
    print("Mean Directionality (P):", mean_directionality)
    print("Normalized Mean Angular Momentum (m):", mean_angular_momentum)