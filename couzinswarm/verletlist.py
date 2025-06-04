import numpy as np
from collections import defaultdict
import sys
sys.path.append("/home/ggu7596/ML_collective_behaviors/softwares/couzinswarm")
from couzinswarm.objects import Fish

class VerletList:
    """
    A Verlet list class for fixed-size simulations with support for both periodic and reflective boundaries.
    """
    def __init__(self, fish_list, 
                 repulsion_radius=1.0, 
                 orientation_width=10.0,
                 attraction_width=10.0,
                 box_lengths=None, 
                 reflect_at_boundary=None):
        self.fish_list = fish_list
        self.repulsion_radius = repulsion_radius
        self.orientation_width = orientation_width
        self.attraction_width = attraction_width
        self.buffer = np.sum([attraction_width,repulsion_radius,orientation_width])/5
        self.box_lengths = np.array(box_lengths)
        self.reflect_at_boundary = reflect_at_boundary
        
        # Calculate the interaction radius
        self.interaction_radius = repulsion_radius + orientation_width + attraction_width + self.buffer
        
        self.neighbor_list = defaultdict(list)
        self.positions_last_updated = [fish.position.copy() for fish in fish_list]
        self.update_list()

    def handle_boundaries(self, r_i, r_j):
        """Adjust distance vector according to the boundary conditions."""
        dx = r_j - r_i
        for dim in range(3):
            if not self.reflect_at_boundary[dim]:  # Periodic boundary
                if dx[dim] >= self.box_lengths[dim] / 2:
                    dx[dim] -= self.box_lengths[dim]
                elif dx[dim] < -self.box_lengths[dim] / 2:
                    dx[dim] += self.box_lengths[dim]
            # else:  # Reflective boundary
            #     if r_j[dim] > self.box_lengths[dim]:
            #         r_j[dim] = 2 * self.box_lengths[dim] - r_j[dim]
            #     elif r_j[dim] < 0:
            #         r_j[dim] = -r_j[dim]
        return dx

    def update_list(self):
        """Build or update the Verlet list with optimized neighbor detection."""
        self.neighbor_list.clear()
        num_fish = len(self.fish_list)

        for i in range(num_fish):
            fish_i = self.fish_list[i]
            for j in range(i + 1, num_fish):
                fish_j = self.fish_list[j]

                # Compute the distance between fish_i and fish_j considering boundaries
                r_ij = self.handle_boundaries(fish_i.position, fish_j.position)
                distance = np.linalg.norm(r_ij)
                #print("distance: "+str(distance))
                # Check if they are within the interaction radius
                if distance < self.interaction_radius:
                    # Add each other as neighbors
                    self.neighbor_list[fish_i.ID].append(fish_j)
                    #self.neighbor_list[fish_j.ID].append(fish_i)

        # Update positions for checking displacement in needs_update()
        self.positions_last_updated = [fish.position.copy() for fish in self.fish_list]
        
    def needs_update(self):
        """Check if any fish moved beyond the buffer distance, requiring an update."""
        for i, fish in enumerate(self.fish_list):
            displacement = self.handle_boundaries(self.positions_last_updated[i], fish.position)
            distance = np.linalg.norm(displacement)
            if distance > self.buffer:
                return True
        return False

    def get_neighbors(self, fish):
        """Get the list of neighbors for a given fish."""
        return self.neighbor_list.get(fish.ID, [])

# Example usage and test case
if __name__ == "__main__":
    # Define parameters for the simulation
    num_fish = 5
    repulsion_radius = 1.0
    orientation_width = 5.0
    attraction_width = 5.0
    box_lengths = [20.0, 20.0, 20.0]
    reflect_at_boundary = [True,True,True]  # Periodic boundaries

    # Create random Fish objects
    fish_list = [Fish(position=np.random.rand(3) * box_lengths, ID=i) for i in range(num_fish)]
    for i,fish in enumerate(fish_list):
        print("Fish "+str(i)+" position: "+str(fish.position))
    # Initialize the VerletList
    verlet_list = VerletList(
        fish_list=fish_list,
        repulsion_radius=repulsion_radius,
        orientation_width=orientation_width,
        attraction_width=attraction_width,
        box_lengths=box_lengths,
        reflect_at_boundary=reflect_at_boundary
    )

    # Print the initial neighbors for each fish
    print("Initial neighbors:")
    for fish in fish_list:
        neighbors = verlet_list.get_neighbors(fish)
        neighbor_ids = [neighbor.ID for neighbor in neighbors]
        print(f"Fish {fish.ID} neighbors: {neighbor_ids}")

    # Move one fish beyond the buffer distance and check if an update is needed
    print("\nMoving Fish 0...")
    fish_list[0].position += np.array([10.0, 0.0, 0.0])  # Move fish 0
    if verlet_list.needs_update():
        print("Verlet list needs an update after moving Fish 0.")
        verlet_list.update_list()
    
    # Print the updated neighbors after moving a fish
    print("\nUpdated neighbors after moving Fish 0:")
    for i,fish in enumerate(fish_list):
        print("Fish "+str(i)+" position: "+str(fish.position))
    for fish in fish_list:
        neighbors = verlet_list.get_neighbors(fish)
        neighbor_ids = [neighbor.ID for neighbor in neighbors]
        print(f"Fish {fish.ID} neighbors: {neighbor_ids}")
