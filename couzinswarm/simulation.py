"""
Simulation module
=================

Contains the `Swarm` class, which is used for simulation.
"""
import numpy as np
import sys
sys.path.append("/home/ggu7596/ML_collective_behaviors/softwares/couzinswarm")
from couzinswarm.objects import Fish
from couzinswarm.verletlist import VerletList
from couzinswarm.groupanalyzer import GroupAnalyzer
from progressbar import ProgressBar as PB

import sys

class Swarm:
    """
    A class for a swarm simulation.
    
    Attributes
    -----------
    number_of_fish : int, default : 20
        The number of fish to be simulated
    fish : list of :mod:`couzinswarm.objects.Fish`
        Contains the `Fish` objects which are simulated in this setup.
    repulsion_radius : float, default : 1.0
        Fish within this radius will repel each other
        (unit: length of a single fish).
    orientation_width : float, default : 10.0
        The width of the hollow ball in which fish adjust their
        orientation.
        (unit: length of a single fish).
    attraction_width : float, default : 10.0
        The width of the hollow ball in which fish attract
        each other
        (unit: length of a single fish).
    angle_of_perception : float, default : 340/360*pi
        angle in which a fish can see other fish
        (unit: radians, with a maximum value of :math:`pi`.
    turning_rate : float, default : 0.1
        Rate at which the new direction is approached.
        The maximum angle change per time step is hence ``turning_rate * dt``
        (unit: radians per unit time).
    speed : float, default : 0.1
        Speed of a fish.
        (unit: fish length per unit time).
    noise_sigma : float, default : 0.01
        Standard deviation of radial noise whith 
        which each direction adjustment is shifted
        (unit: radians).
    dt : float, default : 0.1
        how much time passes per step
        (unit: unit time).
    box_lengths : list or numpy.ndarray of float, default : [100,100,100]
        Dimensions of the simulation box in each dimension
        (unit: fish length)
    reflect_at_boundary list of bool, default : [True, True, True]
        for each spatial dimension decided whether boundaries should reflect.
        If they don't reflect they're considered to be periodic (not implemented yet)
    verbose : bool, default : False
        be chatty.
    show_progress : bool, default : False
        Show the progress of the simulation.

    """

    def __init__(self, 
                 number_of_fish=20,
                 repulsion_radius=1, 
                 orientation_width=10,
                 attraction_width=10,
                 angle_of_perception=340/360*np.pi, 
                 turning_rate=0.1,
                 speed=0.1,
                 noise_sigma=0.01,
                 dt=0.1,
                 box_lengths=[100,100,100],
                 reflect_at_boundary = [True, True, True],
                 verbose=False,
                 show_progress=False,
                 initialization_radii = 15,
                 using_verletlist = True,
                 seed=None,  # New parameter for the seed
                 n_min_group_size = 10
                 ):
        """
        Setup a simulation with parameters as defined in the paper.
        https://www.sciencedirect.com/science/article/pii/S0022519302930651

        Fish will be created at random positions with random directions.

        Parameters
        ----------
        number_of_fish : int, default : 20
            The number of fish to be simulated
        repulsion_radius : float, default : 1.0
            Fish within this radius will repel each other
            (unit: length of a single fish).
        orientation_width : float, default : 10.0
            The width of the hollow ball in which fish adjust their
            orientation.
            (unit: length of a single fish).
        attraction_width : float, default : 10.0
            The width of the hollow ball in which fish attract
            each other
            (unit: length of a single fish).
        angle_of_perception : float, default : 340/360*pi
            angle in which a fish can see other fish
            (unit: radians, with a maximum value of :math:`pi`.
        turning_rate : float, default : 0.1
            Rate at which the new direction is approached.
            The maximum angle change per time step is hence ``turning_rate * dt``
            (unit: radians per unit time).
        speed : float, default : 0.1
            Speed of a fish.
            (unit: fish length per unit time).
        noise_sigma : float, default : 0.01
            Standard deviation of radial noise whith 
            which each direction adjustment is shifted
            (unit: radians).
        dt : float, default : 0.1
            how much time passes per step
            (unit: unit time).
        box_lengths : list or numpy.ndarray of float, default : [100,100,100]
            Dimensions of the simulation box in each dimension
            (unit: fish length)
        reflect_at_boundary list of bool, default : [True, True, True]
            for each spatial dimension decided whether boundaries should reflect.
            If they don't reflect they're considered to be periodic
        verbose : bool, default : False
            be chatty.

        """
        

        self.number_of_fish = number_of_fish
        self.repulsion_radius = repulsion_radius
        self.orientation_width = orientation_width
        self.attraction_width = attraction_width
        self.angle_of_perception = angle_of_perception
        self.turning_rate = turning_rate
        self.speed = speed
        self.noise_sigma = noise_sigma
        self.dt = dt
        self.box_lengths = np.array(box_lengths,dtype=float)
        self.reflect_at_boundary = reflect_at_boundary
        self.verbose = verbose
        self.show_progress = show_progress
        self.using_verletlist = using_verletlist
        self.seed=seed,  # New parameter for the seed
        self.box_copies = [[0.],[0.],[0.]]
        self.initialization_radii = initialization_radii
        self.n_min_group_size = n_min_group_size
        # for dim, reflect in enumerate(self.reflect_at_boundary):
        #     if not reflect:
        #         self.box_copies[dim].extend([-self.box_lengths[dim],+self.box_lengths[dim]])


        self.fish = []

        self.init_random()
        if self.using_verletlist == True:
                # Initialize Verlet list
            self.verlet_list = VerletList(
                fish_list=self.fish,
                repulsion_radius=self.repulsion_radius,
                orientation_width=self.orientation_width,
                attraction_width=self.attraction_width,
                box_lengths=self.box_lengths,
                reflect_at_boundary=self.reflect_at_boundary
            )
    
    # boundary condition, x is the position along dim 
    def boundary(self,x,box_length):
        if x < 0:
            x += box_length
        elif x >= box_length:
            x -= box_length
        return x
    def boundary_3D(self,x):
        if len(x) == 3:
            for dim, reflect in enumerate(self.reflect_at_boundary):
                if not reflect:
                    x[dim] = self.boundary(x[dim],self.box_lengths[dim])
        else:
            print("x's dimension is wrong!")
            sys.exit()
        return x
    # periodic boundary condition
    # compute the minimal distance between the two particles, dx is in one dim
    def pbc(self,dx,box_length):
        if dx >= box_length/2:
            dx -= box_length
        elif dx < -box_length/2:
            dx += box_length
        return dx
    def pbc_3D(self,dx):
        if len(dx) == 3:
            for dim, reflect in enumerate(self.reflect_at_boundary):
                if not reflect:
                    dx[dim] = self.pbc(dx[dim],self.box_lengths[dim])
        else:
            print("x's dimension is wrong!")
            sys.exit()
        return dx
        
    # def init_random(self):
    #     """
    #     Initialize the fish list
    #     """
    #     adjusted_length = (self.number_of_fish)**(1/3)/self.initialization_density
    #     if adjusted_length > self.box_lengths[0]:
    #         adjusted_length = self.box_lengths[0]
    #     print("adjusted length = "+str(adjusted_length))
    #     # self.fish = [ Fish(position=self.box_lengths*np.random.random((3,)),
    #     #                    ID=i,
    #     #                    verbose=self.verbose
    #     #                    ) for i in range(self.number_of_fish) ]
    #     # Set the random seed if provided
    #     if isinstance(self.seed, int):
    #         np.random.seed(self.seed)


    #     # Initialize fish
    #     self.fish = [ Fish(position=adjusted_length*np.random.random((3,)) + 0.5* (self.box_lengths - adjusted_length),
    #                 direction=np.random.randn(3),
    #                 ID=i,
    #                 verbose=self.verbose
    #                 ) for i in range(self.number_of_fish) ]
    def init_random(self):
        """
        Initialize the fish list randomly within a ball of given radius.
        """
        num_fish = self.number_of_fish
        self.initialization_radii = 15.0  # Define the radius of the ball
        box_lengths = self.box_lengths
        fish_list = []

        # Set the random seed if provided
        if isinstance(self.seed, int):
            np.random.seed(self.seed)

        for i in range(num_fish):
            # Generate a random point within a ball of radius `radius`
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)  # Normalize to get a direction vector
            distance = self.initialization_radii * (np.random.random() ** (1/3))  # Random distance within the ball
            position = direction * distance + box_lengths / 2.0  # Center it in the box

            # Assign a random direction for the fish
            fish_direction = np.random.randn(3)
            fish_direction /= np.linalg.norm(fish_direction)

            # Create the fish and add it to the list
            fish_list.append(Fish(position=position, direction=fish_direction, ID=i))

        self.fish = fish_list

    def simulate(self, N_time_steps):
            """Choose which simulation method to run based on the using_verletlist flag."""
            if self.using_verletlist:
                return self.simulate_with_verlet(N_time_steps)
            else:
                return self.simulate_without_verlet(N_time_steps)

    def simulate_with_verlet(self, N_time_steps):
        """Simulation using Verlet list for efficient neighbor detection."""
        positions = np.empty((self.number_of_fish, N_time_steps + 1, 3))
        directions = np.empty((self.number_of_fish, N_time_steps + 1, 3))
        mean_P_array= []
        mean_m_array=[]
        for i in range(self.number_of_fish):
            positions[i, 0, :] = self.fish[i].position
            directions[i, 0, :] = self.fish[i].direction

        #bar = PB(max_value=N_time_steps)

        for t in range(1, N_time_steps + 1):
            if self.verlet_list.needs_update():
                self.verlet_list.update_list()

            for fish in self.fish:
                #"+str(fish.ID))
                neighbors = self.verlet_list.get_neighbors(fish)
                #print("neighbors: ")
                for neighbor in neighbors:
                    #print(neighbor.ID)
                    r_ij = self.verlet_list.handle_boundaries(fish.position, neighbor.position)
                    distance = np.linalg.norm(r_ij)
                    if distance > 1e-5:
                        r_ij /= distance
                    else:
                        r_ij = np.zeros_like(r_ij)
                    r_ji = -r_ij
                    if distance < self.repulsion_radius:
                        fish.zor_update(r_ij)
                        neighbor.zor_update(r_ji)
                    elif distance < self.repulsion_radius + self.orientation_width + self.attraction_width:
                        angle = np.arccos(np.clip(np.dot(r_ij, fish.direction), -1.0, 1.0))
                        angle_j = np.arccos(np.clip(np.dot(r_ji, neighbor.direction), -1.0, 1.0))
                        #print("angle: "+str(angle))
                        #print("angle_j: "+str(angle_j))
                        if angle < self.angle_of_perception:
                            if distance < self.repulsion_radius + self.orientation_width:
                                fish.zoo_update(neighbor.direction)
                            else:
                                fish.zoa_update(r_ij)
                        if angle_j < self.angle_of_perception:
                            if distance < self.repulsion_radius + self.orientation_width:
                                neighbor.zoo_update(fish.direction)
                            else:
                                neighbor.zoa_update(r_ji)
            # for each fish
            for i in range(self.number_of_fish):
                F_i = self.fish[i]

                # evaluate the new demanded direction and reset the influence counters
                new_v = F_i.evaluate_direction(self.turning_rate*self.dt,self.noise_sigma)

                # evaluate the demanded positional change according to the direction
                dr = self.speed * new_v * self.dt

                # check for boundary conditions
                for dim in range(3):

                    # if new position would be out of boundaries
                    if dr[dim]+F_i.position[dim] >= self.box_lengths[dim] or \
                       dr[dim]+F_i.position[dim] < 0.0:

                        # if this boundary is periodic
                        if not self.reflect_at_boundary[dim]:
                            if dr[dim]+F_i.position[dim] >= self.box_lengths[dim]:
                                dr[dim] -= self.box_lengths[dim]
                            else:
                                dr[dim] += self.box_lengths[dim]
                        else:
                            # if this boundary is reflective
                            #dr[dim] *= -1
                            new_v[dim] *= -1
                            if dr[dim] + F_i.position[dim] >= self.box_lengths[dim]:
                                dr[dim] = -abs(dr[dim]) * 0.95
                            elif dr[dim] + F_i.position[dim] <= 0.0:
                                dr[dim] = abs(dr[dim]) * 0.95
                # update the position and direction
                F_i.position += dr
                F_i.direction = new_v

                # save position and direction
                positions[i,t,:] = F_i.position
                directions[i,t,:] = F_i.direction

            #bar.update(t)
            
            if (np.int32((t-1)%100) == 0):
                mean_directionality, mean_angular_momentum = self.detect_behavior()
                mean_P_array.append(mean_directionality)
                mean_m_array.append(mean_angular_momentum)
                print("Mean Directionality (P):", mean_directionality)
                print("Mean Angular Momentum (m):", mean_angular_momentum)
        mean_P_array = np.array(mean_P_array)
        mean_m_array = np.array(mean_m_array)
        return positions, directions, mean_P_array, mean_m_array
    def simulate_without_verlet(self,N_time_steps):
        """Simulate a swarm according to the rules.

        Parameters
        ----------
        N_time_steps : int
            Number of time steps to simulate.

        Returns
        -------
        positions : numpy.ndarray of shape ``(self.number_of_fish, N_time_steps+1, 3_)``
            Keeping track of the fish's positions for each time step.
        directions : numpy.ndarray of shape ``(self.number_of_fish, N_time_steps+1, 3_)``
            Keeping track of the fish's directions for each time step.
        """


        # create result arrays and fill in initial positions
        positions = np.empty((self.number_of_fish,N_time_steps+1,3))
        directions = np.empty((self.number_of_fish,N_time_steps+1,3))
        mean_P_array= []
        mean_m_array=[]
        for i in range(self.number_of_fish):
            positions[i,0,:] = self.fish[i].position
            directions[i,0,:] = self.fish[i].direction
        

        #bar = PB(max_value=N_time_steps)
        # for each time step
        for t in range(1,N_time_steps+1):
            # iterate through fish pairs
            for i in range(self.number_of_fish-1):
                F_i = self.fish[i]
                r_i = F_i.position
                v_i = F_i.direction

                for j in range(i+1,self.number_of_fish):

                    F_j = self.fish[j]
                    r_j = F_j.position
                    v_j = F_j.direction

                    # get their distance, and unit distance vector
                    r_ij = (r_j - r_i)
                    distance = np.linalg.norm(r_ij)
                    if distance > 1e-6:
                        r_ij /= distance
                    else:
                        r_ij = np.zeros_like(r_ij)
                    r_ij = self.pbc_3D(r_ij)
                    r_ji = -r_ij

                    # if their are within the repulsion zone, just add each other to
                    # the repulsion events
                    if distance < self.repulsion_radius:
                        F_i.zor_update(r_ij)
                        F_j.zor_update(r_ji)
                        #relationship_counted = True
                    elif distance < self.repulsion_radius + self.orientation_width + self.attraction_width:

                        # if they are within the hollow balls of orientation and attraction zone, 
                        # decide whether the fish can see each other
                        angle_i = np.arccos(np.clip(np.dot(r_ij, v_i), -1.0, 1.0))
                        angle_j = np.arccos(np.clip(np.dot(r_ji, v_j), -1.0, 1.0))

                        if self.verbose:
                            print("angle_i", angle_i, self.angle_of_perception)
                            print("angle_j", angle_j, self.angle_of_perception)

                        # if i can see j, add j's influence
                        if angle_i < self.angle_of_perception:
                            if distance < self.repulsion_radius + self.orientation_width:
                                F_i.zoo_update(v_j)
                            else:
                                F_i.zoa_update(r_ij)

                        # if j can see i, add i's influence
                        if angle_j < self.angle_of_perception:
                            if distance < self.repulsion_radius + self.orientation_width:
                                F_j.zoo_update(v_i)
                            else:
                                F_j.zoa_update(r_ji)

                        #relationship_counted = True

            # for each fish
            for i in range(self.number_of_fish):

                F_i = self.fish[i]

                # evaluate the new demanded direction and reset the influence counters
                new_v = F_i.evaluate_direction(self.turning_rate*self.dt,self.noise_sigma)

                # evaluate the demanded positional change according to the direction
                dr = self.speed * new_v * self.dt

                # check for boundary conditions
                for dim in range(3):

                    # if new position would be out of boundaries
                    if dr[dim]+F_i.position[dim] >= self.box_lengths[dim] or \
                       dr[dim]+F_i.position[dim] < 0.0:

                        # if this boundary is periodic
                        if not self.reflect_at_boundary[dim]:
                            if dr[dim]+F_i.position[dim] >= self.box_lengths[dim]:
                                dr[dim] -= self.box_lengths[dim]
                            else:
                                dr[dim] += self.box_lengths[dim]
                        else:
                            # if this boundary is reflective
                            #dr[dim] *= -1
                            new_v[dim] *= -1
                            if dr[dim] + F_i.position[dim] >= self.box_lengths[dim]:
                                dr[dim] = -abs(dr[dim]) * 0.95
                            elif dr[dim] + F_i.position[dim] <= 0.0:
                                dr[dim] = abs(dr[dim]) * 0.95
                # update the position and direction
                F_i.position += dr
                F_i.direction = new_v

                # save position and direction
                positions[i,t,:] = F_i.position
                directions[i,t,:] = F_i.direction

            #bar.update(t)
            if (np.int32((t-1)%100) == 0):
                mean_directionality, mean_angular_momentum = self.detect_behavior()
                mean_P_array.append(mean_directionality)
                mean_m_array.append(mean_angular_momentum)
                print("Mean Directionality (P):", mean_directionality)
                print("Mean Angular Momentum (m):", mean_angular_momentum)
        mean_P_array = np.array(mean_P_array)
        mean_m_array = np.array(mean_m_array)
        return positions, directions,mean_P_array,mean_m_array

                ### analyze the results
    def detect_behavior(self):

        # Create an instance of GroupAnalyzer
        if self.using_verletlist == False:
            self.verlet_list = VerletList(
                fish_list=self.fish,
                repulsion_radius=self.repulsion_radius,
                orientation_width=self.orientation_width,
                attraction_width=self.attraction_width,
                box_lengths=self.box_lengths,
                reflect_at_boundary=self.reflect_at_boundary
            )
        analyzer = GroupAnalyzer(self.fish, self.verlet_list, self.n_min_group_size,self.box_lengths,self.reflect_at_boundary)
        size = len(analyzer.detect_largest_group())
        print("group size: "+str(size))
        # Compute the metrics
        mean_directionality, mean_angular_momentum = analyzer.compute_group_metrics()
        return mean_directionality, mean_angular_momentum 

if __name__ == "__main__":

    swarm = Swarm(number_of_fish=2,speed=0.01,noise_sigma=0,turning_rate=0.1)

    swarm.fish[0].position = np.array([47,50.,50.])
    swarm.fish[0].direction = np.array([0.,0.,1.])
    swarm.fish[1].position = np.array([58.,50.,50])
    swarm.fish[1].direction = np.array([1.,0.,0.])



    N_t = 100

    t = np.arange(N_t+1)
    r, v = swarm.simulate(N_t)
    print(
        swarm.fish[0].direction,
        swarm.fish[1].direction,
    )
    mean_directionality, mean_angular_momentum  = swarm.detect_behavior()
    # Print the results
    print("Mean Directionality (P):", mean_directionality)
    print("Mean Angular Momentum (m):", mean_angular_momentum)


