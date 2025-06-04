"""
Object module
=============

Contains the `Fish` class and in the future some other objects maybe.
"""

import numpy as np

from couzinswarm.tools import rotate_towards #, cart2sphere, sphere2cart

class Fish:
    """A class containing information about a single fish.
    
    A single fish is characterized by its position and direction.
    It can be influenced by other fish nearby.

    Attributes
    ----------
        position : numpy.ndarray
            3-dimensional vector containing the fish's current position
        direction : numpy.ndarray
            3-dimensional unit vector giving the current direction of the fish
        d_r : numpy.ndarray
            3-dimensional vector keeping track of directional influence
            within the repulsion zone
        n_r : int
            Counter keeping track of the number of fish in the repulsion zone
        d_o : numpy.ndarray
            3-dimensional vector keeping track of directional influence
            within the orientation zone
        n_o : int
            Counter keeping track of the number of fish in the orientation zone
        d_a : numpy.ndarray
            3-dimensional vector keeping track of directional influence
            within the attraction zone
        n_a : int
            Counter keeping track of the number of fish in the attraction zone
    """

    def __init__(self,position,direction=None,ID=None,verbose=False):
        """
        Initiate a Fish object

        Parameters
        ----------
        position : numpy.ndarray
            3-dimensional vector
        direction : numpy.ndarray, default : None
            3-dimensional unit vector giving the direction of the fish
            If `None`, will be randomly sampled from the uniform sphere
        ID : any type, default : None
            A fish identifier
        verbose : bool, default : False
            be chatty
        """

        self.position = position
        if direction is None:
            self.direction = np.random.randn(3)
            self.direction /= np.linalg.norm(self.direction)
        else:
            self.direction = direction / np.linalg.norm(direction)

        self.ID = ID
        self.verbose = verbose
        self.reset_direction_influences()

    def reset_direction_influences(self):
        """
        Reset all direction influences collected in this time step.
        """

        self.d_r = np.zeros(3,dtype=float)
        self.d_o = np.zeros(3,dtype=float)
        self.d_a = np.zeros(3,dtype=float)
        self.n_r = 0
        self.n_a = 0
        self.n_o = 0

    def zor_update(self,r_ij):
        """
        Add influence of a fish in repulsion zone.

        Parameters
        ----------
        r_ij : numpy.ndarray
            unit vector pointing to the other fish
        """

        self.d_r = self.d_r - r_ij
        self.n_r += 1

    def zoo_update(self,v_j):
        """
        Add influence of a fish in orientation zone.

        Parameters
        ----------
        v_j : numpy.ndarray
            Unit direction vector of the other fish
        """

        self.d_o = self.d_o + v_j
        self.n_o += 1

    def zoa_update(self,r_ij):
        """
        Add influence of a fish in attraction zone.

        Parameters
        ----------
        r_ij : numpy.ndarray
            unit vector pointing to the other fish
        """

        self.d_a += r_ij
        self.n_a += 1

    def evaluate_direction(self,thetatau,sigma):
        """
        Decide on the new direction according to the rules
        stated in the paper and add noise.
        Returns unit vector of demanded new direction.

        Parameters
        ----------
        thetatau : float
            maximally allowed angle to rotate by per time step
        sigma : float
            standard deviation of the noise to be added to
            the evaluated new direction

        Returns
        -------
        new_d : numpy.ndarray
            unit vector of the evaluated new direction
        """

        #no_new_d = False

        if self.n_r > 0:
            new_d = self.d_r
        elif self.n_o > 0 and self.n_a > 0:
            new_d = 0.5*(self.d_o+self.d_a)
        elif self.n_o > 0:
            new_d = self.d_o
        elif self.n_a > 0:
            new_d = self.d_a
        else:
            new_d = self.direction
        #print("new direction: "+str(new_d))
        if self.verbose:
            print("Fish", self.ID)
            print("    direction:", self.direction)
            print("    repulsion:", self.n_r, self.d_r)
            print("    orientation:", self.n_o, self.d_o)
            print("    attraction:", self.n_a, self.d_a)
            print("    new_d:",new_d)

        #print("new direction after conversion: "+str(new_d))
        norm = np.linalg.norm(new_d)
        #print("norm: "+str(norm))
        if norm > 1e-5:  # Check for non-zero norm to avoid division by zero
            self.new_d = new_d / norm
            # get spherical coordinates of directions and add some noise to the angles
            # _rho,_theta, _phi = cart2sphere(self.new_d)
            # _theta += sigma * np.random.randn()
            # _phi += sigma * np.random.randn()
            # self.new_d = sphere2cart(1,_theta, _phi)
            # #print("after normalization: "+str(self.new_d))
        else:
            self.new_d = self.direction  # Fallback to the current direction if zero vector
        # Convert velocity to spherical coordinates
        #speed = np.linalg.norm(velocity)
        phi = np.arccos(self.new_d[2])  # Polar angle
        theta = np.arctan2(self.new_d[1], self.new_d[0])  # Azimuthal angle

        # Apply noise
        phi_noisy = phi + sigma * np.random.randn()
        theta_noisy = theta + sigma * np.random.randn()

        # Convert back to Cartesian coordinates
        self.new_d = np.array([
             np.sin(phi_noisy) * np.cos(theta_noisy),
             np.sin(phi_noisy) * np.sin(theta_noisy),
             np.cos(phi_noisy)
        ])

        # self.new_d += sigma*np.random.randn(3)
        # norm = np.linalg.norm(self.new_d)
        # self.new_d = np.array(self.new_d) / norm
        # if the angle between old and new directions is larger than allowed
        # per step size, rotate the current direction towards the new direction by
        # the maximum radians per step size
        angle = np.arccos(np.clip(np.dot(self.new_d, self.direction), -1.0, 1.0))
        #print("angle: "+str(angle))
        #print("turning rate: "+str(thetatau))
        if angle > thetatau:
            self.new_d = rotate_towards(self.direction, self.new_d, thetatau)
            #print("restricted new d: "+str(self.new_d))
        if self.verbose:
            print("    after noise and rotation:",new_d)

        self.reset_direction_influences()
        #print("new direction in the end: "+str(self.new_d))
        norm = np.linalg.norm(self.new_d)
        if np.abs(norm - 1) > 1e-5:
            print("self.new_d is not properly normalized!")
            print("new_d = "+str(self.new_d))
        return self.new_d
