import numpy as np
from geometry import geographic_to_cartesian, cartesian_to_geographic, even_sphere_points
import matplotlib.pyplot as plt
import quads
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import time


# Move the plates
# Not moving the points but the information from point to point
"""
Data Structures
- Point ((n*m) X 3)
    Stores the height, velocity and plate data 

Algorithm
An optimisation would be to only run the ones that have a difference in value
1) Calculate points new position.
2) Move the plate information to the new position (on a different matrix)
3) 
"""


# If plates overlap then form a mountain by adding edges to one of the plates 
# If no points at a spot get added to the nearest plate at ocean level
class Stationary_Grid:
    def __init__(self, grid, grid_tree, height, velocity, plate_type, resolution) -> None:
        self.grid = grid
        self.grid_tree = grid_tree
        self.height = height
        self.size = len(height)
        self.velocity = velocity
        self.plate_type = plate_type
        self.resolution = resolution
        self.resolution_step = (2*np.pi/resolution[1], np.pi/resolution[1])
        self.volcanism = np.zeros(self.size)
        self.temperature = np.zeros(self.size)
        self.humidity = np.zeros(self.size)
        self.water_level = np.zeros(self.size)
        self.sea_level = 0.5

    
    
    def move_plates(self):
        # There is still an artifact on the center verticle line
        next_step_height = np.zeros(self.size)
        next_step_velocity = self.velocity.copy()
        next_plate_step = -np.ones(self.size)
        for i in range(self.size):
            point = np.array([self.velocity[i].apply(self.grid[i])])
            _, new_index = self.grid_tree.query(point)
            next_step_height[new_index[0]] += self.height[i]
            if next_plate_step[new_index[0]] < self.plate_type[i]: #check if it is the dominant plate
                #transferring information
                next_step_velocity[new_index[0]] = self.velocity[i] 
                next_plate_step[new_index[0]] = self.plate_type[i] 
            elif self.plate_type[i] < 0:
                self.volcanism[new_index[0]] += 0.1 # to allow for simulated volcanism creating mountains and island chains
                next_step_height[i] -= self.height[i] # to try and simulate trenches
        # averaging the poles
        #south_pole = np.sum(next_step_height[:self.resolution[0]])/self.resolution[0]
        #north_pole = np.sum(next_step_height[-self.resolution[0]:])/self.resolution[0]
        #next_step_height[:self.resolution[0]] = np.ones(self.resolution[0])*south_pole
        #next_step_height[-self.resolution[0]:] = np.ones(self.resolution[0])*north_pole
        self.height = next_step_height
        self.velocity = next_step_velocity
        self.plate_type = next_plate_step
        self.plate_type += 0.01
            #select the grid point
            #compute new position
            #take that grid point and compute the index
            #move the information at index i to the new index
        #points_to_check = (self.height - self.previous_step) != 0
    
    def weather(self):
        self.humidity += np.minimum(100, np.maximum(0, self.temperature/100))*(self.height <= self.sea_level)
        wind_matrix = Rotation.from_rotvec(np.random.random(3))
        new_locations = wind_matrix.apply(self.grid)

        # try a global direction vector
        # if move downhill temperature increases
        # if move uphill temperature decreases
        # defuse temperature and humidity
        # if humidity exceeds carrying capacity precipitate
        pass

    def temperature_update(self):
        """
        Input:
        Time of year
        Planet Angle
        Planet Distance

        """
        # use climate model to generate a temperature profile

        # This is impacted by clouds and latitude
        pass

    def erosion(self):
        # to spread out the height map
        pass

    def eruptions(self):
        # Processing the volcanic activity
        pass

    def evolve(self):
        self.move_plates()
        self.eruptions()
        self.erosion()

    def display(self):
        #display a height map using a contour map

        xx, yy = np.meshgrid(np.linspace(0, 2*np.pi, self.resolution[0]), np.linspace(-np.pi/2, np.pi/2, self.resolution[1]))
        _, indices = self.grid_tree.query(geographic_to_cartesian(np.array([xx.flatten(), yy.flatten()]).transpose()))
        height_map = self.height[indices].reshape(self.resolution)
        print(np.min(height_map), np.max(height_map))
        plt.contourf(xx, yy, height_map)
        plt.show()
        

class Tectonics:
    def __init__(self, point_num):
        # need to add shape
        self.grid_value = None
        self.tree = None
        self.grid_coords = None
        self.points = point_num
        self.resolution = (100, 100)
        self.detail = 10000

    def gen_vor_tes(self):
        """
        Generates points and a grid assigning each grid point to a plate.
        :return:
        """
        #generate points in the domain (-1, 1)^3
        start = time.time()
        point_3d = np.random.random((self.points, 3))*2 - 1
        #generate plate information
        #store these points in a lookup tree
        self.tree = KDTree(point_3d)
        #generate the grid of points
        grid_coord_3d = even_sphere_points(self.detail)
        #assigning each grid point a
        self.grid_value = self.tree.query(grid_coord_3d)[1]
        self.grid_coord_3d = grid_coord_3d
        self.grid_tree = KDTree(grid_coord_3d)
        print(f"Grid Initialisation: {time.time()-start}s")

    def evolve(self):
        start = time.time()
        height_map = np.random.random(self.points)
        plate_information = height_map*2 - 1
        velocity = np.random.random((self.points, 3))/100
        plate_velocity = []
        for velo in velocity:
            plate_velocity.append(Rotation.from_rotvec(velo))
        plate_velocity = np.array(plate_velocity)
        grid = Stationary_Grid(self.grid_coord_3d, self.grid_tree, height_map[self.grid_value], plate_velocity[self.grid_value], plate_information[self.grid_value],self.resolution)
        grid.display()
        print(f"Evolve Initialisation: {time.time()-start}s")
        for i in range(10):
            grid.evolve()
        print(f"Evolve Net time: {time.time()-start}s")
        grid.display()
            #now need to generate new coordinates and delete/compress the 


    def time_step(self, plate_position, plate_velocity):
        #generate the vector tangental to the radius and move the grid value
        # could consider the plate as a portion of the sphere and then rotate that wrt the center of the sphere
        #can generate the plate center as just a single vector and store the elements relative position to that center
        plate_position += plate_velocity
        plate_position[:, 0] = (plate_position[:, 0] + np.pi*plate_position[:, 1]   ) % 2*np.pi
        plate_position[:, 1] %= np.pi
        return plate_position

    def display(self):
        """
        Displays the plates

        :return:
        None
        """
        for i in range(self.points):
            test_values = []
            for n in range(len(self.grid_coords)):
                if self.grid_value[n] == i: test_values.append(self.grid_coords[n])
            test_values = np.array(test_values)
            if len(test_values) > 0: plt.scatter(test_values[:, 0], test_values[:, 1])
        plt.show()



if __name__ == '__main__':
    test = Tectonics(20)
    test.gen_vor_tes()
    #test.display()
    test.evolve()
    #test.display()


