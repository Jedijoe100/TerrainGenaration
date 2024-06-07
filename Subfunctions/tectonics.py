import numpy as np
from geometry import geographic_to_cartesian, cartesian_to_geographic, even_sphere_points
import matplotlib.pyplot as plt
import quads
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import time
import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIRECTORY = "../Test_images"

CLOUD_POINT = 0.8
EROSION_FACTOR = 3
VOLCANIC_FACTOR = 1000
PLATE_VELOCITY_FACTOR = 20
WEATHER_STEPS = 1000
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
        self.grid_2d = cartesian_to_geographic(grid)
        self.grid_tree = grid_tree
        self.height = height
        self.size = len(height)
        self.velocity = velocity
        self.plate_type = plate_type
        self.resolution = resolution
        self.resolution_step = (2*np.pi/resolution[1], np.pi/resolution[1])
        self.volcanism = np.zeros(self.size)
        self.num_spots = 5
        self.volcanism_spots = np.random.random((self.num_spots, 3))
        self.temperature = np.zeros(self.size)
        self.humidity = np.zeros(self.size)
        self.water_level = np.zeros(self.size)
        self.net_precipitation = np.zeros(self.size)
        self.sea_level = 0.5
        self.weather_steps = 0

    
    
    def move_plates(self):
        # There is still an artifact on the center vertical line
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
            elif self.plate_type[i] < 0 and next_plate_step[new_index[0]] > self.plate_type[i]:
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
        self.temperature_update(0.5)
        self.humidity += np.minimum(1, np.maximum(0, self.temperature/100))*(self.height <= self.sea_level)/10
        wind_matrix = Rotation.from_rotvec(np.random.random(3)/5)
        new_locations = wind_matrix.apply(self.grid)
        _, new_index = self.grid_tree.query(new_locations)
        height_diff = self.height[new_index] - self.height
        new_temperature = self.temperature[new_index] - height_diff/10
        new_humidity = self.humidity[new_index]
        self.humidity = new_humidity
        self.temperature = new_temperature
        precipitation = np.maximum(0, self.humidity - (np.minimum(100, np.maximum(0.0001, self.temperature)))/100)
        self.water_level += precipitation * (precipitation > 0)
        self.humidity -= precipitation * (precipitation > 0)
        self.net_precipitation +=precipitation
        self.weather_steps += 1

        # try a global direction vector
        # if move downhill temperature increases
        # if move uphill temperature decreases
        # defuse temperature and humidity
        # if humidity exceeds carrying capacity precipitate

    def temperature_update(self, time_of_year):
        """
        Input:
        Time of year
        Planet Angle
        Planet Distance

        """
        Solar_Energy = 10
        Distance = 1
        Planet_angle = np.pi/10
        year_progress = time_of_year * 2 * np.pi
        # this is the angle the current tilt of the planet wrt the sun
        angle_between_sun = Planet_angle * np.cos(year_progress)#np.arccos(np.dot((0, np.sin(Planet_angle), np.cos(Planet_angle)), (np.sin(year_progress), np.cos(year_progress), 0)))
        # this computes the amount of time in the sun
        ratio = self.grid[:, 2]*np.sin(angle_between_sun)/np.sqrt(1-np.square(self.grid[:, 2]))
        proportion_day = (np.pi+2*(np.arcsin(np.maximum(-1, np.minimum(1, ratio)))))/(2*np.pi)
        # energy at each point
        solar_energy = np.sin(np.minimum(np.maximum(self.grid_2d[:, 1] - angle_between_sun, 0), np.pi))*Solar_Energy*proportion_day/(4*np.pi*np.square(Distance))
        self.temperature += ((self.humidity*100/(np.minimum(100, np.maximum(0.0001, self.temperature)))) < CLOUD_POINT)*solar_energy

    def erosion(self):
        self.weather()
        for i in range(self.size):
            _, near_points = self.grid_tree.query(self.grid[i], 8)
            minimum_min = near_points[np.argmin(self.height[near_points]+ self.water_level[near_points])]
            water_flow = np.minimum(self.water_level[i], np.maximum(0, self.water_level[i] + self.height[i] - self.height[minimum_min]- self.water_level[minimum_min])/2)
            if water_flow < 0: print(water_flow)
            #if water_flow > 0: print(water_flow, i, minimum_min)
            self.water_level[i] = np.maximum(self.water_level[i] - water_flow, 0)
            self.height[i] -= water_flow/EROSION_FACTOR
            self.water_level[minimum_min] += (self.height[minimum_min]>self.sea_level)*water_flow
            self.height[minimum_min] += water_flow/EROSION_FACTOR
        # to spread out the height map

    def eruptions(self):
        _, indices = self.grid_tree.query(self.volcanism_spots)
        for index in indices:
            self.height[index] += np.random.random()/VOLCANIC_FACTOR
        self.height += self.volcanism*np.random.random(self.size)*(self.volcanism >= 1)/VOLCANIC_FACTOR
        self.volcanism = self.volcanism * (self.volcanism < 1)

    def evolve(self):
        self.move_plates()
        for i in range(WEATHER_STEPS):
            self.eruptions()
            self.erosion()

    def display(self):
        #display a height map using a contour map

        xx, yy = np.meshgrid(np.linspace(0, 2*np.pi, self.resolution[0]), np.linspace(-np.pi/2, np.pi/2, self.resolution[1]))
        _, indices = self.grid_tree.query(geographic_to_cartesian(np.array([xx.flatten(), yy.flatten()]).transpose()))
        height_map = self.height[indices].reshape(self.resolution)
        fig, axs = plt.subplots(2, 2)
        print(np.min(height_map), np.max(height_map))
        axs[0, 0].contourf(xx, yy, height_map, levels=50)
        axs[0, 0].set_title('Height Map')
        temperature_map = self.temperature[indices].reshape(self.resolution)
        print(np.min(temperature_map), np.max(temperature_map))
        axs[0, 1].contourf(xx, yy, temperature_map)
        axs[0, 1].set_title('Temperature Map')
        humidity_map = self.humidity[indices].reshape(self.resolution)
        print(np.min(humidity_map), np.max(humidity_map))
        axs[1, 0].contourf(xx, yy, humidity_map)
        axs[1, 0].set_title('Humidity Map')
        water_map = self.water_level[indices].reshape(self.resolution)
        print(np.min(water_map), np.max(water_map))
        axs[1, 1].contourf(xx, yy, water_map)
        axs[1, 1].set_title('Water Map')
        plt.savefig(
            os.path.join(FILE_PATH, IMAGE_DIRECTORY, "test.png"))
        plt.show()
        rain_fall = self.net_precipitation[indices].reshape(self.resolution)/self.weather_steps
        print(np.min(rain_fall), np.max(rain_fall))
        # plt.contourf(xx, yy, height_map)
        # plt.contour(xx, yy, rain_fall)

class Tectonics:
    def __init__(self, point_num):
        # need to add shape
        self.grid_value = None
        self.tree = None
        self.grid_coords = None
        self.points = point_num
        self.resolution = (100, 100)
        self.detail = 2000

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
        velocity = np.random.random((self.points, 3))/PLATE_VELOCITY_FACTOR
        plate_velocity = []
        for velo in velocity:
            plate_velocity.append(Rotation.from_rotvec(velo))
        plate_velocity = np.array(plate_velocity)
        grid = Stationary_Grid(self.grid_coord_3d, self.grid_tree, height_map[self.grid_value], plate_velocity[self.grid_value], plate_information[self.grid_value],self.resolution)
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


