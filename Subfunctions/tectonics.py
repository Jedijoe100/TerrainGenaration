import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter
from geometry import geographic_to_cartesian, cartesian_to_geographic, even_sphere_points
from minecraft import chunk_based_generation
from display import display_sub_axes, store_data_template
from biome import Biomes
from PIL import Image
import time
import os
import shutil
import threading
from amulet import load_level

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIRECTORY = "../Test_images"
THREAD_NUMBER = 4

SETTINGS = {
    'ID': 0,
    'CLOUD_POINT': 0.8,
    'EROSION_FACTOR': 2,
    'VOLCANIC_FACTOR': 1000,
    'PLATE_VELOCITY_FACTOR': 20,
    'WEATHER_STEPS': 200,
    'WIND_FACTOR': 50,
    'TEMPERATURE_RADIATE': 0.01,
    'SEA_LEVEL': 0.5,
    'VOLCANIC_SPOTS': 5,
    'PLATE_AGEING': 0.01,
    'HUMIDITY_FACTOR': 10,
    'POINT_NUMBER': 20,
    'DETAIL': 10000,
    'PLATE_STEPS': 10,
    'SEED': 2,
    'SEMI_MAJOR_AXIS': 1,
    'ECCENTRICITY': 0,
    'SOLAR_ENERGY': 2,
    'ANGLE_FACTOR': 0.02,
    'PLATE_RESETS': 10,
    'CELL_NUMBER': 3,
    'SPIN': 1,
    'WIND_SPEED': 0,
    'HEIGHT_TEMP_DROP': 5,
    'ADJACENT_FACTOR': 16,
    'ALTITUDE_FACTOR': 20
}
MINECRAFT_SETTINGS = {
    'HEIGHT_DIFF': 100,
    'WORLD_RESOLUTION': (1000, 500),
    'LOWEST_POINT': 50
}
BIOMES = Biomes()

class Stationary_Grid:
    def __init__(self, settings):#, grid, grid_tree, height, velocity, plate_type, resolution) -> None:
        start = time.time()
        if settings:
            self.settings = settings
        else:
            self.settings = SETTINGS
        self.rng = np.random.default_rng(self.settings['SEED'])
        self.resolution = np.array([400, 400])
        self.points = int(self.settings['POINT_NUMBER'])
        # Generate information
        point_3d = self.rng.random((self.points, 3))*2 - 1
        # store these points in a lookup tree
        tree = KDTree(point_3d)
        # generate the grid of points
        self.grid = even_sphere_points(int(self.settings['DETAIL']))
        # assigning each grid point a
        grid_value = tree.query(self.grid)[1]
        self.grid_tree = KDTree(self.grid)
        self.grid_2d = cartesian_to_geographic(self.grid)
        self.height = self.rng.random(self.points)[grid_value]
        self.size = len(self.height)
        self.plate_type = self.height*2 - 1
        velocity = self.rng.random((self.points, 3))/self.settings['PLATE_VELOCITY_FACTOR']
        self.velocity = []
        for i in grid_value:
            self.velocity.append(Rotation.from_rotvec(velocity[i]))
        # now need to generate new coordinates and delete/compress the
        self.resolution_step = (2*np.pi/self.resolution[1], np.pi/self.resolution[1])
        self.volcanism = np.zeros(self.size)
        self.volcanism_spots = self.rng.random((int(self.settings['VOLCANIC_SPOTS']), 3))
        self.temperature = np.ones(self.size*2)*-10
        self.humidity = np.zeros(self.size*2)
        self.water_level = np.zeros(self.size)
        self.net_precipitation = np.zeros(self.size)
        self.biome = np.zeros(self.size)
        self.biome_blocks = np.zeros((self.size, 15))
        self.weather_steps = 0
        self.weather_index = self.Wind_Map()
        self.layer_2 = np.zeros(self.size*2)
        self.layer_2[self.size:] = 1
        self.water_flow = np.zeros(self.size)
        self.time_taken = 0
        _, self.adjacent_points = self.grid_tree.query(self.grid, self.settings['ADJACENT_FACTOR'])
        print(f"Grid Initialisation: {time.time()-start}s")
    

    def Wind_Map(self):
        bottom_layer_lat = self.settings['WIND_SPEED']*(np.ceil(np.sin(self.settings['CELL_NUMBER']*self.grid_2d[:, 1]*np.pi))*2-1)
        top_layer_lat = -bottom_layer_lat
        bottom_long = bottom_layer_lat*self.settings['SPIN']*np.sin(self.grid_2d[:,1])
        top_long = top_layer_lat*self.settings['SPIN']*np.sin(self.grid_2d[:,1])
        new_index = np.zeros(2*self.size)
        point_density = np.sqrt(4*np.pi/self.settings['DETAIL'])
        altitude_change_points = np.linspace(0, 1, 2*self.settings['CELL_NUMBER']+1)
        rise_grid = []
        fall_grid = []
        for n in range(2*self.settings['CELL_NUMBER']+1):
            point_number = int(1 + 2*np.pi*np.sin(altitude_change_points[n]*np.pi)/point_density)
            coords = np.ones((point_number*2, 2))*altitude_change_points[n]*np.pi
            coords[:point_number, 1] = altitude_change_points[n]*np.pi + point_density/2
            coords[point_number:, 1] = altitude_change_points[n]*np.pi - point_density/2
            coords[:point_number, 0] = np.linspace(0, 2*np.pi, point_number)
            coords[point_number:, 0] = np.linspace(0, 2*np.pi, point_number)
            coords_3d = geographic_to_cartesian(coords)
            if n % 2 == self.settings['CELL_NUMBER']%2:
                fall_grid += list(np.unique(self.grid_tree.query(coords_3d)[1]))
            else: 
                rise_grid += list(np.unique(self.grid_tree.query(coords_3d)[1]))
        rise_grid = np.array(rise_grid)
        fall_grid = np.array(fall_grid)
        for i in range(self.size):
            wind_matrix = Rotation.from_rotvec((bottom_layer_lat[i], bottom_long[i], 0))
            new_location = wind_matrix.apply(self.grid[i])
            new_index[i] = self.grid_tree.query(new_location)[1]
        for i in range(self.size):
            wind_matrix = Rotation.from_rotvec((top_layer_lat[i], top_long[i], 0))
            new_location = wind_matrix.apply(self.grid[i])
            new_index[i+self.size] = self.grid_tree.query(new_location)[1] + self.size
        new_index[rise_grid] = rise_grid+ self.size
        new_index[fall_grid+self.size] = fall_grid
        return new_index.astype(int)
        


    
    def plate_reset(self):
        point_3d = self.rng.random((self.points, 3))*2 - 1
        # store these points in a lookup tree
        tree = KDTree(point_3d)
        grid_value = np.array(tree.query(self.grid)[1])
        average_plate_value = []
        for i in range(self.points):
            average_plate_value.append(np.average(self.plate_type[grid_value == i]))
        average_plate_value = np.array(average_plate_value)
        self.plate_type = average_plate_value[grid_value]
        velocity = self.rng.random((self.points, 3))/self.settings['PLATE_VELOCITY_FACTOR']
        self.velocity = []
        for i in grid_value:
            self.velocity.append(Rotation.from_rotvec(velocity[i]))
        

    def move_plates(self):
        # Setup temporary storage
        next_step_height = np.zeros(self.size)
        next_step_velocity = self.velocity.copy()
        next_plate_step = -np.ones(self.size)
        for i in range(self.size):
            # compute the new location of points
            point = np.array([self.velocity[i].apply(self.grid[i])])
            # find index of new location
            _, new_index = self.grid_tree.query(point)
            # add height to new location
            next_step_height[new_index[0]] += self.height[i]
            # check if it is the dominant plate
            if next_plate_step[new_index[0]] < self.plate_type[i]:
                # transferring information
                next_step_velocity[new_index[0]] = self.velocity[i]
                next_plate_step[new_index[0]] = self.plate_type[i]
            elif self.plate_type[i] < 0 and next_plate_step[new_index[0]] > self.plate_type[i]:
                # to allow for simulated volcanism creating mountains and island chains
                self.volcanism[new_index[0]] += 0.1
                # to try and simulate trenches
                next_step_height[i] = 0  
        # storing plate_type
        self.height = next_step_height
        self.velocity = next_step_velocity
        self.plate_type = next_plate_step
        # Ageing the plate
        self.plate_type += self.settings['PLATE_AGEING']

    def weather(self, time_of_year, final):
        # update temperature
        self.temperature_update(time_of_year)
        # update humidity
        self.humidity[:self.size] += (self.height <= self.settings['SEA_LEVEL'])/self.settings['HUMIDITY_FACTOR'] + self.water_level/self.settings['HUMIDITY_FACTOR']
        self.water_level -= self.water_level/self.settings['HUMIDITY_FACTOR']
        # Transferring temperature to new point and humidity to new point (modifying temperature if going up or downhill)
        height_diff = np.zeros(self.size*2)
        height_diff[:self.size] = self.height[self.weather_index[:self.size]%self.settings['DETAIL']] - self.height
        new_temperature = self.temperature[self.weather_index] - height_diff/self.settings['ALTITUDE_FACTOR']
        new_humidity = self.humidity[self.weather_index]
        # computing precipitation
        precipitation = np.maximum(
            0, new_humidity - (np.minimum(100, np.maximum(0, new_temperature -self.settings['HEIGHT_TEMP_DROP']*self.layer_2)))/100)
        new_humidity -= precipitation * (precipitation > 0)
        net_precipitation = precipitation[:self.size] + precipitation[self.size:]
        self.water_level += net_precipitation * \
            (net_precipitation > 0) * (self.height > self.water_level)
        # radiate temperature
        new_temperature -= ((new_humidity*100/(np.minimum(100,
                             np.maximum(0.0001, new_temperature)))) < self.settings['CLOUD_POINT']) * self.settings['TEMPERATURE_RADIATE']
        #blurring humidity and temperature
        tem_temperature = new_temperature*0.6
        tem_humidity = new_humidity*0.6
        tem_precipitation = net_precipitation*0.6
        for i in range(self.size):
            tem_temperature[self.adjacent_points[i]] += new_temperature[i]*0.4/self.settings['ADJACENT_FACTOR']
            tem_temperature[self.adjacent_points[i]+self.size] += new_temperature[i+self.size]*0.4/self.settings['ADJACENT_FACTOR']
            tem_humidity[self.adjacent_points[i]] += new_humidity[i] *0.4/self.settings['ADJACENT_FACTOR']
            tem_humidity[self.adjacent_points[i]+self.size] += new_humidity[i+self.size] *0.4/self.settings['ADJACENT_FACTOR']
            tem_precipitation[self.adjacent_points[i]] += net_precipitation[i] *0.4/self.settings['ADJACENT_FACTOR']
        self.temperature = tem_temperature
        self.humidity = tem_humidity
        self.net_precipitation += net_precipitation
        self.weather_steps += 1
        #store temperature data
        if final:
            precipitation_max = np.max(net_precipitation)+0.0000000000001
            temp_categories = np.array([-1000, -10, 0, 10, 20, 30, 40, 50, 60, 10000])
            humidity_categories = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
            temperature_values = 5+np.digitize(self.temperature, temp_categories)
            precipitation_values = np.digitize(net_precipitation/precipitation_max, humidity_categories)
            for i in range(self.size):
                self.biome_blocks[i,int(temperature_values[i])] += 1
                self.biome_blocks[i,int(precipitation_values[i])] += 1



    def temperature_update(self, time_of_year):
        """
        Input:
        Planet Angle
        Planet Distance
        """
        Planet_angle = 2*np.pi*self.settings['ANGLE_FACTOR']
        year_progress = time_of_year * 2 * np.pi
        Distance = self.settings['SEMI_MAJOR_AXIS']*(1-np.square(self.settings['ECCENTRICITY']))/(1+ self.settings['ECCENTRICITY']*np.cos(year_progress))
        # this is the angle the current tilt of the planet wrt the sun
        angle_between_sun = Planet_angle * np.cos(year_progress)
        # this computes the amount of time in the sun
        proportion_day = 1/2 - np.sin(self.grid_2d[:, 1])*np.tan(angle_between_sun)/(2*np.sin(self.grid_2d[:, 1]))
        # energy at each point
        solar_energy = np.sin(np.minimum(np.maximum(
            self.grid_2d[:, 1] - angle_between_sun, 0), np.pi))*self.settings['SOLAR_ENERGY']*proportion_day/(4*np.pi*np.square(Distance))
        self.temperature[:self.size] += ((self.humidity[:self.size]*100/(np.minimum(100,
                             np.maximum(0.0001, self.temperature[:self.size])))) < self.settings['CLOUD_POINT'])*solar_energy
        

    def erosion(self, time_of_year):
        # update erosion
        self.weather(time_of_year, False)
        self.water_level = self.water_level*(self.height > self.settings['SEA_LEVEL'])
        for i in range(self.size):
            # find a close point that is lower than current point
            minimum_min = self.adjacent_points[i, np.argmin(
                self.height[self.adjacent_points[i,:]] + self.water_level[self.adjacent_points[i,:]])]
            height_difference = np.maximum(
                0, self.water_level[i] + self.height[i] - self.height[minimum_min] - self.water_level[minimum_min])/2
            #compute the water flow between the lower point and the point
            water_flow = np.minimum(self.water_level[i], height_difference)
            # Transfer the water and erode
            self.water_level[i] = np.maximum(
                self.water_level[i] - water_flow, 0)
            self.height[i] -= water_flow/self.settings['EROSION_FACTOR'] + \
                height_difference/self.settings['WIND_FACTOR']
            self.water_level[minimum_min] += (
                self.height[minimum_min] > self.settings['SEA_LEVEL'])*water_flow
            self.height[minimum_min] += water_flow / \
                self.settings['EROSION_FACTOR'] + height_difference/self.settings['WIND_FACTOR']

    def eruptions(self):
        # compute volcano spots
        _, indices = self.grid_tree.query(self.volcanism_spots)
        for index in indices:
            self.height[index] += self.rng.random()/self.settings['VOLCANIC_FACTOR']
        # compute volcano chains
        self.volcanism = self.volcanism*self.rng.random(self.size)*2
        self.height += self.volcanism * \
            self.rng.random(self.size)*(self.volcanism >= 1)/self.settings['VOLCANIC_FACTOR']
        self.volcanism = self.volcanism * (self.volcanism < 1)

    def evolve(self):
        # Compute all plate steps
        start = time.time()
        for _ in range(self.settings['PLATE_RESETS']):
            for i in range(int(self.settings['PLATE_STEPS'])):
                self.move_plates()
                for n in range(int(self.settings['WEATHER_STEPS'])):
                    self.eruptions()
                    self.erosion(n/self.settings['WEATHER_STEPS'])
                print(f"Evolution Step {i}: {time.time()-start}s")
            self.plate_reset()
            self.net_precipitation = 0
        for i in range(int(self.settings['WEATHER_STEPS'])):
            self.weather(i/self.settings['WEATHER_STEPS'],True)
        print(f"Evolve Net time: {time.time()-start}s")
        self.biomes()
        self.time_taken = time.time()-start
    
    def biomes(self):
        alt_categories = np.array([0, 0.3, 0.6, 0.8, 0.95, 1, 1.5, 10000])
        alt_value = np.arange(0, 7, 1)
        alt_categories[:-1] *= self.settings['SEA_LEVEL']
        #select minimum non negative height category
        alt_value = np.digitize(self.height, alt_categories)
        self.biome_blocks /= self.settings['WEATHER_STEPS']
        self.biome_blocks[:,0] = self.water_level
        self.biome = BIOMES.biome_test(alt_value, self.biome_blocks)
        print(BIOMES.biomes[np.unique(self.biome), 1])
    
    def generate_biome_image(self):
        xx, yy = np.meshgrid(np.linspace(
            0, 2*np.pi, self.resolution[0]), np.linspace(0, np.pi, self.resolution[1]))
        _, indices = self.grid_tree.query(geographic_to_cartesian(
            np.array([xx.flatten(), yy.flatten()]).transpose()))
        colour_points = BIOMES.map_colours[self.biome[indices].reshape(self.resolution)].astype(np.uint8)
        data = Image.fromarray(colour_points, 'RGB')
        data.save(os.path.join(FILE_PATH, IMAGE_DIRECTORY, f"biomes_seed_{self.settings['SEED']}.png"),"png")
        data.show()

    def display(self):
        # display a height, temperature, humidity, water map on a single figure, store image and data
        xx, yy = np.meshgrid(np.linspace(
            0, 2*np.pi, self.resolution[0]), np.linspace(0, np.pi, self.resolution[1]))
        print(np.shape(xx), np.shape(yy))
        indices = self.grid_tree.query(geographic_to_cartesian(
            np.array([xx.flatten(), yy.flatten()]).transpose()))[1]
        fig, axs = plt.subplots(3, 2)
        display_sub_axes(axs[0, 0], xx, yy, self.height[indices].reshape(self.resolution), 'Height Map')
        display_sub_axes(axs[0, 1], xx, yy, self.humidity[indices].reshape(self.resolution), 'Humidity Map')
        display_sub_axes(axs[1, 0], xx, yy, self.net_precipitation[indices].reshape(self.resolution), 'Precipitation Map')
        display_sub_axes(axs[1, 1], xx, yy, self.water_level[indices].reshape(self.resolution), 'Water Map')
        display_sub_axes(axs[2, 0], xx, yy, self.temperature[indices].reshape(self.resolution), 'Temperature Map')
        display_sub_axes(axs[2, 1], xx, yy, self.temperature[indices+self.size].reshape(self.resolution), 'High Alt Temperature Map')
        plt.savefig(
            os.path.join(FILE_PATH, IMAGE_DIRECTORY, f"test_{self.settings['ID']}.png"))
        plt.show()
        self.generate_biome_image()
    
    def store_data(self):
        with open(os.path.join(FILE_PATH, "settings_test.csv"), "a") as file:
            file.write(f"{self.settings['ID']},{self.time_taken},{store_data_template([self.height, self.humidity, self.net_precipitation, self.water_level, self.temperature[:self.size], self.temperature[self.size:]])}\n")
        
    
    def export_to_minecraft_world(self):
        shutil.rmtree(os.path.join(FILE_PATH, '..\\current_world'))
        shutil.copytree(os.path.join(FILE_PATH, '..\\biome_template'), os.path.join(FILE_PATH, '../current_world'))
        level = load_level('current_world')
        for biome in BIOMES.minecraft_biomes:
            level.biome_palette.register(f'universal_minecraft:{biome}')
        level = chunk_based_generation(level, MINECRAFT_SETTINGS['WORLD_RESOLUTION'], self, BIOMES)
        level.save()
        level.close()

def thread_function(settings):
    for setting in settings:
        test = Stationary_Grid(setting)
        test.evolve()
        test.display()
        test.store_data()


if __name__ == '__main__':
    is_main = True
    if is_main:
        """Main run"""
        test = Stationary_Grid(SETTINGS)
        test.evolve()
        test.display()
        test.export_to_minecraft_world()
    else:
        thread_settings = [[] for _ in range(THREAD_NUMBER)]
        test_settings = pd.read_csv(os.path.join(FILE_PATH, "settings.csv"), dtype={'ID': np.int32, 'WEATHER_STEP': np.int32, 'VOLCANIC_SPOTS': np.int32})
        for i in range(len(test_settings)):
            thread_settings[i%THREAD_NUMBER].append(test_settings.loc[i].to_dict())
        for setting in thread_settings:
            x = threading.Thread(target=thread_function, args=(setting,))
            x.start()

        
