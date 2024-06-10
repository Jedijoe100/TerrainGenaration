import numpy as np
from geometry import geographic_to_cartesian, cartesian_to_geographic, even_sphere_points
import matplotlib.pyplot as plt
import quads
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter
import time
import os
import pandas as pd
import threading
from amulet import load_level
from amulet.api.block import Block
from amulet.utils.world_utils import block_coords_to_chunk_coords
from amulet.api.errors import ChunkDoesNotExist
from amulet.api.chunk import Chunk
import shutil

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIRECTORY = "../Test_images"
THREAD_NUMBER = 4

SETTINGS = {
    'ID': 0,
    'CLOUD_POINT': 0.8,
    'EROSION_FACTOR': 4,
    'VOLCANIC_FACTOR': 1000,
    'PLATE_VELOCITY_FACTOR':  20,
    'WEATHER_STEPS': 200,
    'WIND_FACTOR': 100,
    'TEMPERATURE_RADIATE': 0.005,
    'SEA_LEVEL': 0.5,
    'VOLCANIC_SPOTS': 5,
    'PLATE_AGEING': 0.01,
    'HUMIDITY_FACTOR': 10,
    'POINT_NUMBER': 20,
    'DETAIL': 10000,
    'PLATE_STEPS': 16,
    'SEED': 1,
    'SEMI_MAJOR_AXIS': 1,
    'ECCENTRICITY': 0,
    'SOLAR_ENERGY': 7,
    'ANGLE_FACTOR': 20,
    'PLATE_RESETS': 2
}

class Stationary_Grid:
    def __init__(self, settings):#, grid, grid_tree, height, velocity, plate_type, resolution) -> None:
        start = time.time()
        if settings:
            self.settings = settings
        else:
            self.settings = SETTINGS
        self.rng = np.random.default_rng(self.settings['SEED'])
        self.resolution = np.array([200, 200])
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
        self.temperature = np.ones(self.size)*-10
        self.humidity = np.zeros(self.size)
        self.water_level = np.zeros(self.size)
        self.net_precipitation = np.zeros(self.size)
        self.biome = np.zeros(self.size)
        self.temperature_blocks = np.zeros((self.size, 9))
        self.weather_steps = 0
        _, self.adjacent_points = self.grid_tree.query(self.grid, 8)
        print(f"Grid Initialisation: {time.time()-start}s")
    
    def plate_reset(self):
        point_3d = self.rng.random((self.points, 3))*2 - 1
        # store these points in a lookup tree
        tree = KDTree(point_3d)
        grid_value = tree.query(self.grid)[1]
        average_plate_value = []
        for i in range(self.points):
            average_plate_value.append(np.average(self.plate_type[grid_value == i]))
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

    def weather(self, time_of_year):
        # update temperature
        self.temperature_update(time_of_year)
        # update humidity
        self.humidity += np.minimum(1, np.maximum(0, self.temperature/100)
                                    )*(self.height <= self.settings['SEA_LEVEL'])/self.settings['HUMIDITY_FACTOR']
        # set wind direction and compute the new index for each point
        wind_matrix = Rotation.from_rotvec(self.rng.random(3)/5)
        new_locations = wind_matrix.apply(self.grid)
        _, new_index = self.grid_tree.query(new_locations)
        # Transferring temperature to new point and humidity to new point (modifying temperature if going up or downhill)
        height_diff = self.height[new_index] - self.height
        new_temperature = self.temperature[new_index] - height_diff/10
        new_humidity = self.humidity[new_index]
        self.humidity = new_humidity
        self.temperature = new_temperature
        # computing precipitation
        precipitation = np.maximum(
            0, self.humidity - (np.minimum(100, np.maximum(0.0001, self.temperature)))/100)
        self.water_level += precipitation * \
            (precipitation > 0) * (self.height > self.water_level)
        self.humidity -= precipitation * (precipitation > 0)
        self.net_precipitation += precipitation
        self.weather_steps += 1
        # radiate temperature
        self.temperature -= ((self.humidity*100/(np.minimum(100,
                             np.maximum(0.0001, self.temperature)))) < self.settings['CLOUD_POINT']) * self.settings['TEMPERATURE_RADIATE']
        #self.temperature_blocks[:, np.maximum(np.minimum(self.temperature//10, 6), -2)] += 1
        #print(self.temperature_blocks)
        for value in self.adjacent_points:
            self.temperature[value] *= np.array([0.65, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
            self.humidity[value] *= np.array([0.65, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])


    def temperature_update(self, time_of_year):
        """
        Input:
        Planet Angle
        Planet Distance

        """
        Planet_angle = 2*np.pi/self.settings['ANGLE_FACTOR']
        year_progress = time_of_year * 2 * np.pi
        Distance = self.settings['SEMI_MAJOR_AXIS']*(1-np.square(self.settings['ECCENTRICITY']))/(1+ self.settings['ECCENTRICITY']*np.cos(year_progress))
        # this is the angle the current tilt of the planet wrt the sun
        angle_between_sun = Planet_angle * np.cos(year_progress)
        # this computes the amount of time in the sun
        ratio = self.grid[:, 2]*np.sin(angle_between_sun) / \
            np.sqrt(1-np.square(self.grid[:, 2]))
        proportion_day = (
            np.pi+2*(np.arcsin(np.maximum(-1, np.minimum(1, ratio)))))/(2*np.pi)
        # energy at each point
        solar_energy = np.sin(np.minimum(np.maximum(
            self.grid_2d[:, 1] - angle_between_sun, 0), np.pi))*self.settings['SOLAR_ENERGY']*proportion_day/(4*np.pi*np.square(Distance))
        self.temperature += ((self.humidity*100/(np.minimum(100,
                             np.maximum(0.0001, self.temperature)))) < self.settings['CLOUD_POINT'])*solar_energy
        

    def erosion(self, time_of_year):
        # update erosion
        self.weather(time_of_year)
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
            self.plate_reset()
            for i in range(int(self.settings['PLATE_STEPS'])):
                self.move_plates()
                for n in range(int(self.settings['WEATHER_STEPS'])):
                    self.eruptions()
                    self.erosion(n/self.settings['WEATHER_STEPS'])
                print(f"Evolution Step {i}: {time.time()-start}s")
        print(f"Evolve Net time: {time.time()-start}s")
    
    def biomes(self):
        pass

    def display(self):
        # display a height, temperature, humidity, water map on a single figure, store image and data
        xx, yy = np.meshgrid(np.linspace(
            0, 2*np.pi, self.resolution[0]), np.linspace(-np.pi/2, np.pi/2, self.resolution[1]))
        _, indices = self.grid_tree.query(geographic_to_cartesian(
            np.array([xx.flatten(), yy.flatten()]).transpose()))
        height_map = self.height[indices].reshape(self.resolution)
        fig, axs = plt.subplots(2, 2)
        print(np.min(height_map), np.max(height_map))
        axs[0, 0].contourf(xx, yy, height_map, levels=50)
        cb_00 = axs[0, 0].set_title('Height Map')
        # fig.colorbar(cb_00, axs[0, 0])
        temperature_map = self.temperature[indices].reshape(self.resolution)
        print(np.min(temperature_map), np.max(temperature_map))
        axs[0, 1].contourf(xx, yy, temperature_map)
        cb_01 = axs[0, 1].set_title('Temperature Map')
        # fig.colorbar(cb_01, axs[0, 1])
        humidity_map = self.humidity[indices].reshape(self.resolution)
        print(np.min(humidity_map), np.max(humidity_map))
        axs[1, 0].contourf(xx, yy, humidity_map)
        cb_10 = axs[1, 0].set_title('Humidity Map')
        # fig.colorbar(cb_10, axs[1, 0])
        water_map = self.water_level[indices].reshape(self.resolution)
        print(np.min(water_map), np.max(water_map))
        axs[1, 1].contourf(xx, yy, water_map)
        cb_11 = axs[1, 1].set_title('Water Map')
        # fig.colorbar(cb_11, axs[1, 1])
        plt.savefig(
            os.path.join(FILE_PATH, IMAGE_DIRECTORY, f"test_{self.settings['ID']}.png"))
        plt.show()
        rain_fall = self.net_precipitation[indices].reshape(
            self.resolution)/self.weather_steps
        print(np.min(rain_fall), np.max(rain_fall))
        with open(os.path.join(FILE_PATH, "settings_test.csv"), "a") as file:
            file.write(f"{self.settings['ID']},{np.min(height_map)},{np.max(height_map)},{np.min(temperature_map)},{np.max(temperature_map)},{np.min(humidity_map)},{np.max(humidity_map)},{np.min(water_map)},{np.max(water_map)},{np.min(rain_fall)},{np.max(rain_fall)}\n")
    
    def export_to_minecraft_world(self):
        shutil.rmtree(os.path.join(FILE_PATH, '..\\current_world'))
        shutil.copytree(os.path.join(FILE_PATH, '..\\biome_template'), os.path.join(FILE_PATH, '../current_world'))
        world_resolution = self.resolution*5
        level = load_level('current_world')
        xx, yy = np.meshgrid(np.linspace(
            0, 2*np.pi, world_resolution[0]), np.linspace(-np.pi/2, np.pi/2, world_resolution[1]))
        _, indices = self.grid_tree.query(geographic_to_cartesian(
            np.array([xx.flatten(), yy.flatten()]).transpose()))
        height_map = self.height[indices].reshape(world_resolution)
        height_map = gaussian_filter(height_map, sigma=2)
        processed_height = 50*(height_map-np.min(height_map))/(np.max(height_map)-np.min(height_map))
        processed_sea_level = 50*(self.settings['SEA_LEVEL']-np.min(height_map))/(np.max(height_map)-np.min(height_map))
        print(processed_sea_level, np.min(processed_height), np.min(height_map), self.settings['SEA_LEVEL'])
        (
            universal_block_1,
            universal_block_entity_1,
            universal_extra_1,
        ) = level.translation_manager.get_version("java", (1, 19, 4)).block.to_universal(
            Block("minecraft", "stone")
        )
        stone = level.block_palette.get_add_block(universal_block_1) 
        (
            universal_block_1,
            universal_block_entity_1,
            universal_extra_1,
        ) = level.translation_manager.get_version("java", (1, 19, 4)).block.to_universal(
            Block("minecraft", "dirt")
        )
        dirt = level.block_palette.get_add_block(universal_block_1) 
        (
            universal_block_1,
            universal_block_entity_1,
            universal_extra_1,
        ) = level.translation_manager.get_version("java", (1, 19, 4)).block.to_universal(
            Block("minecraft", "grass_block")
        )
        grass = level.block_palette.get_add_block(universal_block_1) 
        (
            universal_block_1,
            universal_block_entity_1,
            universal_extra_1,
        ) = level.translation_manager.get_version("java", (1, 19, 4)).block.to_universal(
            Block("minecraft", "gravel")
        )
        gravel = level.block_palette.get_add_block(universal_block_1) 
        (
            universal_block_1,
            universal_block_entity_1,
            universal_extra_1,
        ) = level.translation_manager.get_version("java", (1, 19, 4)).block.to_universal(
            Block("minecraft", "water")
        )
        water = level.block_palette.get_add_block(universal_block_1) 
        for x in range(world_resolution[0]):
            for z in range(world_resolution[1]):
                cx, cz = block_coords_to_chunk_coords(x, z)
                try:
                    chunk = level.get_chunk(cx, cz, "minecraft:overworld")
                except ChunkDoesNotExist:
                    new_chunk = Chunk(cx, cz)
                    level.put_chunk(new_chunk, "minecraft:overworld")
                    chunk = level.get_chunk(cx, cz, "minecraft:overworld")
                offset_x, offset_z = x - 16 * cx, z - 16 * cz
                chunk.blocks[offset_x, 1:int(processed_height[x,z])-3, offset_z] = stone
                if height_map[x, z] > self.settings['SEA_LEVEL']:
                    chunk.blocks[offset_x, int(processed_height[x,z])-3:int(processed_height[x,z]), offset_z] = dirt
                    chunk.blocks[offset_x, int(processed_height[x,z]), offset_z] = grass
                else:
                    chunk.blocks[offset_x, int(processed_height[x,z])-3:int(processed_height[x,z]), offset_z] = gravel
                    chunk.blocks[offset_x, int(processed_height[x,z]):processed_sea_level, offset_z] = water
                chunk.changed = True
        level.save()
        level.close()

def thread_function(settings):
    for setting in settings:
        test = Stationary_Grid(setting)
        test.evolve()
        test.display()


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

        
