import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
from Subfunctions.geometry import cartesian_to_geographic, even_sphere_points
from Subfunctions.weather import Wind_Map, weather
from Subfunctions.erosion import erosion
from Subfunctions.tectonics import eruptions, plate_reset, move_plates
from Subfunctions.minecraft import export_to_minecraft_world
from Subfunctions.display import display_biome, display_layers, store_data_template
from Subfunctions.biome import Biomes, biomes
import time
import os
import threading

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIRECTORY = "./images"
DATA_DIRECTORY = "./data"
THREAD_NUMBER = 4

SETTINGS = {
    'ID': 0,
    'CLOUD_POINT': 0.6,
    'EROSION_FACTOR': 2,
    'VOLCANIC_FACTOR': 1000,
    'PLATE_VELOCITY_FACTOR': 20,
    'WEATHER_STEPS': 200,
    'WIND_FACTOR': 50,
    'TEMPERATURE_RADIATE': 0.005,
    'SEA_LEVEL': 0.5,
    'VOLCANIC_SPOTS': 5,
    'PLATE_AGEING': 0.01,
    'HUMIDITY_FACTOR': 40,
    'POINT_NUMBER': 20,
    'DETAIL': 10000,
    'PLATE_STEPS': 2,
    'SEED': 2,
    'SEMI_MAJOR_AXIS': 1,
    'ECCENTRICITY': 0,
    'SOLAR_ENERGY': 6,
    'ANGLE_FACTOR': 0.04,
    'PLATE_RESETS': 1,
    'CELL_NUMBER': 3,
    'SPIN': 1,
    'WIND_SPEED': 0.01,
    'HEIGHT_TEMP_DROP': 5,
    'ADJACENT_FACTOR': 32,
    'ALTITUDE_FACTOR': 10,
    'BLUR_FACTOR': 0.2
}
BIOMES = Biomes(os.path.join(FILE_PATH, DATA_DIRECTORY, "biomes.csv"))

class Stationary_Grid:
    def __init__(self, settings) -> None:
        start = time.time()
        if settings:
            self.settings = settings
        else:
            self.settings = SETTINGS
        self.rng = np.random.default_rng(int(self.settings['SEED']))
        self.resolution = np.array([800, 800])
        self.points = int(self.settings['POINT_NUMBER'])
        point_3d = self.rng.random((self.points, 3))*2 - 1
        tree = KDTree(point_3d)
        self.grid = even_sphere_points(int(self.settings['DETAIL']))
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
        self.weather_index = Wind_Map(self)
        self.layer_2 = np.zeros(self.size*2)
        self.layer_2[self.size:] = 1
        self.water_flow = np.zeros(self.size)
        self.time_taken = 0
        _, self.adjacent_points = self.grid_tree.query(self.grid, self.settings['ADJACENT_FACTOR'])
        print(f"Grid Initialisation: {time.time()-start}s")

    def evolve(self):
        # Compute all plate steps
        start = time.time()
        for _ in range(int(self.settings['PLATE_RESETS'])):
            for i in range(int(self.settings['PLATE_STEPS'])):
                move_plates(self)
                for n in range(int(self.settings['WEATHER_STEPS'])):
                    eruptions(self)
                    erosion(self, n/self.settings['WEATHER_STEPS'])
                print(f"Evolution Step {i}: {time.time()-start}s")
            plate_reset(self)
            self.net_precipitation = 0
        for i in range(int(self.settings['WEATHER_STEPS'])):
            weather(self,i/self.settings['WEATHER_STEPS'],True)
        print(f"Evolve Net time: {time.time()-start}s")
        biomes(self, BIOMES)
        self.time_taken = time.time()-start
    
    def store_data(self):
        with open(os.path.join(FILE_PATH, "settings_test.csv"), "a") as file:
            file.write(f"{self.settings['ID']},{self.time_taken},{store_data_template([self.height, self.humidity, self.net_precipitation, self.water_level, self.temperature[:self.size], self.temperature[self.size:]])}\n")

def thread_function(settings):
    path = os.path.join(FILE_PATH, IMAGE_DIRECTORY)
    for setting in settings:
        test = Stationary_Grid(setting)
        test.evolve()
        display_layers(test, path)
        display_biome(test, BIOMES, path)
        test.store_data()


if __name__ == '__main__':
    is_main = True
    if is_main:
        """Main run"""
        test = Stationary_Grid(SETTINGS)
        test.evolve()
        path = os.path.join(FILE_PATH, IMAGE_DIRECTORY)
        display_layers(test, path)
        display_biome(test, BIOMES, path)
        export_to_minecraft_world(test, FILE_PATH,BIOMES)
    else:
        thread_settings = [[] for _ in range(THREAD_NUMBER)]
        test_settings = pd.read_csv(os.path.join(FILE_PATH, "settings.csv"), dtype={'ID': np.int32, 'WEATHER_STEP': np.int32, 'VOLCANIC_SPOTS': np.int32})
        for i in range(len(test_settings)):
            thread_settings[i%THREAD_NUMBER].append(test_settings.loc[i].to_dict())
        for setting in thread_settings:
            x = threading.Thread(target=thread_function, args=(setting,))
            x.start()

        
