import pandas as pd
import numpy as np
import os
from PIL import Image, ImageColor

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
BIOME_PATH = 'biomes.csv'
IMAGE_DIRECTORY = '../Test_images'

HEIGHT_CATEGORIES = 7

class Biomes:
    def __init__(self) -> None:
        biomes = np.array(pd.read_csv(os.path.join(FILE_PATH, BIOME_PATH)))
        self.biomes = biomes
        self.minecraft_biomes = list(set(biomes[:,3]))
        self.biome_correspond = np.array([self.minecraft_biomes.index(value) for value in biomes[:, 3]])
        self.altitude_category = biomes[:,4:6]
        self.water_minimum = biomes[:,6]
        self.biome_vector = biomes[:,6:]
        self.map_colours = np.array([ImageColor.getcolor(biome, 'RGB') for biome in biomes[:,2]])
    
    def biome_test(self, alt_category, element_vector):
        alt_satisfied = np.array([np.where(self.altitude_category[:,0] <= element, 1, 0)*np.where(element <= self.altitude_category[:,1], 1, 0) for element in alt_category]).transpose()
        water_satisfied = np.array([self.water_minimum <= element[0] for element in element_vector]).transpose()
        fix_vector = np.matmul(self.biome_vector, element_vector.transpose())*alt_satisfied*water_satisfied
        return np.array([np.argmax(element) for element in fix_vector.transpose()])
    




