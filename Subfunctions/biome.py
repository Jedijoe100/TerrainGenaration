import pandas as pd
import numpy as np
import os
from PIL import Image, ImageColor
from Subfunctions.geometry import geographic_to_cartesian


HEIGHT_CATEGORIES = 7

def biomes(self, biomes):
    alt_categories = np.array([0, 0.3, 0.6, 0.8, 0.95, 1, 1.5, 10000])
    alt_value = np.arange(0, 7, 1)
    alt_categories[:-1] *= self.settings['SEA_LEVEL']
    # select minimum non negative height category
    alt_value = np.digitize(self.height, alt_categories)
    self.biome_blocks /= self.settings['WEATHER_STEPS']
    self.biome_blocks[:,0] = self.water_level
    self.biome = biomes.biome_test(alt_value, self.biome_blocks)
    print(biomes.biomes[np.unique(self.biome), 1])

class Biomes:
    def __init__(self, biome_path) -> None:
        biomes = np.array(pd.read_csv(biome_path))
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
    




