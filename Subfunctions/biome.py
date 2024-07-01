import pandas as pd
import numpy as np
import os
from PIL import Image, ImageColor
from Subfunctions.geometry import geographic_to_cartesian


HEIGHT_CATEGORIES = 7

def biomes(grid, biomes):
    """
    Sets the biome ids for each point on the grid.
    Requires a biomes object.
    """

    alt_categories = np.array([0, 0.3, 0.6, 0.8, 0.95, 1, 1.5, 10000])
    alt_categories[:-1] *= grid.settings['SEA_LEVEL']
    # select minimum non negative height category
    alt_value = np.digitize(grid.height, alt_categories)
    grid.biome_blocks /= grid.settings['WEATHER_STEPS']
    grid.biome_blocks[:,0] = grid.water_level
    grid.biome = biomes.biome_test(alt_value, grid.biome_blocks)
    print(biomes.biomes[np.unique(grid.biome), 1])

class Biomes:
    """
    The Biomes object stores the information of the biomes.
    It also provides a test to work out which biome best fits given the test vector.
    """

    def __init__(self, biome_path) -> None:
        """
        Processes the biomes ready for use.
        Takes a path to the biome.csv file.
        """

        biomes = np.array(pd.read_csv(biome_path))
        self.biomes = biomes
        self.minecraft_biomes = list(set(biomes[:,3]))
        self.biome_correspond = np.array([self.minecraft_biomes.index(value) for value in biomes[:, 3]])
        self.altitude_category = biomes[:,4:6]
        self.water_minimum = biomes[:,6]
        self.biome_vector = biomes[:,6:]
        self.map_colours = np.array([ImageColor.getcolor(biome, 'RGB') for biome in biomes[:,2]])
    
    def biome_test(self, alt_category, element_vector):
        """
        Returns the best biome id for each point.
        Accepts an array of points.
        """
        print(alt_category)
        alt_satisfied = np.array([np.where(self.altitude_category[:,0] <= element, 1, 0)*np.where(element <= self.altitude_category[:,1], 1, 0) for element in alt_category]).transpose()
        #for some reason we are getting ocean in the mountains.
        print(alt_satisfied[:, 1])
        water_satisfied = np.array([self.water_minimum <= element[0] for element in element_vector]).transpose()
        fix_vector = np.matmul(self.biome_vector, element_vector.transpose())*alt_satisfied*water_satisfied
        print(fix_vector[:, 1])
        return np.array([np.argmax(element) for element in fix_vector.transpose()])
    




