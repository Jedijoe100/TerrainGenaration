import Subfunctions.altitude_generation as Altitude
import Subfunctions.planet as Planet
import Subfunctions.temperature_generation as Temperature
import Subfunctions.precipitation as Precipitation
import numpy as np
import matplotlib.pyplot as plt

"""
Step 1: Research and Implement a Techtonic Plate System
Step 2: Add Random Noise
Check Point 1: Does this generate Ranges, Lakes, Islands and Continents? Can this be easily extended to 
Step 3: Generate
"""

class Generation:
    def __init__(self, seed, angle, distance=1, stellar_luminosity=1, alt_diff=20, sea_level=6,
                     ecentricity=1, shape='circle'):
        self.data = Planet.Planet(seed, angle, distance, stellar_luminosity, alt_diff, sea_level,
                     ecentricity, shape)

    def generate(self):
        altitude_data = Altitude.Altitude(dimensions=self.data.dimensions, seed = self.data.seed)
        altitude_data.perlin_noise()
        altitude_data.display(self.data)
        temperature_data = Temperature.Temperature()
        temperature_data.generate_heatmap(altitude_data, self.data)
        temperature_data.display()
        precipitation = Precipitation.Precipitation()
        pressure = precipitation.pressure_map(altitude_data, temperature_data, self.data)
        for value in pressure:
            plt.matshow(value)
            plt.show()
        precipitation.wind_map(pressure, self.data)



if __name__ == '__main__':
    test = Generation(1, np.pi/10, 1, 1)
    test.generate()