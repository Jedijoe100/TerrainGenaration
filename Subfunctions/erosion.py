from Subfunctions.weather import weather
import numpy as np


def erosion(self, time_of_year):
    # update erosion
    weather(self, time_of_year, False)
    self.water_level = self.water_level * \
        (self.height > self.settings['SEA_LEVEL'])
    for i in range(self.size):
        # find a close point that is lower than current point
        minimum_min = self.adjacent_points[i, np.argmin(
            self.height[self.adjacent_points[i, :]] + self.water_level[self.adjacent_points[i, :]])]
        height_difference = np.maximum(
            0, self.water_level[i] + self.height[i] - self.height[minimum_min] - self.water_level[minimum_min])/2
        # compute the water flow between the lower point and the point
        water_flow = np.minimum(self.water_level[i], height_difference)
        # Transfer the water and erode
        self.water_level[i] = np.maximum(
            self.water_level[i] - water_flow, 0)
        self.height[i] -= water_flow/self.settings['EROSION_FACTOR'] + \
            height_difference/self.settings['WIND_FACTOR']
        self.water_level[minimum_min] += (
            self.height[minimum_min] > self.settings['SEA_LEVEL'])*water_flow
        self.height[minimum_min] += water_flow / \
            self.settings['EROSION_FACTOR'] + \
            height_difference/self.settings['WIND_FACTOR']
