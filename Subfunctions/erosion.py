from Subfunctions.weather import weather
import numpy as np


def erosion(grid, time_of_year):
    """
    Function that erodes the landscape.
    Takes in a time of year for the weather.
    """
    weather(grid, time_of_year, False)
    grid.water_level = grid.water_level * \
        (grid.height > grid.settings['SEA_LEVEL'])
    for i in range(grid.size):
        # find a close point that is lower than current point
        minimum_min = grid.adjacent_points[i, np.argmin(
            grid.height[grid.adjacent_points[i, :]] + grid.water_level[grid.adjacent_points[i, :]])]
        height_difference = np.maximum(
            0, grid.water_level[i] + grid.height[i] - grid.height[minimum_min] - grid.water_level[minimum_min])/2
        # compute the water flow between the lower point and the point
        water_flow = np.minimum(grid.water_level[i], height_difference)
        # Transfer the water and erode
        grid.water_level[i] = np.maximum(
            grid.water_level[i] - water_flow, 0)
        grid.height[i] -= water_flow/grid.settings['EROSION_FACTOR'] + \
            height_difference/grid.settings['WIND_FACTOR']
        grid.water_level[minimum_min] += (
            grid.height[minimum_min] > grid.settings['SEA_LEVEL'])*water_flow
        grid.height[minimum_min] += water_flow / \
            grid.settings['EROSION_FACTOR'] + \
            height_difference/grid.settings['WIND_FACTOR']
