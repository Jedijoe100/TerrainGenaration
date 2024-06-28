import numpy as np
from scipy.spatial.transform import Rotation
from Subfunctions.geometry import geographic_to_cartesian


def Wind_Map(grid):
    """Computes the wind map over the grid."""

    bottom_layer_lat = grid.settings['WIND_SPEED']*(
        np.ceil(np.sin(grid.settings['CELL_NUMBER']*grid.grid_2d[:, 1]*np.pi))*2-1)
    top_layer_lat = -bottom_layer_lat
    bottom_long = bottom_layer_lat * \
        grid.settings['SPIN']*np.sin(grid.grid_2d[:, 1])
    top_long = top_layer_lat*grid.settings['SPIN']*np.sin(grid.grid_2d[:, 1])
    new_index = np.zeros(2*grid.size)
    point_density = np.sqrt(4*np.pi/grid.settings['DETAIL'])
    altitude_change_points = np.linspace(
        0, 1, 2*int(grid.settings['CELL_NUMBER'])+1)
    rise_grid = []
    fall_grid = []
    for n in range(2*int(grid.settings['CELL_NUMBER'])+1):
        point_number = int(
            1 + 2*np.pi*np.sin(altitude_change_points[n]*np.pi)/point_density)
        coords = np.ones((point_number*2, 2))*altitude_change_points[n]*np.pi
        coords[:point_number, 1] = altitude_change_points[n] * \
            np.pi + point_density/2
        coords[point_number:, 1] = altitude_change_points[n] * \
            np.pi - point_density/2
        coords[:point_number, 0] = np.linspace(0, 2*np.pi, point_number)
        coords[point_number:, 0] = np.linspace(0, 2*np.pi, point_number)
        coords_3d = geographic_to_cartesian(coords)
        if n % 2 == int(grid.settings['CELL_NUMBER']) % 2:
            fall_grid += list(np.unique(grid.grid_tree.query(coords_3d)[1]))
        else:
            rise_grid += list(np.unique(grid.grid_tree.query(coords_3d)[1]))
    rise_grid = np.array(rise_grid)
    fall_grid = np.array(fall_grid)
    for i in range(grid.size):
        wind_matrix = Rotation.from_rotvec(
            (bottom_layer_lat[i], bottom_long[i], 0))
        new_location = wind_matrix.apply(grid.grid[i])
        new_index[i] = grid.grid_tree.query(new_location)[1]
    for i in range(grid.size):
        wind_matrix = Rotation.from_rotvec((top_layer_lat[i], top_long[i], 0))
        new_location = wind_matrix.apply(grid.grid[i])
        new_index[i +
                  grid.size] = grid.grid_tree.query(new_location)[1] + grid.size
    new_index[rise_grid] = rise_grid + grid.size
    new_index[fall_grid+grid.size] = fall_grid
    return new_index.astype(int)


def weather(grid, time_of_year, final):
    """
    Computes the weather for a given time of year.
    Final allows for the data to be stored for biome checking.
    """

    # update temperature
    temperature_update(grid, time_of_year)
    # update humidity
    grid.humidity[:grid.size] += (grid.height <= grid.settings['SEA_LEVEL']) / \
        grid.settings['HUMIDITY_FACTOR'] + \
        grid.water_level/grid.settings['HUMIDITY_FACTOR']
    grid.water_level -= grid.water_level/grid.settings['HUMIDITY_FACTOR']
    # Transferring temperature to new point and humidity to new point (modifying temperature if going up or downhill)
    height_diff = np.zeros(grid.size*2)
    height_diff[:grid.size] = grid.height[grid.weather_index[:grid.size] %
                                          grid.size] - grid.height
    new_temperature = grid.temperature[grid.weather_index] - \
        height_diff/grid.settings['ALTITUDE_FACTOR']
    new_humidity = grid.humidity[grid.weather_index]
    # computing precipitation
    precipitation = np.maximum(
        0, new_humidity - (np.minimum(100, np.maximum(0, new_temperature - grid.settings['HEIGHT_TEMP_DROP']*grid.layer_2)))/100)
    new_humidity -= precipitation * (precipitation > 0)
    net_precipitation = precipitation[:grid.size] + precipitation[grid.size:]
    grid.water_level += net_precipitation * \
        (net_precipitation > 0) * (grid.height > grid.water_level)
    # radiate temperature
    new_temperature -= ((new_humidity*100/(np.minimum(100,
                         np.maximum(0.0001, new_temperature)))) < grid.settings['CLOUD_POINT']) * grid.settings['TEMPERATURE_RADIATE']
    # blurring humidity and temperature
    tem_temperature = new_temperature*grid.settings['BLUR_FACTOR']
    tem_humidity = new_humidity*grid.settings['BLUR_FACTOR']
    tem_precipitation = net_precipitation*grid.settings['BLUR_FACTOR']
    for i in range(grid.size):
        tem_temperature[grid.adjacent_points[i]] += new_temperature[i] * \
            (1-grid.settings['BLUR_FACTOR'])/grid.settings['ADJACENT_FACTOR']
        tem_temperature[grid.adjacent_points[i]+grid.size] += new_temperature[i +
                                                                              grid.size]*(1-grid.settings['BLUR_FACTOR'])/grid.settings['ADJACENT_FACTOR']
        tem_humidity[grid.adjacent_points[i]] += new_humidity[i] * \
            (1-grid.settings['BLUR_FACTOR'])/grid.settings['ADJACENT_FACTOR']
        tem_humidity[grid.adjacent_points[i]+grid.size] += new_humidity[i+grid.size] * \
            (1-grid.settings['BLUR_FACTOR'])/grid.settings['ADJACENT_FACTOR']
        tem_precipitation[grid.adjacent_points[i]] += net_precipitation[i] * \
            (1-grid.settings['BLUR_FACTOR'])/grid.settings['ADJACENT_FACTOR']
    net_precipitation = tem_precipitation
    grid.temperature = tem_temperature
    grid.humidity = tem_humidity
    grid.net_precipitation += net_precipitation
    grid.weather_steps += 1
    # store temperature data
    if final:
        precipitation_max = np.max(net_precipitation)+0.0000000000001
        temp_categories = np.array(
            [-1000, -10, 0, 10, 20, 30, 40, 50, 60, 10000])
        humidity_categories = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
        temperature_values = 5+np.digitize(grid.temperature, temp_categories)
        precipitation_values = np.digitize(
            net_precipitation/precipitation_max, humidity_categories)
        for i in range(grid.size):
            grid.biome_blocks[i, int(temperature_values[i])] += 1
            grid.biome_blocks[i, int(precipitation_values[i])] += 1


def temperature_update(grid, time_of_year):
    """
    Calculates the temperature from the star at a given time of year and at each point on the grid.
    """

    Planet_angle = 2*np.pi*grid.settings['ANGLE_FACTOR']
    year_progress = time_of_year * 2 * np.pi
    Distance = grid.settings['SEMI_MAJOR_AXIS']*(1-np.square(grid.settings['ECCENTRICITY']))/(
        1 + grid.settings['ECCENTRICITY']*np.cos(year_progress))
    # this is the angle the current tilt of the planet wrt the sun
    angle_between_sun = Planet_angle * np.cos(year_progress)
    # this computes the amount of time in the sun
    proportion_day = 1/2 - \
        np.sin(grid.grid_2d[:, 1])*np.tan(angle_between_sun) / \
        (2*np.sin(grid.grid_2d[:, 1]))
    # energy at each point
    solar_energy = np.sin(np.minimum(np.maximum(
        grid.grid_2d[:, 1] - angle_between_sun, 0), np.pi))*grid.settings['SOLAR_ENERGY']*proportion_day/(4*np.pi*np.square(Distance))
    grid.temperature[:grid.size] += (((grid.humidity[:grid.size]*100/(np.minimum(100,
                                                                                 np.maximum(0.0001, grid.temperature[:grid.size])))) < grid.settings['CLOUD_POINT'])+1)*solar_energy/2
