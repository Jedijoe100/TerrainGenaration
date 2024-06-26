import numpy as np
from scipy.spatial.transform import Rotation
from Subfunctions.geometry import geographic_to_cartesian


def Wind_Map(self):
    bottom_layer_lat = self.settings['WIND_SPEED']*(
        np.ceil(np.sin(self.settings['CELL_NUMBER']*self.grid_2d[:, 1]*np.pi))*2-1)
    top_layer_lat = -bottom_layer_lat
    bottom_long = bottom_layer_lat * \
        self.settings['SPIN']*np.sin(self.grid_2d[:, 1])
    top_long = top_layer_lat*self.settings['SPIN']*np.sin(self.grid_2d[:, 1])
    new_index = np.zeros(2*self.size)
    point_density = np.sqrt(4*np.pi/self.settings['DETAIL'])
    altitude_change_points = np.linspace(
        0, 1, 2*int(self.settings['CELL_NUMBER'])+1)
    rise_grid = []
    fall_grid = []
    for n in range(2*int(self.settings['CELL_NUMBER'])+1):
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
        if n % 2 == int(self.settings['CELL_NUMBER']) % 2:
            fall_grid += list(np.unique(self.grid_tree.query(coords_3d)[1]))
        else:
            rise_grid += list(np.unique(self.grid_tree.query(coords_3d)[1]))
    rise_grid = np.array(rise_grid)
    fall_grid = np.array(fall_grid)
    for i in range(self.size):
        wind_matrix = Rotation.from_rotvec(
            (bottom_layer_lat[i], bottom_long[i], 0))
        new_location = wind_matrix.apply(self.grid[i])
        new_index[i] = self.grid_tree.query(new_location)[1]
    for i in range(self.size):
        wind_matrix = Rotation.from_rotvec((top_layer_lat[i], top_long[i], 0))
        new_location = wind_matrix.apply(self.grid[i])
        new_index[i +
                  self.size] = self.grid_tree.query(new_location)[1] + self.size
    new_index[rise_grid] = rise_grid + self.size
    new_index[fall_grid+self.size] = fall_grid
    return new_index.astype(int)


def weather(self, time_of_year, final):
    # update temperature
    temperature_update(self, time_of_year)
    # update humidity
    self.humidity[:self.size] += (self.height <= self.settings['SEA_LEVEL']) / \
        self.settings['HUMIDITY_FACTOR'] + \
        self.water_level/self.settings['HUMIDITY_FACTOR']
    self.water_level -= self.water_level/self.settings['HUMIDITY_FACTOR']
    # Transferring temperature to new point and humidity to new point (modifying temperature if going up or downhill)
    height_diff = np.zeros(self.size*2)
    height_diff[:self.size] = self.height[self.weather_index[:self.size] %
                                          self.size] - self.height
    new_temperature = self.temperature[self.weather_index] - \
        height_diff/self.settings['ALTITUDE_FACTOR']
    new_humidity = self.humidity[self.weather_index]
    # computing precipitation
    precipitation = np.maximum(
        0, new_humidity - (np.minimum(100, np.maximum(0, new_temperature - self.settings['HEIGHT_TEMP_DROP']*self.layer_2)))/100)
    new_humidity -= precipitation * (precipitation > 0)
    net_precipitation = precipitation[:self.size] + precipitation[self.size:]
    self.water_level += net_precipitation * \
        (net_precipitation > 0) * (self.height > self.water_level)
    # radiate temperature
    new_temperature -= ((new_humidity*100/(np.minimum(100,
                         np.maximum(0.0001, new_temperature)))) < self.settings['CLOUD_POINT']) * self.settings['TEMPERATURE_RADIATE']
    # blurring humidity and temperature
    tem_temperature = new_temperature*self.settings['BLUR_FACTOR']
    tem_humidity = new_humidity*self.settings['BLUR_FACTOR']
    tem_precipitation = net_precipitation*self.settings['BLUR_FACTOR']
    for i in range(self.size):
        tem_temperature[self.adjacent_points[i]] += new_temperature[i] * \
            (1-self.settings['BLUR_FACTOR'])/self.settings['ADJACENT_FACTOR']
        tem_temperature[self.adjacent_points[i]+self.size] += new_temperature[i +
                                                                              self.size]*(1-self.settings['BLUR_FACTOR'])/self.settings['ADJACENT_FACTOR']
        tem_humidity[self.adjacent_points[i]] += new_humidity[i] * \
            (1-self.settings['BLUR_FACTOR'])/self.settings['ADJACENT_FACTOR']
        tem_humidity[self.adjacent_points[i]+self.size] += new_humidity[i+self.size] * \
            (1-self.settings['BLUR_FACTOR'])/self.settings['ADJACENT_FACTOR']
        tem_precipitation[self.adjacent_points[i]] += net_precipitation[i] * \
            (1-self.settings['BLUR_FACTOR'])/self.settings['ADJACENT_FACTOR']
    net_precipitation = tem_precipitation
    self.temperature = tem_temperature
    self.humidity = tem_humidity
    self.net_precipitation += net_precipitation
    self.weather_steps += 1
    # store temperature data
    if final:
        precipitation_max = np.max(net_precipitation)+0.0000000000001
        temp_categories = np.array(
            [-1000, -10, 0, 10, 20, 30, 40, 50, 60, 10000])
        humidity_categories = np.array([0, 0.2, 0.4, 0.6, 0.8, 1])
        temperature_values = 5+np.digitize(self.temperature, temp_categories)
        precipitation_values = np.digitize(
            net_precipitation/precipitation_max, humidity_categories)
        for i in range(self.size):
            self.biome_blocks[i, int(temperature_values[i])] += 1
            self.biome_blocks[i, int(precipitation_values[i])] += 1


def temperature_update(self, time_of_year):
    """
    Input:
    Planet Angle
    Planet Distance
    """
    Planet_angle = 2*np.pi*self.settings['ANGLE_FACTOR']
    year_progress = time_of_year * 2 * np.pi
    Distance = self.settings['SEMI_MAJOR_AXIS']*(1-np.square(self.settings['ECCENTRICITY']))/(
        1 + self.settings['ECCENTRICITY']*np.cos(year_progress))
    # this is the angle the current tilt of the planet wrt the sun
    angle_between_sun = Planet_angle * np.cos(year_progress)
    # this computes the amount of time in the sun
    proportion_day = 1/2 - \
        np.sin(self.grid_2d[:, 1])*np.tan(angle_between_sun) / \
        (2*np.sin(self.grid_2d[:, 1]))
    # energy at each point
    solar_energy = np.sin(np.minimum(np.maximum(
        self.grid_2d[:, 1] - angle_between_sun, 0), np.pi))*self.settings['SOLAR_ENERGY']*proportion_day/(4*np.pi*np.square(Distance))
    self.temperature[:self.size] += (((self.humidity[:self.size]*100/(np.minimum(100,
                                                                                 np.maximum(0.0001, self.temperature[:self.size])))) < self.settings['CLOUD_POINT'])+1)*solar_energy/2
