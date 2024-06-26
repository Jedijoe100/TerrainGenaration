import numpy as np
import matplotlib.pyplot as plt
from Subfunctions.geometry import geographic_to_cartesian
from PIL import Image
import os

def display_layers(self, path):
    # display a height, temperature, humidity, water map on a single figure, store image and data
    xx, yy = np.meshgrid(np.linspace(
        0, 2*np.pi, self.resolution[0]), np.linspace(0, np.pi, self.resolution[1]))
    print(np.shape(xx), np.shape(yy))
    indices = self.grid_tree.query(geographic_to_cartesian(
        np.array([xx.flatten(), yy.flatten()]).transpose()))[1]
    fig, axs = plt.subplots(3, 2)
    display_sub_axes(axs[0, 0], xx, yy, self.height[indices].reshape(self.resolution), 'Height Map')
    display_sub_axes(axs[0, 1], xx, yy, self.humidity[indices].reshape(self.resolution), 'Humidity Map')
    display_sub_axes(axs[1, 0], xx, yy, self.net_precipitation[indices].reshape(self.resolution), 'Precipitation Map')
    display_sub_axes(axs[1, 1], xx, yy, self.water_level[indices].reshape(self.resolution), 'Water Map')
    display_sub_axes(axs[2, 0], xx, yy, self.temperature[indices].reshape(self.resolution), 'Temperature Map')
    display_sub_axes(axs[2, 1], xx, yy, self.temperature[indices+self.size].reshape(self.resolution), 'High Alt Temperature Map')
    plt.savefig(
        os.path.join(path, f"test_{self.settings['ID']}.png"))
    plt.show()


def display_biome(self, biomes, path):
    xx, yy = np.meshgrid(np.linspace(
        0, 2*np.pi, self.resolution[0]), np.linspace(0, np.pi, self.resolution[1]))
    _, indices = self.grid_tree.query(geographic_to_cartesian(
        np.array([xx.flatten(), yy.flatten()]).transpose()))
    colour_points = biomes.map_colours[self.biome[indices].reshape(self.resolution)].astype(np.uint8)
    data = Image.fromarray(colour_points, 'RGB')
    data.save(os.path.join(path, f"biomes_seed_{self.settings['SEED']}.png"),"png")
    data.show()

def display_sub_axes(ax, xx, yy, data, title):
    print(np.min(data), np.max(data))
    ax.contourf(xx, yy, data)
    ax.set_title(title)

def store_data_template(data):
    string = ''
    for value in data:
        string += f',{np.min(value)},{np.max(value)}'
    return string