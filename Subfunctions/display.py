import numpy as np
import matplotlib.pyplot as plt
from Subfunctions.geometry import geographic_to_cartesian
from PIL import Image
import os

def display_layers(grid, path):
    """
    display a height, temperature, humidity, water map on a single figure, store image and data.
    Takes in the path to where the image should be stored
    """

    xx, yy = np.meshgrid(np.linspace(
        0, 2*np.pi, grid.resolution[0]), np.linspace(0, np.pi, grid.resolution[1]))
    print(np.shape(xx), np.shape(yy))
    indices = grid.grid_tree.query(geographic_to_cartesian(
        np.array([xx.flatten(), yy.flatten()]).transpose()))[1]
    fig, axs = plt.subplots(3, 2)
    display_sub_axes(axs[0, 0], xx, yy, grid.height[indices].reshape(grid.resolution), 'Height Map')
    display_sub_axes(axs[0, 1], xx, yy, grid.humidity[indices].reshape(grid.resolution), 'Humidity Map')
    display_sub_axes(axs[1, 0], xx, yy, grid.net_precipitation[indices].reshape(grid.resolution), 'Precipitation Map')
    display_sub_axes(axs[1, 1], xx, yy, grid.water_level[indices].reshape(grid.resolution), 'Water Map')
    display_sub_axes(axs[2, 0], xx, yy, grid.temperature[indices].reshape(grid.resolution), 'Temperature Map')
    display_sub_axes(axs[2, 1], xx, yy, grid.temperature[indices+grid.size].reshape(grid.resolution), 'High Alt Temperature Map')
    plt.savefig(
        os.path.join(path, f"test_{grid.settings['ID']}.png"))
    plt.show()


def display_biome(grid, biomes, path):
    """
    Generates a biome png from the biome grid.
    Takes a path for where to store the image.
    """

    xx, yy = np.meshgrid(np.linspace(
        0, 2*np.pi, grid.resolution[0]), np.linspace(0, np.pi, grid.resolution[1]))
    _, indices = grid.grid_tree.query(geographic_to_cartesian(
        np.array([xx.flatten(), yy.flatten()]).transpose()))
    colour_points = biomes.map_colours[grid.biome[indices].reshape(grid.resolution)].astype(np.uint8)
    data = Image.fromarray(colour_points, 'RGB')
    data.save(os.path.join(path, f"biomes_seed_{grid.settings['SEED']}.png"),"png")
    data.show()

def display_sub_axes(ax, xx, yy, data, title):
    """
    Subfunction for display_layers.
    Creates the contour plot for each data set.
    """

    print(np.min(data), np.max(data))
    ax.contourf(xx, yy, data)
    ax.set_title(title)

def store_data_template(data):
    """
    Creates a string to store the data.
    Used in the grid class.
    """

    string = ''
    for value in data:
        string += f',{np.min(value)},{np.max(value)}'
    return string