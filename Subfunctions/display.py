import numpy as np


def display_sub_axes(ax, xx, yy, data, title):
    print(np.min(data), np.max(data))
    ax.contourf(xx, yy, data)
    ax.set_title(title)

def store_data_template(data):
    string = ''
    for value in data:
        string += f',{np.min(value)},{np.max(value)}'
    return string