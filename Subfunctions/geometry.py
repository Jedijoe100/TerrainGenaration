import numpy as np


def geographic_to_cartesian(points):
    return np.array(
        [np.sin(points[:, 0]) * np.sin(points[:, 1]), np.cos(points[:, 0]) * np.sin(points[:, 1]),
         np.cos(points[:, 1])]).transpose()


def cartesian_to_geographic(points):
    """
    Input:
    - points n X 3 matrix of points to convert
    Output:
    - n X 2 matrix in domain (0, 2\pi), ()
    """
    return np.array([np.arctan2(points[:, 0], points[:, 1])+np.pi, np.arctan2(np.sqrt(np.square(points[:, 1])+ np.square(points[:, 0])), points[:, 2])]).transpose()


def pitch_yaw_rotation_matrix(x):
    return np.array([
        [np.cos(x[1]) * np.cos(x[0]), -np.sin(x[0]), np.sin(x[1]) * np.sin(x[0])],
        [np.cos(x[1]) * np.sin(x[0]), np.cos(x[0]), np.sin(x[1]) * np.cos(x[0])],
        [-np.sin(x[1]), 0, np.cos(x[1])]
    ])

def even_sphere_points(number):
    indices = np.arange(0, number) + 0.5
    phi = np.arccos(1-2*indices/number)
    theta = np.pi * (1+ 5**0.5)* indices
    return np.array([np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)]).transpose()