import numpy as np


def geographic_to_cartesian(points):
    """ 
    Returns cartesian coordinates(x, y, z) from geographic coordinates (longitude and latitude)
    Input:
        (2, n) array like
    Returns:
        (3, n) numpy array
    """

    return np.array(
        [np.sin(points[:, 0]) * np.sin(points[:, 1]), np.cos(points[:, 0]) * np.sin(points[:, 1]),
         np.cos(points[:, 1])]).transpose()


def cartesian_to_geographic(points):
    """
    Returns geographic coordinates (longitude and latitude) from cartesian
    Input:
        (3, n) array like
    Output:
        (2, n) numpy array
    """

    return np.array([np.arctan2(points[:, 0], points[:, 1])+np.pi, np.arctan2(np.sqrt(np.square(points[:, 1])+ np.square(points[:, 0])), points[:, 2])]).transpose()


def pitch_yaw_rotation_matrix(x):
    """
    Computes the rotation matrix from a rotation vector.
    """

    return np.array([
        [np.cos(x[1]) * np.cos(x[0]), -np.sin(x[0]), np.sin(x[1]) * np.sin(x[0])],
        [np.cos(x[1]) * np.sin(x[0]), np.cos(x[0]), np.sin(x[1]) * np.cos(x[0])],
        [-np.sin(x[1]), 0, np.cos(x[1])]
    ])

def even_sphere_points(number):
    """
    Generates a set of n evenly spaced points on a sphere using the golden ratio.
    """

    indices = np.arange(0, number) + 0.5
    phi = np.arccos(1-2*indices/number)
    theta = np.pi * (1+ 5**0.5)* indices
    return np.array([np.cos(theta)*np.sin(phi), np.sin(theta)*np.sin(phi), np.cos(phi)]).transpose()