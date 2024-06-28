import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

def eruptions(grid):
    """
    Computing the volcanic eruptions from techtonics and hot spots
    """

    # compute volcano spots
    _, indices = grid.grid_tree.query(grid.volcanism_spots)
    for index in indices:
        grid.height[index] += grid.rng.random() / \
            grid.settings['VOLCANIC_FACTOR']
    # compute volcano chains
    grid.volcanism = grid.volcanism*grid.rng.random(grid.size)*2
    grid.height += grid.volcanism * \
        grid.rng.random(grid.size)*(grid.volcanism >= 1) / \
        grid.settings['VOLCANIC_FACTOR']
    grid.volcanism = grid.volcanism * (grid.volcanism < 1)


def plate_reset(grid):
    """
    Resets the plates.
    I.e. generate a new set of plates.
    """

    point_3d = grid.rng.random((grid.points, 3))*2 - 1
    # store these points in a lookup tree
    tree = KDTree(point_3d)
    grid_value = np.array(tree.query(grid.grid)[1])
    average_plate_value = []
    for i in range(grid.points):
        average_plate_value.append(np.average(
            grid.plate_type[grid_value == i]))
    average_plate_value = np.array(average_plate_value)
    grid.plate_type = average_plate_value[grid_value]
    velocity = grid.rng.random((grid.points, 3)) / \
        grid.settings['PLATE_VELOCITY_FACTOR']
    grid.velocity = []
    for i in grid_value:
        grid.velocity.append(Rotation.from_rotvec(velocity[i]))


def move_plates(grid):
    """
    Moves the plates.
    """

    # Setup temporary storage
    next_step_height = np.zeros(grid.size)
    next_step_velocity = grid.velocity.copy()
    next_plate_step = -np.ones(grid.size)
    for i in range(grid.size):
        # compute the new location of points
        point = np.array([grid.velocity[i].apply(grid.grid[i])])
        # find index of new location
        _, new_index = grid.grid_tree.query(point)
        # add height to new location
        next_step_height[new_index[0]] += grid.height[i]
        # check if it is the dominant plate
        if next_plate_step[new_index[0]] < grid.plate_type[i]:
            # transferring information
            next_step_velocity[new_index[0]] = grid.velocity[i]
            next_plate_step[new_index[0]] = grid.plate_type[i]
        elif grid.plate_type[i] < 0 and next_plate_step[new_index[0]] > grid.plate_type[i]:
            # to allow for simulated volcanism creating mountains and island chains
            grid.volcanism[new_index[0]] += 0.1
            # to try and simulate trenches
            next_step_height[i] = 0
    # storing plate_type
    grid.height = next_step_height
    grid.velocity = next_step_velocity
    grid.plate_type = next_plate_step
    # Ageing the plate
    grid.plate_type += grid.settings['PLATE_AGEING']
