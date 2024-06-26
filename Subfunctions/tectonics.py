import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

def eruptions(self):
    # compute volcano spots
    _, indices = self.grid_tree.query(self.volcanism_spots)
    for index in indices:
        self.height[index] += self.rng.random() / \
            self.settings['VOLCANIC_FACTOR']
    # compute volcano chains
    self.volcanism = self.volcanism*self.rng.random(self.size)*2
    self.height += self.volcanism * \
        self.rng.random(self.size)*(self.volcanism >= 1) / \
        self.settings['VOLCANIC_FACTOR']
    self.volcanism = self.volcanism * (self.volcanism < 1)


def plate_reset(self):
    point_3d = self.rng.random((self.points, 3))*2 - 1
    # store these points in a lookup tree
    tree = KDTree(point_3d)
    grid_value = np.array(tree.query(self.grid)[1])
    average_plate_value = []
    for i in range(self.points):
        average_plate_value.append(np.average(
            self.plate_type[grid_value == i]))
    average_plate_value = np.array(average_plate_value)
    self.plate_type = average_plate_value[grid_value]
    velocity = self.rng.random((self.points, 3)) / \
        self.settings['PLATE_VELOCITY_FACTOR']
    self.velocity = []
    for i in grid_value:
        self.velocity.append(Rotation.from_rotvec(velocity[i]))


def move_plates(self):
    # Setup temporary storage
    next_step_height = np.zeros(self.size)
    next_step_velocity = self.velocity.copy()
    next_plate_step = -np.ones(self.size)
    for i in range(self.size):
        # compute the new location of points
        point = np.array([self.velocity[i].apply(self.grid[i])])
        # find index of new location
        _, new_index = self.grid_tree.query(point)
        # add height to new location
        next_step_height[new_index[0]] += self.height[i]
        # check if it is the dominant plate
        if next_plate_step[new_index[0]] < self.plate_type[i]:
            # transferring information
            next_step_velocity[new_index[0]] = self.velocity[i]
            next_plate_step[new_index[0]] = self.plate_type[i]
        elif self.plate_type[i] < 0 and next_plate_step[new_index[0]] > self.plate_type[i]:
            # to allow for simulated volcanism creating mountains and island chains
            self.volcanism[new_index[0]] += 0.1
            # to try and simulate trenches
            next_step_height[i] = 0
    # storing plate_type
    self.height = next_step_height
    self.velocity = next_step_velocity
    self.plate_type = next_plate_step
    # Ageing the plate
    self.plate_type += self.settings['PLATE_AGEING']
