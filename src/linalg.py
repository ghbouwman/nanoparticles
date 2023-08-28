# Some general linear algebra code used throughout the programme.

import numpy as np

def distance_matrix(x_position, y_position):

    size = x_position.size

    x_difference = np.tile(x_position, (size, 1))
    y_difference = np.tile(y_position, (size, 1))

    for index in range(size):

        x_difference[index, :] -= x_position[index]
        y_difference[index, :] -= y_position[index]

    return np.sqrt(x_difference**2 + y_difference**2)

def sum_of_radii_matrix(diameter):

    size = diameter.size

    first_diameter = np.tile(diameter, (size, 1))
    second_diameter = first_diameter.transpose()

    return first_diameter/2 + second_diameter/2