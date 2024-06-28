# Functions that define the physical model.

import numpy as np

from settings import *

"""
Unfortunately, numpy arrays are not to keen on being thrown into if-statements.
As a result, we need to use binary arrays that we multiply with a number and then add the results to obtain an answer.
This works because boolean arrays will be treated as if they contain zeroes and ones.


For example, in stead of a program like:


if x < y:
    return a
if x > y:
    return b


we need to write

x_lt_y = x < y
x_gt_y = x > y
return a*x_lt_y + b*x_gt_y


"""

def resistance(distance, sum_of_radii, true_distances):

    touching_particle = true_distances < 0
    untouching_particle = true_distances > 0

    filament_formed = distance < 0
    filament_unformed = distance > 0

    # For seperated nanoparticles:
    overlap_resistance    = RESISTANCE_QUANTUM                                                         * filament_formed   * untouching_particle
    seperation_resistance = RESISTANCE_QUANTUM * np.exp(2*distance.astype(np.float32)/TUNNELING_SCALE) * filament_unformed * untouching_particle
    np.nan_to_num(seperation_resistance, copy=False, nan=MAX_RESISTANCE, posinf=MAX_RESISTANCE, neginf=None)

    # For touching nanoparticles:
    touching_resistance = MIN_RESISTANCE * touching_particle

    total_resistance = overlap_resistance + seperation_resistance +  touching_resistance

    return total_resistance * (total_resistance <= MAX_RESISTANCE) + MAX_RESISTANCE * (total_resistance > MAX_RESISTANCE)

    # return TOUCHING_RESISTANCE*np.exp(distance.astype(np.float32)*EXPONENTIAL_DECAY_CONSTANT)

def joule_heating(currents, resistances, distances, true_distances):

    is_filament = (distances < 0) * (true_distances > 0)

    power = currents**2 * resistances

    return (is_filament * power).astype(np.float32)

def filament_acceleration(distances, voltages):

    is_seperated = distances > 0

    field_strength = voltages / distances
    acceleration = field_strength * MATERIAL_CHARGE_DENSITY

    return (is_seperated * acceleration).astype(np.float32)

def break_filaments(distances, energies, true_distances):

    to_break = energies > BREAKAWAY_ENERGY
    not_to_break = energies < BREAKAWAY_ENERGY

    new_distances = to_break * true_distances + not_to_break * distances
    new_energies = energies * not_to_break

    return new_distances.astype(np.float32), new_energies.astype(np.float32)
