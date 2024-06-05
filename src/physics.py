# Functions that define the physical model.

import numpy as np

from settings import *

def branchless_min(a, b):
    
    return a * (a <= b) + b * (b < a)

def branchless_max(a, b):
    
    return a * (a >= b) + b * (b > a)

def resistance(distance, sum_of_radii):

    is_overlapping = distance < 0
    is_seperated = distance > 0

    overlap = -distance

    overlap_resistance = is_overlapping * RESISTANCE_QUANTUM

    seperation_resistance = is_seperated * RESISTANCE_QUANTUM*np.exp(2*distance.astype(np.float32)/TUNNELING_SCALE)
    
    np.nan_to_num(seperation_resistance, copy=False, nan=MAX_RESISTANCE, posinf=MAX_RESISTANCE, neginf=None)

    total_resistance = overlap_resistance + seperation_resistance

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
