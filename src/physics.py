# Functions that define the physical model.

# Physical parameters.
SUBSTRATE_SIZE = 1e-6 # width/height of the substrate
PARTICLE_DIAMETER_MEAN = 20e-9 # 10 nm
PARTICLE_DIAMETER_STD = 1e-9 # 1 nm
HIGH_RESISTANCE = 1e15 # very high resistance between the source and drain
BIAS = 2e-3 # 2mV; voltage over the source and drain
TOUCHING_RESISTANCE = 1e-7
EXPONENTIAL_DECAY_CONSTANT = 1e9 # 1/m # quantum tunneling effect or something
RESISTIVITY = 53.4e-9 # Resistivity of the material
MATERIAL_CHARGE_DENSITY = 1e-3
BREAKAWAY_ENERGY = 6.4e-16
BREAKAWAY_ENERGY *= 30
MAX_DISTANCE = 10e-9

ELEMENTARY_CHARGE = 1.6e-19
PLANCK_CONSTANT = 6.626e-34
CONDUCTANCE_QUANTUM = 2*ELEMENTARY_CHARGE**2/PLANCK_CONSTANT
RESISTANCE_QUANTUM = 1/CONDUCTANCE_QUANTUM
TOUCHING_RESISTANCE = RESISTANCE_QUANTUM
DALTON = 1.66e-27
MATERIAL_CHARGE_DENSITY = ELEMENTARY_CHARGE/DALTON # 1e8
MAX_CURRENT = BIAS / HIGH_RESISTANCE
FILAMENT_TIMESCALE = 1e-19

assert MAX_DISTANCE < SUBSTRATE_SIZE

import numpy as np

def resistance(distance, sum_of_radii):

    is_overlapping = distance < 0
    overlap_resistance = is_overlapping/(-distance/RESISTIVITY+CONDUCTANCE_QUANTUM)
    

    is_seperated = distance > 0
    seperation_resistance = is_seperated * RESISTANCE_QUANTUM*np.exp(distance.astype(np.float32)*EXPONENTIAL_DECAY_CONSTANT)

    return overlap_resistance + seperation_resistance

    # return TOUCHING_RESISTANCE*np.exp(distance.astype(np.float32)*EXPONENTIAL_DECAY_CONSTANT)

def joule_heating(currents, resistances, distances, true_distances):

    is_filament = (distances < 0) * (true_distances > 0)

    power = currents**2 * resistances

    return (is_filament * power).astype(np.float32)

def grow_filaments(distances, voltages):

    is_seperated = distances > 0

    field_strength = voltages/distances
    grow_speed = field_strength * MATERIAL_CHARGE_DENSITY * FILAMENT_TIMESCALE

    return (is_seperated * grow_speed).astype(np.float32)

def break_filaments(distances, energies, true_distances):

    to_break = energies > BREAKAWAY_ENERGY
    not_to_break = energies < BREAKAWAY_ENERGY

    new_distances = to_break * true_distances + not_to_break * distances
    new_energies = energies * not_to_break

    return new_distances.astype(np.float32), new_energies.astype(np.float32)