# Functions that define the physical model.

import numpy as np

from simulation_settings import *

# Physical constants:
ELEMENTARY_CHARGE = 1.6e-19
PLANCK_CONSTANT = 6.626e-34
HBAR = PLANCK_CONSTANT / (2*np.pi)
CONDUCTANCE_QUANTUM = 2*ELEMENTARY_CHARGE**2/PLANCK_CONSTANT
RESISTANCE_QUANTUM = 1/CONDUCTANCE_QUANTUM
DALTON = 1.66e-27
ELECTRON_MASS = 9.11e-31
SCHRODINGER_CONSTANT = 2*ELECTRON_MASS / HBAR**2

# Material constants:
RESISTIVITY = 53.4e-9 # Resistivity of the material
MATERIAL_CHARGE_DENSITY = 1e-3
TOUCHING_RESISTANCE = 1e-7
BREAKAWAY_ENERGY = 6.4e-16
BREAKAWAY_ENERGY *= 30 # this is a magic value, needs investigation
WORK_FUNCTION_MO = 7.44e-19 # 6.985~7.931; GM: 7.44, 4.65 eV
TUNNELING_SCALE = 1 / np.sqrt(SCHRODINGER_CONSTANT*WORK_FUNCTION_MO)

# might need to be an order of 36 higher at most; this would increase the speed at which the filaments form
MATERIAL_CHARGE_DENSITY = ELEMENTARY_CHARGE/DALTON # 1e8; not sure what to do with this

# Physical parameters.
SUBSTRATE_SIZE = .5e-6 # width/height of the substrate
PARTICLE_DIAMETER_MEAN = 20e-9 # 10 nm
PARTICLE_DIAMETER_STD = 1e-9 # 1 nm
BIAS = 2e-3 # 2mV; voltage over the source and drain
# TOUCHING_RESISTANCE = RESISTANCE_QUANTUM
# FILAMENT_TIMESCALE = 1e-19

# Simulation parameters
HIGH_RESISTANCE = 1e15 # very high resistance between the source and drain
MAX_CURRENT = BIAS / HIGH_RESISTANCE
MAX_DISTANCE = 10e-9 # Important for making sure we don't get a singular matrix.
MAX_RESISTANCE = 1e15

assert MAX_DISTANCE < SUBSTRATE_SIZE

print(TUNNELING_SCALE)

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
