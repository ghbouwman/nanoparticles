from numpy import pi as PI
from numpy import sqrt

# All numbers are in SI unless mentioned otherwise

# Simulation parameters
HIGH_RESISTANCE = 1e30 # very high resistance between the source and drain
MAX_DISTANCE = 10e-9 # Important for making sure we don't get a singular matrix.
MAX_RESISTANCE = 1e15 # max resistance allowed in the distance computation
MAX_PARTICLES = 20_000 # high numbers wil cause large memory to be used

# np.log(np.finfo(np.float32))

NR_STEPS = 100 # Number of iterations in the simulation loop
DELTA_T = 1e-12 # Simulation timestep

# Plotting
IMAGE_SIZE = 500
PLOTTING = False
if PLOTTING:
    assert INDEX_MAX <= 1000 # GIF converter cannot handle too many images in memory
PRINTING = True

# Misc. simulation settings
NETLIST_FILENAME = "resistance_circuit"

# Physical constants:
ELEMENTARY_CHARGE = 1.6e-19
PLANCK_CONSTANT = 6.626e-34
HBAR = PLANCK_CONSTANT / (2*PI)
CONDUCTANCE_QUANTUM = 2*ELEMENTARY_CHARGE**2/PLANCK_CONSTANT
RESISTANCE_QUANTUM = 1/CONDUCTANCE_QUANTUM
DALTON = 1.66e-27
ELECTRON_MASS = 9.11e-31
SCHRODINGER_CONSTANT = 2*ELECTRON_MASS / HBAR**2

# Material constants for molybdenum:
MO_MASS = 95.95 * DALTON
MO_CHARGE = 1 * ELEMENTARY_CHARGE
# RESISTIVITY = 53.4e-9 # Temperature dependent?
# MATERIAL_CHARGE_DENSITY = 1e-3
# TOUCHING_RESISTANCE = 1e-7
BREAKAWAY_ENERGY = 6.4e-16
# BREAKAWAY_ENERGY *= 30 # this is a magic value, needs investigation
WORK_FUNCTION_MO = 7.44e-19 # 6.985~7.931; GM: 7.44, 4.65 eV
TUNNELING_SCALE = 1 / sqrt(SCHRODINGER_CONSTANT*WORK_FUNCTION_MO)
MATERIAL_CHARGE_DENSITY = MO_CHARGE/MO_MASS # 1e8; not sure what to do with this; might need to be an order of 36 higher at most; this would increase the speed at which the filaments form

# Physical parameters.
SUBSTRATE_SIZE = .1e-6 # width/height of the substrate
PARTICLE_DIAMETER_MEAN = 20e-9 # 10 nm
PARTICLE_DIAMETER_STD = 1e-9 # 1 nm
BIAS = 2e-3 # 2mV; voltage over the source and drain

MAX_CURRENT = BIAS / HIGH_RESISTANCE
assert MAX_DISTANCE < SUBSTRATE_SIZE
