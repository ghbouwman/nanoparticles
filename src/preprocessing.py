from physics import SUBSTRATE_SIZE, PARTICLE_DIAMETER_MEAN, PARTICLE_DIAMETER_STD, MAX_DISTANCE

import numpy as np

from settings import MAX_PARTICLES


from physics import SUBSTRATE_SIZE
from plotting import IMAGE_SIZE

# Scaling meters to pixes
M2PIX = IMAGE_SIZE / SUBSTRATE_SIZE

def deposit_nanoparticles():
    '''
    Method to 'deposit' nanoparticles. This is done by giving them a random
    location within the sample, paired with a randomly distributed diameter
    around the given diameter.
    The nanoparticles are also inserted into clusters if they are directly
    in contact with eachother.

    Credits for inspiration: Jesse Luchtenveld

    Returns
    -------
    None.

    '''
    x_positions = np.zeros(MAX_PARTICLES)
    y_positions = np.zeros(MAX_PARTICLES)
    diameters = np.zeros(MAX_PARTICLES)
    cluster_ids = np.zeros(MAX_PARTICLES)

    index = 0
    cluster_id = 0

    rng = np.random.default_rng()

    connected = False
    while not connected:

        # Pos_x
        x_positions[index] = rng.uniform(
            low = 0,
            high = SUBSTRATE_SIZE)

        # Pos_y
        y_positions[index] = rng.uniform(
            low = 0,
            high = SUBSTRATE_SIZE)

        # Diameter
        diameters[index] = rng.normal(PARTICLE_DIAMETER_MEAN, PARTICLE_DIAMETER_STD)

        # Cluster id
        cluster_ids[index] = cluster_id

        # New particle and all previous particles
        new_x_position = x_positions[index]
        new_y_position = y_positions[index]
        new_diameter = diameters[index]

        old_x_positions = x_positions[:index]
        old_y_positions = y_positions[:index]
        old_diameters = diameters[:index]
        old_cluster_ids = cluster_ids[:index]

        # Create mask for all particles touching new_particle
        # Done via calculating distance between all old particles
        # and the new particle via np.hypot(x,y). Then comparing
        # to radius new + radius old
        mask = (np.hypot(old_x_positions - new_x_position,
                        old_y_positions - new_y_position)
                <= old_diameters/2 + new_diameter/2)

        # Group cluster_ids in contact - basically setting all
        # touching nanoparticle cluster ids equal.
        clusters = np.unique(old_cluster_ids[mask].flatten())

        # Update self.particles - mask array where only cluster_id
        # is in clusters.
        mask = np.zeros(MAX_PARTICLES, dtype=bool)
        mask[:index] = np.isin(old_cluster_ids, clusters)
        cluster_ids[mask] = cluster_id

        # Now check if the new cluster connects to left and right.
        connected = np.any(x_positions[mask] <= diameters[mask]/2) and \
                    np.any(x_positions[mask] >= SUBSTRATE_SIZE - diameters[mask]/2)

        # Update number of particles and cluster id
        index += 1
        cluster_id += 1

    # Remove empty rows
    cut = index
    
    return x_positions[:cut], y_positions[:cut], diameters[:cut], cluster_ids[:cut]
    
def add_source_and_drain(true_distances, diameters, x_positions):

    left_electrode_distance = x_positions - diameters
    right_electrode_distance = (SUBSTRATE_SIZE - x_positions) - diameters

    edge_distances = np.vstack([left_electrode_distance, right_electrode_distance])

    bottom_right = np.array([[0, SUBSTRATE_SIZE],[SUBSTRATE_SIZE, 0]])

    true_distances = np.block([
        [true_distances, edge_distances.transpose()],
        [edge_distances, bottom_right]
    ])

    return true_distances

def extract_nodes(distances, diameters):

    size = distances.shape[0]

    edge_value = np.full(size**2, None)
    radii_value = np.full(size**2, None)
    first_node = np.full(size**2, None)
    second_node = np.full(size**2, None)

    index = 0
    iteration = 0
    for (i, j), distance in np.ndenumerate(distances):

        progress = 100 * iteration / distances.size
        print(f"Extracting nodes: {progress:.0f}%", end='\r')

        if distance <= MAX_DISTANCE:

            sum_of_radii = 0

            if i >= j:
                continue

            if i == size-2:
                i = "source"

            elif i == size-1:
                i = "drain"

            else:
                sum_of_radii += diameters[i]/2

            if j == size-2:
                j = "source"

            elif j == size-1:
                j = "drain"

            else:
                sum_of_radii += diameters[j]/2

            edge_value[index] = distance
            radii_value[index] = sum_of_radii
            first_node[index] = i
            second_node[index] = j

            index += 1

        iteration += 1

    cut = index

    print("Extracting nodes: finished.")

    return edge_value[:cut], radii_value[:cut], first_node[:cut], second_node[:cut]

def extract_positions_and_diameters(x_positions, y_positions, diameters, first_nodes, second_nodes):

    nodes = np.hstack([first_nodes, second_nodes])

    new_x_positions = np.empty(MAX_PARTICLES)
    new_y_positions = np.empty(MAX_PARTICLES)
    new_diameters = np.empty(MAX_PARTICLES)

    index = 0
    for i in range(MAX_PARTICLES):

        progress = 100 * i/MAX_PARTICLES
        print(f"Extracting positions and diameters: {progress:.0f}%", end='\r')

        for node in nodes:

            if node == i:

                new_x_positions[index] = x_positions[i]
                new_y_positions[index] = y_positions[i]
                new_diameters[index] = diameters[i]
                index += 1
                break

    print("Extracting positions and diameters: finished.")

    a = 0.2 # parameter free to be set; only used for plotting

    new_x_positions[index] = -a
    new_y_positions[index] = .5
    new_diameters[index] = 0
    index += 1

    new_x_positions[index] = 1+a
    new_y_positions[index] = .5
    new_diameters[index] = 0
    index += 1

    cut = index

    return new_x_positions[:cut], new_y_positions[:cut], new_diameters[:cut]


def rename_nodes(first_edges, second_edges):

    size = first_edges.size
    

    first_edges[first_edges == "source"] = MAX_PARTICLES
    first_edges[first_edges == "drain"] = MAX_PARTICLES + 1
    second_edges[second_edges == "source"] = MAX_PARTICLES
    second_edges[second_edges == "drain"] = MAX_PARTICLES + 1

    max_index = max(
        np.max(first_edges[first_edges < MAX_PARTICLES]), 
        np.max(second_edges[second_edges < MAX_PARTICLES])
    )

    first_edges[first_edges == MAX_PARTICLES] = max_index + 1
    first_edges[first_edges == MAX_PARTICLES + 1] = max_index + 2
    second_edges[second_edges == MAX_PARTICLES] = max_index + 1
    second_edges[second_edges == MAX_PARTICLES + 1] = max_index + 2

    max_index += 2

    i = 0
    while i <= max_index:

        upper_bound = max([np.max(first_edges), np.max(second_edges)]) # goes from max_index to i

        progress = 100*i/max_index

        print(f"Renaming edges: {progress:.0f}%", end='\r')

        if np.max(first_edges) < i and np.max(second_edges) < i:

            break

        if i in first_edges or i in second_edges:

            i += 1
        
        else:
            first_edges[first_edges > i] -= 1
            second_edges[second_edges > i] -= 1

    print(f"Renaming edges: finished.")

    return first_edges, second_edges

def compute_plotting_centers_and_radii(x_positions, y_positions, diameters):

    size = x_positions.size
    plotting_centers = np.empty((size, 2), dtype=int)
    plotting_radii = np.empty(size, dtype=int)

    for index, (x_pos, y_pos, diam) in enumerate(zip(x_positions, y_positions, diameters)):

        plotting_centers[index] = np.array([int(x_pos * M2PIX), int(y_pos * M2PIX)])
        plotting_radii[index] = int(M2PIX * diam/2)


    return plotting_centers[:-2], plotting_radii[:-2]

