import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from physics import SUBSTRATE_SIZE

IMAGE_SIZE = 500
image_nps = np.zeros(shape = (IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.int32)

def plotting_nanoparticles(x_positions, y_positions, diameters, cluster_ids):
    '''
    This method plots the nanoparticles paired with colorcoding per
    cluster.

    Returns
    -------
    A figure of the nanoparticles if desired.

    '''
    # Scaling meters to pixels
    m2pix = IMAGE_SIZE / SUBSTRATE_SIZE

    rng = np.random.default_rng()

    # Drawing circles
    clusters = np.unique(cluster_ids.flatten())
    for cluster in clusters:
        x_poss = x_positions[cluster_ids == cluster]
        y_poss = y_positions[cluster_ids == cluster]
        diams = diameters[cluster_ids == cluster]
        col = np.hstack((rng.uniform(0.1, 0.9, size=3)))*255
        for x_pos, y_pos, diam in zip(x_poss, y_poss, diams):
            center_particle = (int(x_pos*m2pix), int(y_pos*m2pix))

            cv.circle(image_nps,
                        center = center_particle,
                        radius = int(diam/2*m2pix),
                        color = col,
                        thickness = -1)

    img = plt.imshow(image_nps)
    plt.axis('off')
    print("Generating plot...")
    plt.savefig("../plots/cluster_plot")
    print("Saved figure.")
    plt.close()

def plot_currents(first_nodes, second_nodes, currents, centers, radii, index, time):
    
    size = radii.size

    values = np.zeros(size)

    for first, second, curr in zip(first_nodes, second_nodes, currents):

        if first >= size or second >= size:
            continue

        values[first] += abs(curr)
        values[second] += abs(curr) # not needed I think?

    values /= 2

    # Normalise for colors.
    colors = values / np.amax(values) # maybe use a max consistent over time?

    # Drawing circles
    for center, radius, color in zip(centers, radii, colors):

        color = list(plt.cm.plasma(color))
        for i in range(4):
            color[i] *= 256

        color = tuple(color)
        
        cv.circle(image_nps, center=center, radius=radius, color=color, thickness=-1)

    plt.figure()
    plt.imshow(image_nps, cmap = plt.cm.plasma)
    plt.colorbar(label='Current (A)')
    plt.clim(0, 5e-6)
    plt.axis('off')
    plt.title(f"t={1e3*time:.3f}ms")
    plt.savefig(f"../figures/current{index:04}")
    plt.close()

def plot_voltages(first_nodes, second_nodes, voltages, centers, radii, index, time):
    
    size = radii.size

    values = np.zeros(size)

    for first, second, volt in zip(first_nodes, second_nodes, voltages):

        if first >= size or second >= size:
            continue

        values[first] += abs(volt)
        values[second] += abs(volt) # not needed I think?

    values /= 2

    # Normalise for colors.
    colors = values / np.amax(values) # maybe use a max consistent over time?

    # Drawing circles
    for center, radius, color in zip(centers, radii, colors):

        color = list(plt.cm.plasma(color))
        for i in range(4):
            color[i] *= 256

        color = tuple(color)
        
        cv.circle(image_nps, center=center, radius=radius, color=color, thickness=-1)

    plt.figure()
    plt.imshow(image_nps, cmap = plt.cm.plasma)
    plt.colorbar(label='Voltage (V)')
    plt.clim(0, 5e-6)
    plt.axis('off')
    plt.title(f"t={1e3*time:.3f}ms")
    plt.savefig(f"../figures/voltage{index:04}")
    plt.close()
