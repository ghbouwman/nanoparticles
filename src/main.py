#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import linalg, physics, netlist, solver, plotting, preprocessing
from multiprocessing import Process
import time

from settings import *


def main():

    if PLOTTING:
        D = np.linspace(-physics.PARTICLE_DIAMETER_MEAN, physics.MAX_DISTANCE, 1000)
        R = physics.resistance(D, 2)
        plt.plot(D, R)
        plt.show()

    print("Initialising substrate...")
    
    x_positions, y_positions, diameters, cluster_ids = preprocessing.deposit_nanoparticles()
    if PLOTTING:
        plotting.plotting_nanoparticles(x_positions, y_positions, diameters, cluster_ids)

    print(f"amount of clusters: {1+np.max(cluster_ids):.0f}")

    # Compute the distance between all the centres of the nanoparticles.
    central_distances = linalg.distance_matrix(x_positions, y_positions)

    sum_of_radii = linalg.sum_of_radii_matrix(diameters)

    true_distances = central_distances - sum_of_radii # negative if there is overlap

    # Append the distances from both the electrodes
    true_distances = preprocessing.add_source_and_drain(true_distances, diameters, x_positions)

    # Remove all nodes that are too far away.
    true_distances, sum_of_radii, first_nodes, second_nodes = preprocessing.extract_nodes(true_distances, diameters)

    x_positions, y_positions, diameters = preprocessing.extract_positions_and_diameters(x_positions, y_positions, diameters, first_nodes, second_nodes)
    
    first_nodes, second_nodes = preprocessing.rename_nodes(first_nodes, second_nodes)

    size = true_distances.size

    print("no. edges:", size)
    print("no. particles", 1+max(np.max(first_nodes), np.max(second_nodes)))
    # print(first_nodes.size, x_positions.size)

    # Initalise the filament energies.
    energies = np.zeros(size)

    # Initialise filament velocities.
    velocities = np.zeros(size)

    # Set the initial distances.
    distances = true_distances

    plotting_centers, plotting_radii = preprocessing.compute_plotting_centers_and_radii(x_positions, y_positions, diameters)

    system_current = np.full(INDEX_MAX, np.nan)


    # Simulation loop
    solve_time = 0
    total_tic = time.time()
    T = np.linspace(0, INDEX_MAX*DELTA_T, INDEX_MAX)
    for index, t in zip(range(INDEX_MAX), T):

        total_toc = time.time()
        total_time = total_toc - total_tic
        print(f"real time elapsed: {total_time//3600:.0f}:{total_time//60 % 60:02.0f}:{total_time % 60:02.0f}s --- simulation time: {t:.6f}s ({100*index/INDEX_MAX:.0f}% done)", end='\r')

        resistances = physics.resistance(distances, sum_of_radii)
        netlist.construct_netlist(resistances, first_nodes, second_nodes, NETLIST_FILENAME)

        tic = time.time()
        voltages, currents = solver.solve_cicuit(f"../utils/{NETLIST_FILENAME}")
        toc = time.time()
        solve_time += toc-tic

        if PLOTTING:
            Process(target=plotting.plot_currents, args=(first_nodes, second_nodes, currents, plotting_centers, plotting_radii, index, t)).start()
            Process(target=plotting.plot_voltages, args=(first_nodes, second_nodes, voltages, plotting_centers, plotting_radii, index, t)).start()
            plotting.plot_currents(first_nodes, second_nodes, currents, plotting_centers, plotting_radii, index, t)
            plotting.plot_voltages(first_nodes, second_nodes, voltages, plotting_centers, plotting_radii, index, t)

        system_current[index] = currents[-1]

        energies += physics.joule_heating(currents, resistances, distances, true_distances) * DELTA_T
        distances, velocities = distances - velocities * DELTA_T, velocities + physics.filament_acceleration(distances, voltages) * DELTA_T
        distances, energies = physics.break_filaments(distances, energies, true_distances)


    total_toc = time.time()
    total_time = total_toc - total_tic

    print(f"real time elapsed: {total_time:.1f}s --- simulation time: {t:.6f}s ({100*index/INDEX_MAX:.0f}% done)")
    print("Finished simulation.")
    print(f"Time taken up by solver: {solve_time:.2f} ({100*solve_time/total_time:.1f}%)")

    if PLOTTING:
        plt.plot(1e3*T, np.abs(system_current))
        plt.xlabel("Time (ms)")
        plt.ylabel("Current (A)")
        plt.savefig("../plots/system_current")
        plt.close()

# Make sure code only runs when the file is executed as a script.
if __name__ == "__main__":
    main()
