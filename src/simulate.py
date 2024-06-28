import numpy as np
import linalg, physics, netlist, solver, plotting, preprocessing, analysis
from multiprocessing import Process
import time, datetime
from log import Logger
from settings import *

def simulate(run_name):

    log = Logger(run_name)

    log("Initialising substrate...")
    
    x_positions, y_positions, diameters, cluster_ids = preprocessing.deposit_nanoparticles()

    plotting.plotting_nanoparticles(x_positions, y_positions, diameters, cluster_ids, run_name)

    log(f"amount of clusters: {1+np.max(cluster_ids):.0f}")

    # Compute the distance between all the centres of the nanoparticles.
    central_distances = linalg.distance_matrix(x_positions, y_positions)

    sum_of_radii = linalg.sum_of_radii_matrix(diameters)

    true_distances = central_distances - sum_of_radii # negative if there is overlap

    # Append the distances from both the electrodes
    true_distances = preprocessing.add_source_and_drain(true_distances, diameters, x_positions)

    log(f"{true_distances}")

    plotting.plot_distances(true_distances, run_name)

    # Remove all nodes that are too far away.
    log("Extracting nodes...")
    true_distances, sum_of_radii, first_nodes, second_nodes = preprocessing.extract_nodes(true_distances, diameters, run_name)

    x_positions, y_positions, diameters = preprocessing.extract_positions_and_diameters(x_positions, y_positions, diameters, first_nodes, second_nodes, run_name)
    
    first_nodes, second_nodes = preprocessing.rename_nodes(first_nodes, second_nodes, run_name)

    size = true_distances.size
    
    log(f"no. edges: {size}")
    log(f"no. particles {1+max(np.max(first_nodes), np.max(second_nodes))}")

    # Initalise values
    energies = np.zeros(size)
    velocities = np.zeros(size) # filament velocities.
    distances = true_distances

    plotting_centers, plotting_radii = preprocessing.compute_plotting_centers_and_radii(x_positions, y_positions, diameters)

    # Output file
    with open(f"../output/{run_name}.csv", 'w') as csv:
        csv.write("Time (s),Current (A)\n")

    # Simulation loop
    t = 0 # time in the simulation itself
    index = 0
    solve_time = 0
    total_tic = time.time()

    while index < NR_STEPS:

        total_toc = time.time()
        total_time = total_toc - total_tic
        log(f"real time elapsed: {datetime.timedelta(seconds=total_time)} ({100*index/NR_STEPS:.1f}% done) ETA: {datetime.timedelta(seconds=(total_time/(max(index, 1)/NR_STEPS) - total_time))}")

        # Calculate new resistances.
        resistances = physics.resistance(distances, sum_of_radii)
        netlist.construct_netlist(resistances, first_nodes, second_nodes, f"../output/{run_name}.net")

        # Solve for the new voltages and currents. (we also time this step)
        solve_tic = time.time()
        voltages, currents = solver.solve_cicuit(f"../output/{run_name}.net")
        solve_toc = time.time()
        solve_time += solve_toc - solve_tic

        if PLOTTING:
            Process(target=plotting.plot_currents, args=(first_nodes, second_nodes, currents, plotting_centers, plotting_radii, index, t, run_name)).start()
            Process(target=plotting.plot_voltages, args=(first_nodes, second_nodes, voltages, plotting_centers, plotting_radii, index, t, run_name)).start()
            # plotting.plot_currents(first_nodes, second_nodes, currents, plotting_centers, plotting_radii, index, t)
            # plotting.plot_voltages(first_nodes, second_nodes, voltages, plotting_centers, plotting_radii, index, t)

        # Save the system current to the .csv
        with open(f"../output/{run_name}.csv", 'a') as csv:
            csv.write(f"{t},{currents[-1]}\n")

        # Grown and break filaments
        energies += physics.joule_heating(currents, resistances, distances, true_distances) * DELTA_T
        distances, velocities = distances - velocities * DELTA_T, velocities + physics.filament_acceleration(distances, voltages) * DELTA_T
        distances, energies = physics.break_filaments(distances, energies, true_distances)
    
        # Increment the time and index
        t += DELTA_T
        index += 1

    total_toc = time.time()
    total_time = total_toc - total_tic
   
    log(f"real time elapsed: {total_time//3600:02.0f}:{total_time//60 % 60:02.0f}:{total_time % 60:02.0f}s (100% done)")
    log("Finished simulation.")
    log(f"Time taken up by solver: {solve_time//3600:02.0f}:{solve_time//60 % 60:02.0f}:{solve_time % 60:02.0f}s ({100*solve_time/total_time:.1f}%)")

    analysis.analyse(run_name)

