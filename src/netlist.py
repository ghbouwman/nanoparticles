from settings import BIAS, HIGH_RESISTANCE

import numpy as np

def construct_netlist(resistances, first_edges, second_edges, filename):

    size = resistances.size

    with open(filename, 'w') as f:

        for resistor_number, (resistance, i, j) in enumerate(zip(resistances, first_edges, second_edges)):

            # Convention for .net file. Insert resistance
            resistance_text = f"R{resistor_number} {i} {j} {resistance}\n"
            f.write(resistance_text) # Write resistance into file

        # For the source and drain:
        i = max(np.max(first_edges), np.max(second_edges))-1 # source
        j = i+1 # drain

        resistance_text = f"R{size} {i} {j} {HIGH_RESISTANCE}\n"
        f.write(resistance_text) # Write resistance into file

        voltage_text = f"V{size} {i} {j} {BIAS}\n"
        f.write(voltage_text) # Write voltage into file

        f.write('.op') # .op at the end for operating point circuit.
