import spicepy.netlist
import spicepy.netsolve

def solve_cicuit(filename):

    # Import network
    net = spicepy.netlist.Network(filename)

    # Solving the circuit
    spicepy.netsolve.net_solve(net)
    net.branch_voltage()
    net.branch_current()

    # Extract voltages
    voltages = net.vb

    # Extract currents
    currents = net.ib

    # Get rid of the final edge (source to drain)
    return voltages[:-2], currents[:-2]
