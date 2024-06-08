#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import linalg, physics, netlist, solver, plotting, preprocessing, simulate, analysis
from multiprocessing import Process
import time

from settings import *

def main():

    simulate.simulate("run1")
    analysis.analyse("run1")

# Make sure code only runs when the file is executed as a script.
if __name__ == "__main__":
    main()
