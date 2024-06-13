#!/usr/bin/env python

import simulate, analysis
from multiprocessing import Process

from settings import *

def main():

    for i in range(NR_RUNS):
        Process(target=simulate.simulate, args=(f"run{i}",)).start()

    print("Started all runs.")

# Make sure code only runs when the file is executed as a script.
if __name__ == "__main__":
    main()
