#!/usr/bin/env python

import simulate, analysis
from multiprocessing import Process

from settings import *

def main():

    for i in range(1, NR_RUNS+1):
        Process(target=simulate.simulate, args=(f"run_{i:02}",)).start()

# Make sure code only runs when the file is executed as a script.
if __name__ == "__main__":
    main()
