#!/usr/bin/env python

import simulate, analysis
from multiprocessing import Process

from settings import *

def main():

    Process(target=simulate.simulate, args=(f"run1",)).start()

# Make sure code only runs when the file is executed as a script.
if __name__ == "__main__":
    main()
