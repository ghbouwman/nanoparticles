#!/usr/bin/env python

import simulate
# from multiprocessing import Process
from index import RUN_NAME

def main():

    simulate.simulate(RUN_NAME)

# Make sure code only runs when the file is executed as a script.
if __name__ == "__main__":
    main()
