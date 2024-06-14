#!/usr/bin/env python

import simulate
# from multiprocessing import Process

def main():

    with open("../output/index.txt") as f:
        run_name = f.readline().strip()[0]
    
    simulate.simulate(run_name)

# Make sure code only runs when the file is executed as a script.
if __name__ == "__main__":
    main()
