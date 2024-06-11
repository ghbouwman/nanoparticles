import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def analyse(run_name):

    df = pd.read_csv(f"../output/{run_name}.csv")

    T = df["Time (s)"]
    I = np.abs(df["Current (A)"])

    plt.xlabel("Time (ns)")
    plt.ylabel("Current (nA)")
    # plt.yscale('log')
    plt.tight_layout()

    plt.scatter(1e9*T, 1e9*I, s=2)

    plt.savefig(f"../output/{run_name}.png")
    plt.close()

