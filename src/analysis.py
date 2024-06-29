import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from settings import *
from scipy import optimize

def autocorr(x):
    
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]

def acf(x, length):

    return np.array([1] + [np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, length)])

def analyse(run_name):

    path = "output"

    df = pd.read_csv(f"../{path}/{run_name}.csv")

    cut = 1
    T = np.array(df["Time (s)"][cut:])
    I = np.array(np.abs(df["Current (A)"])[cut:])
    # I = running_mode(I)

    G = I / BIAS
    N = G / CONDUCTANCE_QUANTUM

    plt.xlabel("Time (s)")
    plt.ylabel("Conductance ($G_0$)")

    # cut outliers
    # normal_vals = (N < 1e-4) 
    # N = N * normal_vals
    
    plt.scatter(T, N, s=2)
    # plt.xlim(0.6e-9, 0.7e-9)
    plt.savefig(f"../{path}/{run_name}_sys.png")
    plt.close()

    l = len(N)
    A = autocorr(N)
    A = acf(N, l)
    # A = acf(np.sin(np.linspace(0, 20, len(A))))
    print(A)
    plt.scatter(T[:l],A)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-3, 1)
    plt.savefig(f"../{path}/{run_name}_autocorr.png")
    plt.close()


    # delta G
    
    dG = G[1:] - G[:-1]
    dN = dG / CONDUCTANCE_QUANTUM
    T_half = (T[1:] + T[:-1])/2
    
    plt.xlabel("Time (s)")
    plt.ylabel("Conductance Change ($G_0$)")
    plt.scatter(T_half, dN, s=2)
    plt.savefig(f"../{path}/{run_name}_sys_change.png")
    plt.close()

    # delta delta G
    
    ddG = dG[1:] - dG[:-1]
    ddN = ddG / CONDUCTANCE_QUANTUM
    T_half_half = (T_half[1:] + T_half[:-1])/2
    
    plt.xlabel("Time (s)")
    plt.ylabel("Conductance Second Difference ($G_0$)")
    plt.scatter(T_half_half, ddN, s=2)
    plt.savefig(f"../{path}/{run_name}_sys_change_change.png")
    plt.close()

    val = 400
    res = 5
    bins = np.linspace(-val, val, round(2*val/res))
    bins = 200
    densities, edges, _ = plt.hist(dN, bins=bins, density=True)
    midpoints = (edges[1:] + edges[:-1])/2
    pseudo_voigt = lambda x, mu, sigma, gamma, a: a*np.exp(-((x-mu)/(sqrt(2)*sigma))**2)/(sigma*sqrt(2*np.pi)) + (1-a)/(np.pi*gamma*(1+((x-mu)/gamma)**2))
    max_val_gauss = 1e-2
    max_val_cauchy = 1e-2
    a_guess = .8
    popt, pcov, *_ = optimize.curve_fit(pseudo_voigt, midpoints, densities,
                                        p0=[0, 1/(max_val_gauss*np.sqrt(2*np.pi)), 1/(max_val_cauchy*np.pi), a_guess],
                                        bounds=([-np.inf, 0, 0, 0], [np.inf, np.inf, np.inf, 1]),
                                        maxfev=10_000)
    perr = np.sqrt(np.diag(pcov))
    f = lambda x: pseudo_voigt(x, *popt)
    lim = max(-min(dN), max(dN))
    X = np.linspace(-lim, lim, 1_000)
    Y = f(X)
    plt.scatter(X, Y, s=.5, c='orange')
    plt.xlabel("Conductance Change ($G_0$)")
    plt.xlim(-lim, lim)
    plt.ylabel("Empirical Probability Density")
    plt.savefig(f"../{path}/{run_name}_hist.png")
    plt.close()

    return (popt, perr)

def analyse_set():

    mus = []
    sigmas = []
    gammas = []
    ayys = []

def analyse_julien(run_name, val=1000, res=3):

    df = pd.read_csv(f"../results/julien/{run_name}.csv", sep=', ')
    
    V = np.array(df["vol"])
    I = np.array(df["cur"])
    R = np.array(df["res"])
    T = np.array(df["time"])

    T /= 3600 # Convert to hours

    G = 1 / R
    N = G / CONDUCTANCE_QUANTUM

    plt.xlabel("Time (h)")
    plt.ylabel("Conductance (m$G_0$)")
    plt.ylim(149, 152.5)
    plt.scatter(T, 1000*N, s=1e-4)
    plt.savefig(f"../results/julien/{run_name}.png")
    plt.close()

    plt.clf()
    
    # delta G
    dG = G[1:] - G[:-1]
    dN = dG / CONDUCTANCE_QUANTUM
    T_half = (T[1:] + T[:-1])/2

    plt.xlabel("Time (h)")
    plt.ylabel("Absolute Conductance Change ($\mu G_0$)")
    plt.ylim(1e-1, 1e3)
    plt.yscale('log')
    plt.scatter(T_half, 1e6*abs(dN), s=1e-4)
    plt.savefig(f"../results/julien/{run_name}_log_abs_change.png")
    plt.close()

    """
    guesses = np.linspace(0.20, 0.23, 1000)
    mads = np.empty(guesses.size)

    for index, guess in enumerate(guesses):
        guess_array = np.full(dG.size, guess)
        remainders = np.mod(1e6*dG, guess_array)
        ads = 0.5*guess_array - np.abs(remainders - 0.5*guess_array)
        mads[index] = np.sum(ads) / dG.size

    print(mads) 
    mads_percent_max = 100 * mads / (0.5*guesses)


    def autocorr(array):
        result = np.correlate(array, array, mode='full')
        return result[result.size//2]

    plt.scatter(guesses, 1e6*mads, s=.5)
    plt.xlabel("Effective Conductance Change Quantum ($\mu G_0$)")
    plt.ylabel("Mean Absolute Deviation ($\mu G_0$)")
    plt.savefig(f"../results/julien/{run_name}_effective_G_quantum.png")
    plt.close()
    """

    plt.xlabel("Time (h)")
    plt.ylabel("Conductance Change ($\mu G_0$)")
    plt.ylim(-1.6, 1.6)
    plt.scatter(T_half, 1e6*dN, s=1e-2)
    plt.savefig(f"../results/julien/{run_name}_change.png")
    plt.close()
    '''
    plt.hist(dN, bins=5_000, range=band_range, density=True)
    plt.scatter(X, f(X), s=1, c="orange")
    plt.xscale('symlog', linthresh=1e-7)
    plt.xlabel("Conductance Change ($G_0$)")
    plt.yscale('log')
    plt.ylim(1, 30000)
    plt.savefig(f"../results/julien/{run_name}_hist_log_log.png")
    plt.close()

    plt.hist(dN, bins=5_000, range=band_range, density=True)
    plt.scatter(X, f(X), s=1, c="orange")
    plt.xscale('symlog', linthresh=1e-7)
    plt.xlabel("Conductance Change ($G_0$)")
    plt.savefig(f"../results/julien/{run_name}_hist_logx.png")
    plt.close()
    '''

    bins = np.linspace(-val, val, round(2*val/res))
    densities, edges, _ = plt.hist(1e6*dN, bins=bins, density=True)
    midpoints = (edges[1:] + edges[:-1])/2
    pseudo_voigt = lambda x, sigma, gamma, a: a*np.exp(-(x/(sqrt(2)*sigma))**2)/(sigma*sqrt(2*np.pi)) + (1-a)/(np.pi*gamma*(1+(x/gamma)**2))
    max_val_gauss = 1e-2
    max_val_cauchy = 1e-2
    a_guess = .8
    popt, pcov, *_ = optimize.curve_fit(pseudo_voigt, midpoints, densities, p0=[1/(max_val_gauss*np.sqrt(2*np.pi)), 1/(max_val_cauchy*np.pi), a_guess])
    print(popt, pcov)
    f = lambda x: pseudo_voigt(x, *popt)
    lim = 200
    X = np.linspace(-lim, lim, 1_000)
    Y = f(X)
    plt.scatter(X, Y, s=.5, c='orange')
    plt.xlabel("Conductance Change ($\mu G_0$)")
    plt.xlim(-lim, lim)
    plt.ylabel("Empirical Probability Density")
    plt.savefig(f"../results/julien/{run_name}_hist.png")
    plt.close()
    
    """
    plt.hist(1e6*dN, bins=bins, density=True)
    plt.scatter(X, Y, s=.5, c='orange')
    plt.xlabel("Conductance Change ($\mu G_0$)")
    lim = 200
    plt.xlim(-lim, lim)
    plt.ylabel("Emperical Probability Density")
    plt.yscale('log')
    plt.ylim(1e-5, 1e-2)
    plt.savefig(f"../results/julien/{run_name}_hist_log.png")
    plt.close()
    """

    # plt.xlabel("Time (h)")
    # plt.ylabel("Conductance Change ($G_0$)")
    # plt.xlim(begin, end)
    # plt.ylim(bandsize/10, bandsize/2)
    # plt.scatter(T_half, abs(dN), s=1)
    # plt.savefig(f"../results/julien/{run_name}_abs_cut.png")
    # plt.close()

    '''
    plt.xlabel("Time (h)")
    plt.ylabel("Conductance Change ($G_0$)")
    plt.xlim(begin, end)
    plt.ylim(1e-5, 1e-4)
    plt.yscale('log')
    plt.scatter(T_half, dN, s=s)
    plt.savefig(f"../results/julien/{run_name}_e-5_to_e-4.png")
    plt.close()

    plt.xlabel("Time (h)")
    plt.ylabel("Conductance Change ($G_0$)")
    plt.xlim(begin, end)
    plt.ylim(-bandsize/2, bandsize/2)
    plt.yscale('symlog', linthresh=1e-7)
    plt.scatter(T_half, dN, s=s)
    plt.savefig(f"../results/julien/{run_name}_symlog_all.png")
    plt.close()

    plt.xlabel("Time (h)")
    plt.ylabel("Conductance Change ($G_0$)")
    plt.xlim(begin, end)
    plt.ylim(-bandsize/2, bandsize/2)
    plt.yscale('symlog', linthresh=bandsize/10)
    plt.scatter(T_half, dN, s=s)
    plt.savefig(f"../results/julien/{run_name}_symlog_big.png")
    plt.close()
    '''

def smooth(array):
    ones = np.ones(n)
    return np.convolve(array, ones, 'same')

def running_mode(array, n=1):

    modes = np.empty_like(array)

    for index in range(array.size):
        mode, count = stats.mode(array[index-n:index+n+1])
        if index < n or index+n > array.size-1 or count <= 2:
            mode = None
        modes[index] = mode

    return modes

