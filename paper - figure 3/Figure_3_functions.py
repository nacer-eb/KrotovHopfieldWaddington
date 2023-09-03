import sys
sys.path.append('../')

import numpy as np

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)

import matplotlib.pyplot as plt

from nullcline_gather.GatherNullClines import *


def load_data():
    # Fetch FPs data from C Code output
    data_dir = "data_2_2_2/"
    C_data_dir = "C_Code_FPs/"
    
    data = np.loadtxt(C_data_dir+"save0.dat", delimiter=",", skiprows=1)
    p_mask = (np.abs(data[:, -2]) > 0.001) * (np.abs(data[:, -3]) > 0.001 ) * (data[:, 1] <= 7) + (np.abs(data[:, -2]) > 0.03) * (np.abs(data[:, -3]) > 0.03 ) * (data[:, 1] > 7)
    data = data[p_mask, :]

    data_T = np.load(data_dir+"run_[1, 4]_n15_T700_alpha0.2_l_0-0.5.npz")['miniBatchs_images'][0]
    A, B = data_T[0], data_T[1]
    
    return data, A, B


def plot_nullclines(ax, n, temp, l_0_mesh, alpha_mesh, data):
   
    GNC = GatherNullClines(753, 494, 719, n, temp/(2.0**(1.0/n)), +1)  
    alpha_nullcline = GNC.alpha_nullcline(alpha_mesh, l_0_mesh)
    l_nullcline = GNC.l_0_nullcline(alpha_mesh, l_0_mesh)

    ax.contour(l_0_mesh, alpha_mesh, alpha_nullcline, [0], colors="purple", linewidths=4, alpha=0.5)
    ax.contour(l_0_mesh, alpha_mesh, l_nullcline, [0], colors="orange", linewidths=4, alpha=0.5)

    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    ax.text(0.07, 0.94, r"$n=$"+str(n), transform=ax.transAxes, fontsize=18, verticalalignment='top', horizontalalignment='left', bbox=props)

    # Fetching all FPs relevant to that n-value
    n_mask = data[:, 1] == n

    # Sort by l_0
    l_0s = data[n_mask, -1]
    index_sort = np.argsort(l_0s)

    # Then pick Leftmost, middle and rightmost
    l_0s = l_0s[index_sort][[0, np.sum(n_mask)//2, -1]]
    alphas = data[n_mask, -3][index_sort][[0, np.sum(n_mask)//2, -1]]
    betas = data[n_mask, -2][index_sort][[0, np.sum(n_mask)//2, -2]]
        
    # Plotting 3 FPs
    ax.scatter(l_0s, alphas, s=40, facecolor=['green', 'red', 'green'], edgecolor="k", linewidths=1, zorder=10)
    

    return alphas, betas


def plot_snapshot(ax, A, B, alpha, beta, isStable=True):
    sample_mem = alpha * A + beta * B
    ax.imshow(sample_mem.reshape(28, 28), cmap="bwr")

    ax.set_title("Stable", color="green", fontsize=14)
    if not isStable:
        ax.set_title("Unstable", color="red", fontsize=14)
