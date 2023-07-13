import sys
sys.path.append('../')

import numpy as np

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)

import matplotlib.pyplot as plt
import matplotlib.animation as anim

from nullcline_gather.GatherNullClines import *

data_dir = "data_2_2_2/"
C_data_dir = "C_Code_FPs/"

data = np.loadtxt(C_data_dir+"save0.dat", delimiter=",", skiprows=1)
p_mask = (np.abs(data[:, -2]) > 0.001) * (np.abs(data[:, -3]) > 0.001 ) * (data[:, 1] <= 7) + (np.abs(data[:, -2]) > 0.03) * (np.abs(data[:, -3]) > 0.03 ) * (data[:, 1] > 7)
data = data[p_mask, :]

data_T = np.load(data_dir+"run_[1, 4]_n15_T700_alpha0.2_l_0-0.5.npz")['miniBatchs_images'][0]
A, B = data_T[0], data_T[1]


fig = plt.figure()
ax = fig.subplot_mosaic(
    """
    .AABBCC...DDEEFF...GGHHII...JJKKLL.
    .AABBCC...DDEEFF...GGHHII...JJKKLL.
    .MMMMMM...NNNNNN...OOOOOO...PPPPPP.
    .MMMMMM...NNNNNN...OOOOOO...PPPPPP.
    .MMMMMM...NNNNNN...OOOOOO...PPPPPP.
    .MMMMMM...NNNNNN...OOOOOO...PPPPPP.
    .MMMMMM...NNNNNN...OOOOOO...PPPPPP.
    .MMMMMM...NNNNNN...OOOOOO...PPPPPP.
    """
)


nullcline_axes = np.asarray([ax['M'], ax['N'], ax['O'], ax['P']])
for i in range(1, len(nullcline_axes)):
    nullcline_axes[i].set_yticks([]);
ax['M'].set_xlabel(r'$l_0$'); ax['N'].set_xlabel(r'$l_0$'); ax['O'].set_xlabel(r'$l_0$'); ax['P'].set_xlabel(r'$l_0$'); 
ax['M'].set_ylabel(r'$\alpha$')


temp = 700

alpha_range = np.linspace(0, 1, 1000); l_range = np.linspace(-1, 1, 1000)
l_0_mesh, alpha_mesh = np.meshgrid(l_range, alpha_range)

n_range = np.asarray([5, 10, 15, 20]) + 40 
for n_i, n in enumerate(n_range):
    GNC = GatherNullClines(753, 494, 719, n, temp/(2.0**(1.0/n)), +1)  
    alpha_nullcline = GNC.alpha_nullcline(alpha_mesh, l_0_mesh)
    l_nullcline = GNC.l_0_nullcline(alpha_mesh, l_0_mesh)
    
    nullcline_axes[n_i].contour(l_0_mesh, alpha_mesh, alpha_nullcline, [0], colors="purple", linewidths=4, alpha=0.5)
    nullcline_axes[n_i].contour(l_0_mesh, alpha_mesh, l_nullcline, [0], colors="orange", linewidths=4, alpha=0.5)

    n_mask = data[:, 1] == n
    nullcline_axes[n_i].scatter(data[n_mask, -1][[0, np.sum(n_mask)//2, -1]], data[n_mask, -3][[0, np.sum(n_mask)//2, -1]], s=7, color="k", zorder=10)
    
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)

    nullcline_axes[n_i].text(0.07, 0.94, r"$n=$"+str(n), transform=nullcline_axes[n_i].transAxes, fontsize=18, verticalalignment='top', bbox=props)
    

mem_snap_axes = np.asarray([ax['A'], ax['B'], ax['C'],
                           ax['D'], ax['E'], ax['F'],
                           ax['G'], ax['H'], ax['I'],
                           ax['J'], ax['K'], ax['L']])


for i, ax in enumerate(mem_snap_axes):
    ax.set_xticks([]); ax.set_yticks([]);

    n = n_range[i//3]
    p = i%3

    n_mask = data[:, 1] == n
    alphas, betas = data[n_mask, -3][[0, np.sum(n_mask)//2, -1]], data[n_mask, -2][[0, np.sum(n_mask)//2, -1]]
    sample_mem = alphas[p] * A + betas[p] * B
    
    ax.imshow(sample_mem.reshape(28, 28), cmap="bwr")

plt.subplots_adjust(top=0.65, bottom=0.34, left=0.15, right=0.85, hspace=0.05, wspace=0.05) 
plt.show()

