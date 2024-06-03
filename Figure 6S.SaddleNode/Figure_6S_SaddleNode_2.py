import sys
sys.path.append('../')

import numpy as np

import matplotlib
fontsize=35
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : fontsize}
matplotlib.rc('font', **font)

import matplotlib.pyplot as plt
import matplotlib.animation as anim

from nullcline_gather.GatherNullClines import *


# E here (and in general in this repo) is simply a reference to the cost function.
def E(l_0, alpha, n, temp):
    d_AA, d_AB, d_BB = 753, 494, 719

    l_0 = np.clip(l_0, -1, 1)
    alpha = np.clip(alpha, -1, 1)
    
    beta=1-np.abs(alpha)
    
    M_dot_A = (alpha * d_AA   +   beta * d_AB)
    M_dot_B = (alpha * d_AB   +   beta * d_BB)

    l_A_o_A = np.tanh(l_0 * (M_dot_A/temp)**n)
    l_A_o_B = np.tanh(l_0 * (M_dot_B/temp)**n)

    l_gamma_o_A = -np.tanh((M_dot_A/temp)**n)
    l_gamma_o_B = -np.tanh((M_dot_B/temp)**n)

    E = 2 * np.abs(1 - l_A_o_A)**(2*n) + 2 * np.abs(1 + l_A_o_B)**(2*n) + 8 * np.abs(1 + l_gamma_o_A)**(2*n) + 8 * np.abs(1 + l_gamma_o_B)**(2*n)

    return E



data_dir = "../Figure 6/data/"
C_data_dir = "C_Code_FPs/"
dataset = "../defaults/miniBatchs_images_Fig_6.npy"

data = np.loadtxt(C_data_dir+"low.dat", delimiter=",", skiprows=1)
p_mask = (np.abs(data[:, -2]) > 0.001) * (np.abs(data[:, -3]) > 0.001 ) * (data[:, 1] <= 7) + (np.abs(data[:, -2]) > 0.03) * (np.abs(data[:, -3]) > 0.03 ) * (data[:, 1] > 7)
data = data[p_mask, :]

data_T = np.load(dataset)[0]
A, B = data_T[0], data_T[1]

fig = plt.figure(figsize=(35, 70-54+1))



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
    ...................................
    ...................................
    ...................................
    .QQQQQQ...RRRRRR...SSSSSS...TTTTTT.
    .QQQQQQ...RRRRRR...SSSSSS...TTTTTT.
    .QQQQQQ...RRRRRR...SSSSSS...TTTTTT.
    .QQQQQQ...RRRRRR...SSSSSS...TTTTTT.
    .QQQQQQ...RRRRRR...SSSSSS...TTTTTT.
    .QQQQQQ...RRRRRR...SSSSSS...TTTTTT.
"""
)


nullcline_axes = np.asarray([ax['M'], ax['N'], ax['O'], ax['P']])
for i in range(1, len(nullcline_axes)):
    nullcline_axes[i].set_yticks([]);
ax['M'].set_xlabel(r'$\ell$'); ax['N'].set_xlabel(r'$\ell$'); ax['O'].set_xlabel(r'$\ell$'); ax['P'].set_xlabel(r'$\ell$');

ax['Q'].set_xlabel(r'$\ell$', labelpad=10); ax['R'].set_xlabel(r'$\ell$', labelpad=10); ax['S'].set_xlabel(r'$\ell$', labelpad=10); ax['T'].set_xlabel(r'$\ell$', labelpad=10); 
ax['M'].set_ylabel(r'$\alpha$')

E_axes = np.asarray([ax['Q'], ax['R'], ax['S'], ax['T']])
ax['Q'].xaxis.tick_top(); ax['R'].xaxis.tick_top(); ax['S'].xaxis.tick_top(); ax['T'].xaxis.tick_top()
ax['Q'].xaxis.set_label_position('top'); ax['R'].xaxis.set_label_position('top'); ax['S'].xaxis.set_label_position('top'); ax['T'].xaxis.set_label_position('top') 

ax['Q'].set_ylabel(r'log$(C)$', labelpad=10); ax['R'].set_ylabel(r'log$(C)$', labelpad=10); ax['S'].set_ylabel(r'log$(C)$', labelpad=10); ax['T'].set_ylabel(r'log$(C)$', labelpad=10)


temp = 700

alpha_range = np.linspace(0, 1, 1000); l_range = np.linspace(-1, 1, 1000)
l_0_mesh, alpha_mesh = np.meshgrid(l_range, alpha_range)

n_range = np.asarray([20, 23, 24.2, 25.5]) 
for n_i, n in enumerate(n_range):
    
    GNC = GatherNullClines(753, 494, 719, n, temp/(2.0**(1.0/n)), +1)  
    d_alpha_dt, norm_condition = GNC.calc_d_alpha_dt(alpha_mesh, l_0_mesh)
    d_ell_dt, d_ell_dt_p_sat, d_ell_dt_m_sat = GNC.calc_d_ell_dt(alpha_mesh, l_0_mesh)
    
    nullcline_axes[n_i].contour(l_0_mesh, alpha_mesh, d_alpha_dt, [0], colors="purple", linewidths=6, alpha=0.5)
    nullcline_axes[n_i].contour(l_0_mesh, alpha_mesh, d_ell_dt, [0], colors="orange", linewidths=6, alpha=0.5)

    n_mask = data[:, 1] == n
    nullcline_axes[n_i].scatter(data[n_mask, -1][[0, np.sum(n_mask)//2, -1]], data[n_mask, -3][[0, np.sum(n_mask)//2, -1]], s=20, color="k", zorder=10)
    
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)

    nullcline_axes[n_i].text(0.07, 0.94, r"$n=$"+str(n), transform=nullcline_axes[n_i].transAxes, fontsize=fontsize, verticalalignment='top', bbox=props)

    
    # Line across which to compute E
    fit_pow = [2, 2, 1, 1][n_i]
    pnts_x, pnts_y = data[n_mask, -1], data[n_mask, -3]
    p = np.polyfit(pnts_x, pnts_y, fit_pow)
    pnts_x_detailed = np.linspace(np.min(pnts_x)-0.15*2, np.max(pnts_x)+0.15*2, 1000)
    pnts_y_detailed = np.polyval(p, pnts_x_detailed)
    nullcline_axes[n_i].plot(pnts_x_detailed, pnts_y_detailed, lw=3, c="red")

    # Compute E
    
    log_E = np.log(E(pnts_x_detailed, pnts_y_detailed, n, 700.0/(2.0**(1.0/n)) ))
    E_axes[n_i].plot(pnts_x_detailed, log_E, c="red", linestyle="", marker=".", ms=1)
    E_axes[n_i].set_ylim([1.25, 1.38, 1.4, 1.45][n_i], 1.5)

    E_axes[n_i].text(0.07, 0.94, r"$n=$"+str(n), transform=E_axes[n_i].transAxes, fontsize=fontsize, verticalalignment='top', bbox=props)
    



    
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

    if i == 6 or i == 8 or i==9 or i==11:
        ax.remove()

    
    
plt.subplots_adjust(top=0.99, bottom=0.05, left=0.075, right=0.925, hspace=0.05, wspace=0.05) 


plt.savefig("Figure_6S_SaddleNode_2_tmp.png")

