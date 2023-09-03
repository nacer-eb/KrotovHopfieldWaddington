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


fig = plt.figure(figsize=(15*1.5, (5+1+1+3)*1.5 ))
axs = fig.subplot_mosaic(
    """
    111122222223333
    111122222223333
    111122222223333
    111122222223333
    111122222223333
    ...............
    abc.def.ghi.jkl
    AAA.BBB.CCC.DDD
    AAA.BBB.CCC.DDD
    AAA.BBB.CCC.DDD
    """
)


# Loading bifurcation scatter plot (1, 2, 3) data
data_dir = "data_2_2_2/"
C_data_dir = "C_Code_FPs/"

data = np.loadtxt(C_data_dir+"save0.dat", delimiter=",", skiprows=1)
p_mask = (np.abs(data[:, -2]) > 0.001) * (np.abs(data[:, -3]) > 0.001 ) * (data[:, 1] <= 7) + (np.abs(data[:, -2]) > 0.03) * (np.abs(data[:, -3]) > 0.03 ) * (data[:, 1] > 7)
data = data[p_mask, :]

# Cosmetics
default_pos_ax2 = axs['2'].get_position()
axs['2'].remove() # Making space for 3d plot
axs['1'].set_xlabel(r'$\alpha$'); axs['3'].set_xlabel(r'$\alpha$');
axs['1'].set_ylabel('n-power'); axs['3'].set_ylabel('n-power');

# Creating the colormap for n
norm = plt.Normalize(np.min(data[:, 1]), np.max(data[:, 1])) # Norm map for n-power
c1 = np.asarray([191/256.0, 127/256.0, 191/256.0, 1]) # purple
c2 = np.asarray([255/256.0, 209/256.0, 127/256.0, 1]) # golden yellow

k = np.linspace(0, 1, 256)
vals = np.zeros((256, 4))
for i in range(0, 256):
    vals[i] = c1*(1 - k[i]) + c2*k[i]
cmap = matplotlib.colors.ListedColormap(vals)


# Plotting
axs['1'].scatter(data[:, -3], data[:, 1], c=data[:, 1] , cmap=cmap, norm=norm, s=0.1)
axs['3'].scatter(data[:, -1], data[:, 1], c=data[:, 1] , cmap=cmap, norm=norm, s=0.1)



# Loading training examples
data_T = np.load(data_dir+"run_[1, 4]_n15_T700_alpha0.2_l_0-0.5.npz")['miniBatchs_images'][0]
A, B = data_T[0], data_T[1]


# Define nullcline axes
nullcline_axes = np.asarray([axs['A'], axs['B'], axs['C'], axs['D']])
# Cosmetics
for i in range(0, len(nullcline_axes)):
    nullcline_axes[i].set_xlabel(r'$l_0$');
    if i >= 1:
        nullcline_axes[i].set_yticks([]);
axs['A'].set_ylabel(r'$\alpha$')


temp = 700
alpha_range = np.linspace(0, 1, 1000); l_range = np.linspace(-1, 1, 1000)
l_0_mesh, alpha_mesh = np.meshgrid(l_range, alpha_range)


# Define the general nullcline plot
def plot_nullclines(ax, n):
    GNC = GatherNullClines(753, 494, 719, n, temp/(2.0**(1.0/n)), +1)  
    alpha_nullcline = GNC.alpha_nullcline(alpha_mesh, l_0_mesh)
    l_nullcline = GNC.l_0_nullcline(alpha_mesh, l_0_mesh)

    ax.contour(l_0_mesh, alpha_mesh, alpha_nullcline, [0], colors="purple", linewidths=4, alpha=0.5)
    ax.contour(l_0_mesh, alpha_mesh, l_nullcline, [0], colors="orange", linewidths=4, alpha=0.5)

    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    ax.text(0.07, 0.94, r"$n=$"+str(n), transform=ax.transAxes, fontsize=18, verticalalignment='top', horizontalalignment='left', bbox=props)

    # Fetching all FPs relevant to that n-value
    n_mask = data[:, 1] == n

    l_0s = data[n_mask, -1][[0, np.sum(n_mask)//2, -1]]
    alphas = data[n_mask, -3][[0, np.sum(n_mask)//2, -1]]
    betas = data[n_mask, -2][[0, np.sum(n_mask)//2, -2]]
    # Plotting 3 FPs
    ax.scatter(l_0s, alphas, s=40, facecolor=['green', 'red', 'green'], edgecolor="k", linewidths=1, zorder=10)

    return alphas, betas

# The snapshot of memories at fixed points
mem_snapshots_axs = np.asarray( [ [axs['a'], axs['b'], axs['c']],
                                  [axs['d'], axs['e'], axs['f']],
                                  [axs['g'], axs['h'], axs['i']],
                                  [axs['j'], axs['k'], axs['l']]] )

# Cosmetics
for ax in mem_snapshots_axs.ravel():
    ax.set_xticks([]); ax.set_yticks([]);


def plot_snapshot(ax, alpha, beta, isStable=True):
    sample_mem = alpha * A + beta * B
    ax.imshow(sample_mem.reshape(28, 28), cmap="bwr")

    ax.set_title("Stable", color="green", fontsize=14)
    if not isStable:
        ax.set_title("Unstable", color="red", fontsize=14)


n_range = np.asarray([10, 20, 30, 40])
for i, n in enumerate(n_range):
    alphas, betas = plot_nullclines(nullcline_axes[i], n)
     
    for j in range(3):
        plot_snapshot(mem_snapshots_axs[i, j], alphas[j], betas[j], isStable=int(j!=1)+int(i==2)) # if j==1 it's unstable unless i==2 in which case the center point is stable


# Cosmetics - remove duplicates for single FPs
mem_snapshots_axs[2, 0].remove()
mem_snapshots_axs[2, -1].remove()


# Now plotting the center axis
dx_adjust = 0.02
dy_adjust = 0.0
center_ax_3d = fig.add_axes([default_pos_ax2.x0 + dx_adjust-0.02,
                             default_pos_ax2.y0 + dy_adjust,
                             default_pos_ax2.x1-default_pos_ax2.x0-2*dx_adjust,
                             default_pos_ax2.y1 - default_pos_ax2.y0-2*dy_adjust],
                            projection='3d')

center_ax_3d.scatter(data[:, -3], data[:, -1], data[:, 1], c=data[:, 1], cmap=cmap, norm=norm, s=1)
center_ax_3d.set_xlabel(r"$\alpha$", labelpad=10); center_ax_3d.set_ylabel(r"$l_0$", labelpad=13); center_ax_3d.set_zlabel(r"$n$", labelpad=5)
center_ax_3d.locator_params(axis='x', nbins=5)
center_ax_3d.locator_params(axis='y', nbins=5)


plt.savefig("Figure_3_plot.png")


