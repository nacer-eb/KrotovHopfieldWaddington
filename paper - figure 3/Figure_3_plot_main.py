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
from Figure_3_functions import *

fig = plt.figure(figsize=(15*1.5, (5+1+1+3+1)*1.5 ))
axs = fig.subplot_mosaic(
    """
    111122222223333
    111122222223333
    111122222223333
    111122222223333
    111122222223333
    ...............
    ...............
    abc.def.ghi.jkl
    AAA.BBB.CCC.DDD
    AAA.BBB.CCC.DDD
    AAA.BBB.CCC.DDD
    """
)


# Figure 3 object
f3 = Figure_3("run_[1, 4]_n15_T700_alpha0.8_l_00.5.npz") 

# Saving space (Pun intended)
default_pos_ax2 = axs['2'].get_position()
axs['2'].remove() # Making space for 3d plot

# Cosmetics
axs['1'].set_xlabel(r'$\alpha$'); axs['3'].set_xlabel(r'$\alpha$');
axs['1'].set_ylabel('n-power'); axs['3'].set_ylabel('n-power');

# Creating the colormap for n
norm = plt.Normalize(np.min(f3.FP_data[:, 1]), np.max(f3.FP_data[:, 1])) # Norm map for n-power
c1 = np.asarray([191/256.0, 127/256.0, 191/256.0, 1]) # purple
c2 = np.asarray([255/256.0, 209/256.0, 127/256.0, 1]) # golden yellow

k = np.linspace(0, 1, 256)
vals = np.zeros((256, 4))
for i in range(0, 256):
    vals[i] = c1*(1 - k[i]) + c2*k[i]
cmap = matplotlib.colors.ListedColormap(vals)



# Define nullcline axes
nullcline_axes = np.asarray([axs['A'], axs['B'], axs['C'], axs['D']])
# Cosmetics
for i in range(0, len(nullcline_axes)):
    nullcline_axes[i].set_xlabel(r'$l_0$');
    if i >= 1:
        nullcline_axes[i].set_yticks([]);
axs['A'].set_ylabel(r'$\alpha$')


# The snapshot of memories at fixed points
mem_snapshots_axs = np.asarray( [ [axs['a'], axs['b'], axs['c']],
                                  [axs['d'], axs['e'], axs['f']],
                                  [axs['g'], axs['h'], axs['i']],
                                  [axs['j'], axs['k'], axs['l']]] )

# Cosmetics
for ax in mem_snapshots_axs.ravel():
    ax.set_xticks([]); ax.set_yticks([]);



n_range = np.asarray([10, 20, 30, 40])
for i, n in enumerate(n_range):
    alphas, betas = f3.plot_nullclines(nullcline_axes[i], n)
     
    for j in range(3):
        f3.plot_snapshot(mem_snapshots_axs[i, j], alphas[j], betas[j], isStable=int(j!=1)+int(i==2)) # if j==1 it's unstable unless i==2 in which case the center point is stable


# Cosmetics - remove duplicates for single FPs
mem_snapshots_axs[2, 0].remove()
mem_snapshots_axs[2, -1].remove()


# Now plotting the center 3d axis
dx_adjust = 0.02
dy_adjust = 0.0
center_ax_3d = fig.add_axes([default_pos_ax2.x0 + dx_adjust-0.02,
                             default_pos_ax2.y0 + dy_adjust,
                             default_pos_ax2.x1-default_pos_ax2.x0-2*dx_adjust,
                             default_pos_ax2.y1 - default_pos_ax2.y0-2*dy_adjust],
                            projection='3d')


center_ax_3d.set_xlabel(r"$\alpha$", labelpad=10); center_ax_3d.set_ylabel(r"$l_0$", labelpad=13); center_ax_3d.set_zlabel(r"$n$", labelpad=5)
center_ax_3d.locator_params(axis='x', nbins=5)
center_ax_3d.locator_params(axis='y', nbins=5)



ns = f3.FP_data[:, 1]
alphas = np.zeros((len(ns), 3))
l_0s = np.zeros((len(ns), 3))
for i, n in enumerate(ns):
    n_mask = f3.FP_data[:, 1]==n
    index_sort = np.argsort(f3.FP_data[n_mask, -1])
    
    alphas[i] = f3.FP_data[n_mask, -3][index_sort][[0, np.sum(n_mask)//2, -1]]
    l_0s[i] = f3.FP_data[n_mask, -1][index_sort][[0, np.sum(n_mask)//2, -1]]

for i in range(3):
    axs['1'].scatter(alphas[:, i], ns, c=ns, cmap=cmap, norm=norm, s=0.7)
    axs['3'].scatter(l_0s[:, i], ns, c=ns, cmap=cmap, norm=norm, s=0.7)
    center_ax_3d.scatter(alphas[:, i], l_0s[:, i], ns, c=ns, cmap=cmap, norm=norm, s=1)

alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
rx = [1/4.0, 1.0]
ry = [1.0/5.0, 1.0]
for i, char in enumerate(['1', 'a']):
    axs[char].text(-0.4*rx[i], 1.0+0.1*ry[i], alphabet[i], transform=axs[char].transAxes, fontsize=44, verticalalignment='bottom', ha='right', fontfamily='Times New Roman', fontweight='bold')

plt.subplots_adjust(wspace=0.03, hspace=0.03)
plt.savefig("Figure-3_tmp.png")


