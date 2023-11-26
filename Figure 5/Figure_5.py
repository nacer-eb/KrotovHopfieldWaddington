import sys
sys.path.append('../')

import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from Figure_5_functions import *

fontsize=44
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : fontsize}
matplotlib.rc('font', **font)


fig = plt.figure(figsize=(27, 65-22+1+1 ))
axs = fig.subplot_mosaic(
    """
    1111111.22222222222.3333333
    1111111.22222222222.3333333
    1111111.22222222222.3333333
    1111111.22222222222.3333333
    1111111.22222222222.3333333
    1111111.22222222222.3333333
    1111111.22222222222.3333333
    1111111.22222222222.3333333
    1111111.22222222222.3333333
    ...........................
    ...........................
    ...........................
    aabbcc.ddeeff.gghhii.jjkkll
    aabbcc.ddeeff.gghhii.jjkkll
    AAAAAA.BBBBBB.CCCCCC.DDDDDD
    AAAAAA.BBBBBB.CCCCCC.DDDDDD
    AAAAAA.BBBBBB.CCCCCC.DDDDDD
    AAAAAA.BBBBBB.CCCCCC.DDDDDD
    AAAAAA.BBBBBB.CCCCCC.DDDDDD
    AAAAAA.BBBBBB.CCCCCC.DDDDDD
    ...........................
    ...........................
    ...........................
    mmmnnn.oooppp.qqqrrr.sssttt
    mmmnnn.oooppp.qqqrrr.sssttt
    mmmnnn.oooppp.qqqrrr.sssttt
    EEEEEE.FFFFFF.GGGGGG.HHHHHH
    EEEEEE.FFFFFF.GGGGGG.HHHHHH
    EEEEEE.FFFFFF.GGGGGG.HHHHHH
    EEEEEE.FFFFFF.GGGGGG.HHHHHH
    EEEEEE.FFFFFF.GGGGGG.HHHHHH
    EEEEEE.FFFFFF.GGGGGG.HHHHHH
    ...........................
    ...........................
    ...........................
    ...........................
    IIIIII.JJJJJJ.KKKKKK.LLLLLL
    IIIIII.JJJJJJ.KKKKKK.LLLLLL
    IIIIII.JJJJJJ.KKKKKK.LLLLLL
    IIIIII.JJJJJJ.KKKKKK.LLLLLL
    IIIIII.JJJJJJ.KKKKKK.LLLLLL
    IIIIII.JJJJJJ.KKKKKK.LLLLLL
    uuuvvv.wwwxxx.yyyzzz.!!!@@@
    uuuvvv.wwwxxx.yyyzzz.!!!@@@
    uuuvvv.wwwxxx.yyyzzz.!!!@@@
    """
)

f3 = Figure_5("run_[1, 4]_n15_T700_alpha0.8_l_00.5.npz") 

default_pos_ax2 = axs['2'].get_position()
axs['2'].remove()

axs['1'].set_xlabel(r'$\alpha_1$', labelpad=10); axs['3'].set_xlabel(r'$\ell$', labelpad=20);
axs['1'].set_ylabel('n-power', labelpad=20); axs['3'].set_ylabel('n-power', labelpad=20);

sample_axis_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '!', '@']
for c in sample_axis_letters:
    axs[c].set_xticks([]); axs[c].set_yticks([])


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
    nullcline_axes[i].set_xlabel(r'$\ell$', labelpad=20);
    if i >= 1:
        nullcline_axes[i].set_yticks([]);
axs['A'].set_ylabel(r'$\alpha_1$', labelpad=20)


# The snapshot of memories at fixed points
mem_snapshots_axs = np.asarray( [ [axs['a'], axs['b'], axs['c']],
                                  [axs['d'], axs['e'], axs['f']],
                                  [axs['g'], axs['h'], axs['i']],
                                  [axs['j'], axs['k'], axs['l']]] )

n_range = np.asarray([6, 20, 30, 40])
for i, n in enumerate(n_range):
    alphas, betas = f3.plot_nullclines(nullcline_axes[i], n)
     
    for j in range(3):
        f3.plot_snapshot(mem_snapshots_axs[i, j], alphas[j], betas[j], isStable=int(j!=1)+int(i==2), fontsize=30) # if j==1 it's unstable unless i==2 in which case the center point is stable


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


center_ax_3d.set_xlabel(r"$\alpha_1$", labelpad=30); center_ax_3d.set_ylabel(r"$\ell$", labelpad=30); center_ax_3d.set_zlabel(r"$n$", labelpad=15)
center_ax_3d.set_xticks([0, 0.5, 1])
center_ax_3d.set_xticklabels([0, 0.5, 1])
center_ax_3d.set_yticks([-1, 0, 1])
center_ax_3d.set_yticklabels([-1, 0, 1])
center_ax_3d.set_zticks([10, 60])
center_ax_3d.set_zticklabels([10, 60])
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
    axs['1'].scatter(alphas[:, i], ns, c=ns, cmap=cmap, norm=norm, s=3)
    axs['3'].scatter(l_0s[:, i], ns, c=ns, cmap=cmap, norm=norm, s=3)
    center_ax_3d.scatter(alphas[:, i], l_0s[:, i], ns, c=ns, cmap=cmap, norm=norm, s=2.5)



# Dynamics

n_range = np.asarray([39, 15])
t_range = np.asarray([[0, 1000, 2000, 3000], [0, 1000, 2000, 3000]])

# Fetch all nullcline axes
dynamics_nullcline_axes = np.asarray([[axs['E'], axs['F'], axs['G'], axs['H']],
                                      [axs['I'], axs['J'], axs['K'], axs['L']]])


# Cosmetics
props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)

for i in range(0, 2):
    for j in range(0, 4):
        dynamics_nullcline_axes[i, j].set_xlabel(r'$\ell$', labelpad=10);
        
        if i == 1:
            dynamics_nullcline_axes[i, j].xaxis.tick_top()
            dynamics_nullcline_axes[i, j].xaxis.set_label_position('top')
            dynamics_nullcline_axes[i, j].set_xlabel(r'$\ell$', labelpad=10)
    
        if j != 0:
            dynamics_nullcline_axes[i, j].set_yticks([]);
            continue
        
        dynamics_nullcline_axes[i, j].set_ylabel(r'$\alpha_1$', labelpad=20)
        dynamics_nullcline_axes[i, j].text(-0.5, 0.5, r"$n=$"+str(n_range[i]), transform=dynamics_nullcline_axes[i, j].transAxes, fontsize=fontsize, verticalalignment='center', horizontalalignment='right', rotation=90, bbox=props)


# Fetching all the memory snapshot axes - The snapshot of memories at fixed points
dynamics_mem_snapshots_axs = np.asarray( [[ [axs['m'], axs['n']],
                                  [axs['o'], axs['p']],
                                  [axs['q'], axs['r']],
                                  [axs['s'], axs['t']]],
                                  [[axs['u'], axs['v']],
                                  [axs['w'], axs['x']],
                                  [axs['y'], axs['z']],
                                  [axs['!'], axs['@']]]] )



    
for i, n in enumerate(n_range):
    # Create Figure 5 Obj
    f3 = Figure_5("run_[1, 4]_n"+str(n)+"_T700_alpha0.8_l_00.5.npz")
    
    for j, t in enumerate(t_range[i]):
        alphas, betas = f3.plot_nullclines(dynamics_nullcline_axes[i, j], n, t_0=t_range[i, j-1], t=t, plotDynamics=True, density=(j<2))

        for k in range(2):
            f3.plot_snapshot(dynamics_mem_snapshots_axs[i, j, k], alphas[k], betas[k], hasStabilityTitle=False) # if j==1 it's unstable unless i==2 in which case the center point is stable



plt.savefig("Figure_5_tmp.png")
