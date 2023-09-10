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

fig = plt.figure(figsize=(19*1.2, 7*2*1.2 ))
axs = fig.subplot_mosaic(
    """
    aabb.ccdd.eeff.gghh
    aabb.ccdd.eeff.gghh
    AAAA.BBBB.CCCC.DDDD
    AAAA.BBBB.CCCC.DDDD
    AAAA.BBBB.CCCC.DDDD
    AAAA.BBBB.CCCC.DDDD
    ...................
    ...................
    EEEE.FFFF.GGGG.HHHH
    EEEE.FFFF.GGGG.HHHH
    EEEE.FFFF.GGGG.HHHH
    EEEE.FFFF.GGGG.HHHH
    iijj.kkll.mmnn.oopp
    iijj.kkll.mmnn.oopp
    """
)

# Relevant later
n_range = np.asarray([39, 15])
t_range = np.asarray([[0, 1000, 2000, 3000], [0, 1000, 2000, 3000]])

# Fetch all nullcline axes
nullcline_axes = np.asarray([[axs['A'], axs['B'], axs['C'], axs['D']],
                             [axs['E'], axs['F'], axs['G'], axs['H']]])

# Cosmetics
props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)

for i in range(0, 2):
    for j in range(0, 4):
        nullcline_axes[i, j].set_xlabel(r'$l_0$');
        
        if i == 1:
            nullcline_axes[i, j].xaxis.tick_top()
            nullcline_axes[i, j].xaxis.set_label_position('top')
            nullcline_axes[i, j].set_xlabel(r'$l_0$', labelpad=10)
    
        if j != 0:
            nullcline_axes[i, j].set_yticks([]);
            continue
        
        nullcline_axes[i, j].set_ylabel(r'$\alpha$')
        nullcline_axes[i, j].text(-0.3, 0.5, r"$n=$"+str(n_range[i]), transform=nullcline_axes[i, j].transAxes, fontsize=18, verticalalignment='center', horizontalalignment='right', rotation=90, bbox=props)


        
        
# Fetching all the memory snapshot axes - The snapshot of memories at fixed points
mem_snapshots_axs = np.asarray( [[ [axs['a'], axs['b']],
                                  [axs['c'], axs['d']],
                                  [axs['e'], axs['f']],
                                  [axs['g'], axs['h']]],
                                  [[axs['i'], axs['j']],
                                  [axs['k'], axs['l']],
                                  [axs['m'], axs['n']],
                                  [axs['o'], axs['p']]]] )

# Cosmetics
for ax in mem_snapshots_axs.ravel():
    ax.set_xticks([]); ax.set_yticks([]);




    
for i, n in enumerate(n_range):
    # Create Figure 3 Obj
    f3 = Figure_3("run_[1, 4]_n"+str(n)+"_T700_alpha0.8_l_00.5.npz")
    
    for j, t in enumerate(t_range[i]):
        alphas, betas = f3.plot_nullclines(nullcline_axes[i, j], n, t_0=t_range[i, j-1], t=t, plotDynamics=True, density=(j<2))

        for k in range(2):
            f3.plot_snapshot(mem_snapshots_axs[i, j, k], alphas[k], betas[k], hasStabilityTitle=False) # if j==1 it's unstable unless i==2 in which case the center point is stable

        


alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
rx = [1.0, 1.0/2.0]
ry = [1.0, 1.0/2.0]
for i, char in enumerate(['a', 'E']):
    axs[char].text(-0.4*rx[i], 1.0+0.1*ry[i], alphabet[i], transform=axs[char].transAxes, fontsize=44, verticalalignment='bottom', ha='right', fontfamily='Times New Roman', fontweight='bold')

plt.subplots_adjust(wspace=0.03, hspace=0.06)
plt.savefig("Figure-3-4.png")
