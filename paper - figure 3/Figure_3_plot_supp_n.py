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

fig = plt.figure(figsize=(15*1.5, 5*2*1.5 ))
axs = fig.subplot_mosaic(
    """
    abc.def.ghi.jkl
    AAA.BBB.CCC.DDD
    AAA.BBB.CCC.DDD
    AAA.BBB.CCC.DDD
    ...............
    mno.pqr.stu.vwx
    EEE.FFF.GGG.HHH
    EEE.FFF.GGG.HHH
    EEE.FFF.GGG.HHH
    """
)


# Figure 3 object
f3 = Figure_3("run_[1, 4]_n15_T700_alpha0.8_l_00.5.npz") 

# Fetch all nullcline axes
nullcline_axes = np.asarray([axs['A'], axs['B'], axs['C'], axs['D'],
                             axs['E'], axs['F'], axs['G'], axs['H']])
# Cosmetics
for i in range(0, len(nullcline_axes)):
    nullcline_axes[i].set_xlabel(r'$l_0$');
    if i != 0 and i != 4:
        nullcline_axes[i].set_yticks([]);
        continue
    nullcline_axes[i].set_ylabel(r'$\alpha$')


# Fetching all the memory snapshot axes
# The snapshot of memories at fixed points
mem_snapshots_axs = np.asarray( [ [axs['a'], axs['b'], axs['c']],
                                  [axs['d'], axs['e'], axs['f']],
                                  [axs['g'], axs['h'], axs['i']],
                                  [axs['j'], axs['k'], axs['l']],
                                  [axs['m'], axs['n'], axs['o']],
                                  [axs['p'], axs['q'], axs['r']],
                                  [axs['s'], axs['t'], axs['u']],
                                  [axs['v'], axs['w'], axs['x']]] )

# Cosmetics
for ax in mem_snapshots_axs.ravel():
    ax.set_xticks([]); ax.set_yticks([]);



n_range = np.asarray([6, 10, 20, 30, 35, 37, 38, 40])
for i, n in enumerate(n_range):
    alphas, betas = f3.plot_nullclines(nullcline_axes[i], n)
     
    for j in range(3):
        f3.plot_snapshot(mem_snapshots_axs[i, j], alphas[j], betas[j], isStable=int(j!=1)+int(i==3)+int(i==4)) # if j==1 it's unstable unless i==2 in which case the center point is stable


# Cosmetics - remove duplicates for single FPs
mem_snapshots_axs[3, 0].remove()
mem_snapshots_axs[3, -1].remove()
mem_snapshots_axs[4, 0].remove()
mem_snapshots_axs[4, -1].remove()
    
plt.savefig("Figure_3_plot_supp.png")
