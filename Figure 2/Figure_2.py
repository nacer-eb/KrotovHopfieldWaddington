import sys
sys.path.append('../')

import numpy as np


import matplotlib.pyplot as plt

from Figure_functions import *

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 67}
matplotlib.rc('font', **font)

matplotlib.rcParams['axes.linewidth'] = 3

fig = plt.figure(figsize=(16*2+4+1, 80-19+1))

axs = fig.subplot_mosaic("""
1111111111111111.....2222222222222222
1111111111111111.....2222222222222222
1111111111111111.....2222222222222222
1111111111111111.....2222222222222222
1111111111111111.....2222222222222222
1111111111111111.....2222222222222222
1111111111111111.....2222222222222222
1111111111111111.....2222222222222222
1111111111111111.....2222222222222222
1111111111111111.....2222222222222222
.....................................
.....................................
.....................................
.....................................
.....................................
.....................................
aaaaaaaaaaaaaaaa....bbbbbbbbbbbbbbbb.
aaaaaaaaaaaaaaaa....bbbbbbbbbbbbbbbb.
AAAAAAAAAAAAAAAA....BBBBBBBBBBBBBBBB@
AAAAAAAAAAAAAAAA....BBBBBBBBBBBBBBBB@
AAAAAAAAAAAAAAAA....BBBBBBBBBBBBBBBB@
AAAAAAAAAAAAAAAA....BBBBBBBBBBBBBBBB@
AAAAAAAAAAAAAAAA....BBBBBBBBBBBBBBBB@
AAAAAAAAAAAAAAAA....BBBBBBBBBBBBBBBB@
AAAAAAAAAAAAAAAA....BBBBBBBBBBBBBBBB@
AAAAAAAAAAAAAAAA....BBBBBBBBBBBBBBBB@
AAAAAAAAAAAAAAAA....BBBBBBBBBBBBBBBB@
.....................................
.....................................
cccccccccccccccc....dddddddddddddddd.
cccccccccccccccc....dddddddddddddddd.
CCCCCCCCCCCCCCCC....DDDDDDDDDDDDDDDD#
CCCCCCCCCCCCCCCC....DDDDDDDDDDDDDDDD#
CCCCCCCCCCCCCCCC....DDDDDDDDDDDDDDDD#
CCCCCCCCCCCCCCCC....DDDDDDDDDDDDDDDD#
CCCCCCCCCCCCCCCC....DDDDDDDDDDDDDDDD#
CCCCCCCCCCCCCCCC....DDDDDDDDDDDDDDDD#
CCCCCCCCCCCCCCCC....DDDDDDDDDDDDDDDD#
CCCCCCCCCCCCCCCC....DDDDDDDDDDDDDDDD#
CCCCCCCCCCCCCCCC....DDDDDDDDDDDDDDDD#
.....................................
.....................................
eeeeeeeeeeeeeeee....ffffffffffffffff.
eeeeeeeeeeeeeeee....ffffffffffffffff.
EEEEEEEEEEEEEEEE....FFFFFFFFFFFFFFFF$
EEEEEEEEEEEEEEEE....FFFFFFFFFFFFFFFF$
EEEEEEEEEEEEEEEE....FFFFFFFFFFFFFFFF$
EEEEEEEEEEEEEEEE....FFFFFFFFFFFFFFFF$
EEEEEEEEEEEEEEEE....FFFFFFFFFFFFFFFF$
EEEEEEEEEEEEEEEE....FFFFFFFFFFFFFFFF$
EEEEEEEEEEEEEEEE....FFFFFFFFFFFFFFFF$
EEEEEEEEEEEEEEEE....FFFFFFFFFFFFFFFF$
EEEEEEEEEEEEEEEE....FFFFFFFFFFFFFFFF$
""")



from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color="blue", lw=20, marker="", alpha=0.2),
                Line2D([0], [0], color="orange", lw=20, marker="", alpha=0.2),
                Line2D([0], [0], color="red", lw=20, marker="", alpha=0.2),
                Line2D([0], [0], color=plt.cm.tab10(0.9), marker="*", linestyle="", ms=40),
                Line2D([0], [0], marker="o", markeredgecolor="k", markerfacecolor="white", markeredgewidth=7, linestyle="", ms=40)]

fig.legend(custom_lines, [r'$T_r$ = '+'{0:.2f}'.format(temp_range[0]/784.0), r'$T_r$ = '+'{0:.2f}'.format(temp_range[1]/784.0),  r'$T_r$ = '+'{0:.2f}'.format(temp_range[2]/784.0),  'Training Data',  'Memory (Visible layer)'], loc='upper center', ncol=2)



plot_max_alpha(axs['1'])
plot_reconstruction_samples(axs['2'], isFirstRun=False)

plot_mem_samples(axs['a'], 1, 8)
plot_mem_samples(axs['c'], 1, 23)
plot_mem_samples(axs['e'], 1, 48)

plot_mem_samples(axs['b'], -1, 8)
plot_mem_samples(axs['d'], -1, 23)
plot_mem_samples(axs['f'], -1, 48)


plot_UMAP(axs['A'], 1, 8)
plot_UMAP(axs['C'], 1, 23)
plot_UMAP(axs['E'], 1, 48)

plot_UMAP(axs['B'], -1, 8)
plot_UMAP(axs['D'], -1, 23)
plot_UMAP(axs['F'], -1, 48)

axs['A'].set_xticks([]); axs['A'].set_ylabel("UMAP 2")
axs['B'].set_xticks([]); axs['B'].set_yticks([])

axs['C'].set_xticks([]); axs['C'].set_ylabel("UMAP 2")
axs['D'].set_xticks([]); axs['D'].set_yticks([])

axs['E'].set_xlabel("UMAP 1"); axs['E'].set_ylabel("UMAP 2")
axs['F'].set_xlabel("UMAP 1"); axs['F'].set_yticks([]); 


axs['a'].set_title(r'$T_r$ = '+'{0:.2f}'.format(temp_range[1]/784.0), fontsize=55, pad=75, bbox=props)
axs['b'].set_title(r'$T_r$ = '+'{0:.2f}'.format(temp_range[2]/784.0), fontsize=55, pad=75, bbox=props)



for c in ['@', '#', '$']:
    ax_cb_UMAP = axs[c]
    tab10_cmap = matplotlib.cm.tab10
    tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)
    cb_UMAP = matplotlib.colorbar.ColorbarBase(ax_cb_UMAP, cmap=tab10_cmap, norm=tab10_norm, orientation='vertical')
    cb_UMAP.set_ticks(np.arange(0, 10, 1) + 0.5) # Finally found how to center these things 
    cb_UMAP.set_ticklabels(np.arange(0, 10, 1))
    cb_UMAP.set_label("Digit class", labelpad=20)


plt.savefig("Figure_2_tmp.png")


