import sys
sys.path.append('../')

import numpy as np
import sys

from main_module.KrotovV2 import *

data_dir = "../paper - figure 1/data/main/"

fig = plt.figure(figsize=(10+2+10, 10+2+10))

axs = fig.subplot_mosaic("""

AAAAAAAAAA.BBBBBBBBBBx
AAAAAAAAAA.BBBBBBBBBBx
AAAAAAAAAA.BBBBBBBBBBx
AAAAAAAAAA.BBBBBBBBBBx
AAAAAAAAAA.BBBBBBBBBBx
AAAAAAAAAA.BBBBBBBBBBx
AAAAAAAAAA.BBBBBBBBBBx
AAAAAAAAAA.BBBBBBBBBBx
AAAAAAAAAA.BBBBBBBBBBx
AAAAAAAAAA.BBBBBBBBBBx
......................
......................
CCCCCCCCCC.DDDDDDDDDDX
CCCCCCCCCC.DDDDDDDDDDX
CCCCCCCCCC.DDDDDDDDDDX
CCCCCCCCCC.DDDDDDDDDDX
CCCCCCCCCC.DDDDDDDDDDX
CCCCCCCCCC.DDDDDDDDDDX
CCCCCCCCCC.DDDDDDDDDDX
CCCCCCCCCC.DDDDDDDDDDX
CCCCCCCCCC.DDDDDDDDDDX
CCCCCCCCCC.DDDDDDDDDDX

""")


ax = np.asarray([[axs['A'], axs['B']],
                 [axs['C'], axs['D']]])

cb_ax = np.asarray([axs['x'], axs['X']])


props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)

for i, n in enumerate([30, 3]):
    data = np.load(data_dir + "trained_net_n"+str(n)+"_T670.npz")
    data_M = data["M"]
    data_T = data["miniBatchs_images"][0]
    data_T_inv = np.linalg.pinv(data_T)

    ax[i, 0].imshow(merge_data(data_M[-1], 10, 10), cmap="bwr", vmin=-1, vmax=1)
    ax[i, 1].imshow(merge_data(data_M[-1]@data_T_inv@data_T, 10, 10), cmap="bwr", vmin=-1, vmax=1)

    ax[0, 0].set_title("Original memories", pad=30, fontsize=25, bbox=props); ax[0, 1].set_title("Moore-Penrose reconstructed memories", pad=30, fontsize=25, bbox=props)
    ax[i, 0].set_ylabel(r"$n=$"+str(n), labelpad=30, fontsize=25, bbox=props); 
    
    
    # Cosmetics
    for j in range(2):
        ax[i, j].set_xticks([]); ax[i, j].set_yticks([])


    bwr_cmap = matplotlib.cm.bwr
    bwr_norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    cb_mem = matplotlib.colorbar.ColorbarBase(cb_ax[i], cmap=bwr_cmap, norm=bwr_norm, orientation='vertical')
    cb_mem.set_label("Pixel value")

plt.savefig("Figure_MP_supp.png")

