import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2_utils import *

data_dir = '../Figure 1/data/[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]/'

dataset = "../defaults/miniBatchs_images.npy"
data_T = np.load(dataset)[0]


fig, ax = plt.subplots(3, 2, figsize=(2*10, 3*10))

for i, n in enumerate([3, 30]):
    saving_dir = data_dir + "trained_net_n"+str(n)+"_T670.npz"

    data_M = np.load(saving_dir)['M']
    data_M = data_M[-1, :25]

    ax[0, i].set_xticks([]);
    ax[0, i].set_yticks([])
    ax[1, i].set_xticks([]);
    ax[1, i].set_yticks([])
    
    ax[0, i].imshow(merge_data(data_M, 5, 5), cmap="bwr", vmin=-1, vmax=1)

    coefs = data_M@np.linalg.pinv(data_T)
    for p_i, p in enumerate([200, 10]):
        ax[2-p_i, i].set_xticks([])
        ax[2-p_i, i].set_yticks([])
        for m_i in range(25):
            coefs_sort = np.argsort(np.abs(coefs[m_i]))[::-1]
            coefs[m_i, coefs_sort[p:]] = 0
        

        ax[2-p_i, i].imshow(merge_data(coefs@data_T, 5, 5), cmap="bwr", vmin=-1, vmax=1)
    
plt.subplots_adjust(top=0.983, bottom=0.017, left=0.2, right=0.9, hspace=0.135, wspace=0.1)
plt.savefig("Figure_1S_MoorePenrose_tmp.png")


