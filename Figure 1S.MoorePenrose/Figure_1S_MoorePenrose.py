import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2_utils import *

data_dir = '../Figure 1/data/[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]/'

dataset = "../defaults/miniBatchs_images.npy"
data_T = np.load(dataset)[0]


fig, ax = plt.subplots(2, 2, figsize=(16, 9))

for i, n in enumerate([3, 30]):
    saving_dir = data_dir + "trained_net_n"+str(n)+"_T670.npz"

    data_M = np.load(saving_dir)['M']

    ax[i, 0].set_xticks([]); ax[i, 0].set_yticks([])
    ax[i, 1].set_xticks([]); ax[i, 1].set_yticks([])
    
    ax[i, 0].imshow(merge_data(data_M[-1], 10, 10), cmap="bwr", vmin=-1, vmax=1)

    coefs = data_M[-1]@np.linalg.pinv(data_T)
    ax[i, 1].imshow(merge_data(coefs@data_T, 10, 10), cmap="bwr", vmin=-1, vmax=1)
    
plt.subplots_adjust(top=0.983, bottom=0.017, left=0.2, right=0.9, hspace=0.135, wspace=0.0)
plt.savefig("Figure_1S_MoorePenrose_tmp.png")


