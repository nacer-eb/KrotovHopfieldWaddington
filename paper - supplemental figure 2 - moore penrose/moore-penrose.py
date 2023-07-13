import sys
sys.path.append('../')

import numpy as np
import sys

from main_module.KrotovV2 import *

data_dir = "F:/KrotovHopfieldClean/paper - figure 1/data_100_10_200/"

n = 30
data = np.load(data_dir + "run_0_n"+str(n)+"_T670.npz")
data_M = data["M"]
data_T = data["miniBatchs_images"][0]

data_T_inv = np.linalg.pinv(data_T)

fig, ax = plt.subplots(1, 2)

for a in ax:
    a.set_xticks([]); a.set_yticks([])

ax[0].imshow(merge_data(data_M[-1], 10, 10), cmap="bwr", vmin=-1, vmax=1)
ax[1].imshow(merge_data(data_M[-1]@data_T_inv@data_T, 10, 10), cmap="bwr", vmin=-1, vmax=1)

plt.show()
