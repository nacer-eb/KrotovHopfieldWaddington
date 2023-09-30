import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *

data_dir = "data/"

isFirstRun = False

selected_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#
prefix = str(selected_digits)+"/" # I used main,and momentum #"main"#

tmax = 10000
N_mem = 100

n_range = np.arange(1, 61, 1)
Nn = len(n_range)

temp_range = [550, 650, 750]
Nt = len(temp_range)

data_Ms = np.zeros((Nt, Nn, N_mem, 784))
data_Ls = np.zeros((Nt, Nn, N_mem, 10))

# THIS ASSUME THE DATA WAS GENERATED i.e. FIRST RUN = FALSE
data_Ms = np.load(data_dir + prefix + "data_Ms_T.npy")

n_i = 2
t_i = 2

mem_size = 16
index = np.random.randint(0, 100, size=mem_size)


fig, ax = plt.subplots(3, 6, figsize=(16, 9), dpi=500)

props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)

for t_i in range(0, 3):
    for j, n_i in enumerate(n_range[::len(n_range)//6]):#[2, 13, 22, 27, 45]
        print(j)
        print(n_i)
        ax[t_i, j].imshow(merge_data(data_Ms[t_i, n_i, index], 4, 4), cmap="bwr", vmin=-1, vmax=1)
        ax[t_i, j].set_xticks([]); ax[t_i, j].set_yticks([])

        if t_i == 0:
            ax[0, j].set_title(r"$n="+str(n_range[n_i]) + r"$", pad=30, fontsize=16, bbox=props)





            
plt.subplots_adjust(hspace=0.3, wspace=0.05)
plt.tight_layout()



plt.savefig("samples.png", transparent=True)
