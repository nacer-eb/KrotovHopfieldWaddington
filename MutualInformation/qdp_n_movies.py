import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *

data_dir = "data/"

isFirstRun = True

selected_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#
prefix = str(selected_digits)+"/" # I used main,and momentum #"main"#

tmax = 10000
N_mem = 100

n_range = np.arange(1, 61, 1)
Nn = len(n_range)

temp_range = [450, 650, 750, 800] #550, 650, 750
Nt = len(temp_range)

data_Ms = np.zeros((Nt, Nn, N_mem, 784))
data_Ls = np.zeros((Nt, Nn, N_mem, 10))

data_T = np.load(data_dir + "miniBatchs_images.npy")[0]
data_T_inv = np.linalg.pinv(data_T)



if isFirstRun:
    for h, temp in enumerate(temp_range):
        for i, n in enumerate(n_range):
            print(n, temp)
        
            saving_dir = data_dir+prefix+"trained_net_end_n"+str(n)+"_T"+str(temp)+".npz"
            
            data = np.load(saving_dir)
            data_Ms[h, i] = data['M'][-1]
            data_Ls[h, i] = data['L'][-1]
            
    data_coefs = (data_Ms@data_T_inv).reshape(Nt, Nn, N_mem, 10, 20)
    
    np.save(data_dir + prefix + "data_Ms_T.npy", data_Ms)
    np.save(data_dir + prefix + "data_Ls_T.npy", data_Ls)
    np.save(data_dir + prefix + "data_Coefs_T.npy", data_coefs)

data_Ms = np.load(data_dir + prefix + "data_Ms_T.npy")
data_Ls = np.load(data_dir + prefix + "data_Ls_T.npy")
data_coefs = np.load(data_dir + prefix + "data_Coefs_T.npy")


fig, ax = plt.subplots(2, 2, figsize=(16, 16))

im = [0]*4

n_i = 0
for t_i, temp in enumerate(temp_range):
    i, j = t_i//2, t_i%2
    im[t_i] = ax[i, j].imshow(merge_data(data_Ms[t_i, n_i], 10, 10), cmap="bwr", vmin=-1, vmax=1)
    ax[i, j].set_title("T="+str(temp)+", n="+str(n_range[n_i]))
    ax[i, j].set_xticks([])
    ax[i, j].set_yticks([])


plt.tight_layout()
plt.subplots_adjust(hspace=0.2, wspace=0.1)


import matplotlib.animation as anim


def update(n_i):
    for t_i, temp in enumerate(temp_range):
        i, j = t_i//2, t_i%2
        im[t_i].set_data(merge_data(data_Ms[t_i, n_i], 10, 10))
        ax[i, j].set_title("T="+str(temp)+", n="+str(n_range[n_i]))
        ax[i, j].set_xticks([])
        ax[i, j].set_yticks([])

    return *ax[0], *ax[1], *im 

ani = anim.FuncAnimation(fig, update, frames=len(n_range), interval=500, blit=False)
ani.save("final_states_n.mov", writer="ffmpeg")

#plt.show()
