import sys
sys.path.append('../')

import numpy as np
import sys

from main_module.KrotovV2 import *

import matplotlib.pyplot as plt

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 31}
matplotlib.rc('font', **font)


data_dir = "data/"
selected_digits = [1, 7, 8]#
prefix = str(selected_digits)+"_long_3/" # _coarse_stable

N_runs = 1

temp_range = [550, 750, 800] #[550, 650, 750] #np.arange(700, 900, 20)[::2] #temp_range = np.arange(500, 900, 20)
n_range = np.arange(2, 60, 1)[::1] #n_range = np.arange(2, 32, 2)

#temp_range = np.arange(600, 900, 20)[6::2] #temp_range = np.arange(500, 900, 20)
#n_range = np.arange(2, 61, 1)[::2] #n_range = np.arange(2, 32, 2)

N_mem = 100

data_Ms = np.zeros((N_runs, len(temp_range), len(n_range), 2, N_mem, 784))
data_M_saddles = np.zeros((len(temp_range), len(n_range), 2, 784))
data_Ls = np.zeros((N_runs, len(temp_range), len(n_range), 2, N_mem, 10))

isFirstRun = True
if isFirstRun:

    for i, temp in enumerate(temp_range):
        for j, n in enumerate(n_range):
            for k in range(2):
                for r in range(N_runs):

                    run_prefix = "end_states_g" + str(r) + "/"
                    saving_dir=data_dir+prefix+run_prefix+"trained_net_end_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz"
                    
                    if os.path.isfile(saving_dir):
                        data_Ms[r, i, j, k] = np.load(saving_dir)['M']
                        data_Ls[r, i, j, k] = np.load(saving_dir)['L']
                    else:
                        print("WARNING: File not found, ", saving_dir)

                saving_dir=data_dir+prefix+"saddles/net_saddle_n"+str(n)+"_T"+str(temp)+"ic"+str(selected_digits[k])+".npz"
                
                if os.path.isfile(saving_dir):
                    data_M_saddles[i, j, k] = np.load(saving_dir)['M'][0]
                    
                else:
                    print("WARNING: File not found, ", saving_dir)
                    
        print(temp)

    np.save(data_dir+prefix+"data_Ms.npy", data_Ms)
    np.save(data_dir+prefix+"data_M_saddles.npy", data_M_saddles)
    np.save(data_dir+prefix+"data_Ls.npy", data_Ls)


# Then
data_Ms = np.load(data_dir+prefix+"data_Ms.npy")
data_M_saddles = np.load(data_dir+prefix+"data_M_saddles.npy")
data_Ls = np.load(data_dir+prefix+"data_Ls.npy")

data_T = np.load(data_dir+prefix+"miniBatchs_images.npy")[0]
data_T_inv = np.linalg.pinv(data_T)


fig, ax = plt.subplots(1, 3, figsize=(16, 16))

im = [0]*3
ic=0
n_i = 0
for t_i, temp in enumerate(temp_range):
    im[t_i] = ax[t_i].imshow(merge_data(data_Ms[0, t_i, n_i, ic], 10, 10), cmap="bwr", vmin=-1, vmax=1)
    ax[t_i].set_title("T="+str(temp)+", n="+str(n_range[n_i]))
    ax[t_i].set_xticks([])
    ax[t_i].set_yticks([])

#(N_runs, len(temp_range), len(n_range), 2, N_mem, 784)

plt.tight_layout()
plt.subplots_adjust(hspace=0.2, wspace=0.1)


import matplotlib.animation as anim


def update(n_i):
    for t_i, temp in enumerate(temp_range):
        im[t_i].set_data(merge_data(data_Ms[0, t_i, n_i, ic], 10, 10))
        ax[t_i].set_title("T="+str(temp)+", n="+str(n_range[n_i]))
        ax[t_i].set_xticks([])
        ax[t_i].set_yticks([])

    return *ax, *im 

ani = anim.FuncAnimation(fig, update, frames=len(n_range), interval=500, blit=False)
ani.save("final_states3_n"+str(ic)+".mov", writer="ffmpeg")
