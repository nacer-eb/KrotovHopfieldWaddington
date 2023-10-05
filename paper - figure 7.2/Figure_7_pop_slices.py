import sys
sys.path.append('../')

import numpy as np
import sys

from main_module.KrotovV2 import *

import matplotlib.pyplot as plt

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 21}
matplotlib.rc('font', **font)


data_dir = "data/"
selected_digits = [1, 7]#
prefix = str(selected_digits)+"/" # _coarse_stable

N_runs = 2

temp_range = np.arange(700, 900, 20)[::1] #temp_range = np.arange(500, 900, 20)
n_range = np.arange(2, 61, 1)[::1] #n_range = np.arange(2, 32, 2)

#temp_range = np.arange(600, 900, 20)[6::2] #temp_range = np.arange(500, 900, 20)
#n_range = np.arange(2, 61, 1)[::2] #n_range = np.arange(2, 32, 2)

N_mem = 100

data_Ms = np.zeros((N_runs, len(temp_range), len(n_range), 2, N_mem, 784))
data_M_saddles = np.zeros((len(temp_range), len(n_range), 2, 784))
data_Ls = np.zeros((N_runs, len(temp_range), len(n_range), 2, N_mem, 10))

isFirstRun = False
if isFirstRun:

    for i, temp in enumerate(temp_range):
        for j, n in enumerate(n_range):
            for k in range(2):
                for r in range(N_runs):

                    run_prefix = "end_states_" + str(r) + "/"
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


data_M_saddles_coefs = data_M_saddles@data_T_inv

data_Ms_pop_run = np.zeros((N_runs, len(temp_range), len(n_range), 2, 2)) # Population proportion
data_Ms_pop = np.zeros((len(temp_range), len(n_range), 2, 2)) # Population proportion

for r in range(N_runs):
    for i, temp in enumerate(temp_range):
        for j, n in enumerate(n_range):
            for k in range(2):
                for l in range(2):
                    data_Ms_pop_run[r, i, j, k, l] = np.sum(np.argmax(data_Ls[r, i, j, k], axis=-1) == selected_digits[l], axis=-1) # not strict
                    #data_Ms_pop_run[r, i, j, k, l] = np.sum( (data_Ls[r, i, j, k, :, selected_digits[l]] >= -0.95), axis=-1 ) # stricter


# Standard mean
data_Ms_pop = np.mean(data_Ms_pop_run, axis=0)


cmap_tab10 = matplotlib.cm.tab10
norm = matplotlib.colors.Normalize(vmin=0, vmax=10)

fig, ax = plt.subplots(3, 2, figsize=(16, 16), sharey=True)

for t_i, t_slices in enumerate([-1, -3, -4]):
    for ic_i, ic in enumerate([1, 7]):
        for r in range(N_runs):
            ax[t_i, 0].scatter(data_Ms_pop_run[r, t_slices, :, ic_i, 0].T/100.0, n_range, s=10, alpha=0.5, color=cmap_tab10(norm(ic)))
        ax[t_i, 1].scatter(data_M_saddles_coefs[t_slices, :, ic_i, 0], n_range, s=10, alpha=0.5, color=cmap_tab10(norm(ic)))

        ax[t_i, 0].set_ylabel("Temperature: "+str(temp_range[t_slices]) + " \n n-range")

        ax[-1, 0].set_xlabel(r"Proportion of $1$s")
        ax[-1, 1].set_xlabel(r"$\alpha_1$ of saddle")





plt.tight_layout()
plt.savefig("Figure_pop_slice.png")
exit()
