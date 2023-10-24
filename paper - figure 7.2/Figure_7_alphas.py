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
selected_digits = [1, 7]#
prefix = str(selected_digits)+"_long/" # _coarse_stable

N_runs = 3

temp_range = [550, 750, 800] #[550, 650, 750] #np.arange(700, 900, 20)[::2] #temp_range = np.arange(500, 900, 20)
n_range = np.arange(2, 60, 1)[::1] #n_range = np.arange(2, 32, 2)

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


data_M_saddles_coefs = data_M_saddles@data_T_inv


data_M_saddles_coefs = (data_M_saddles_coefs.reshape(len(temp_range), len(n_range), 2, 10, 20)).sum(axis=-1)
data_M_saddles_coefs = data_M_saddles_coefs[:, :, :, selected_digits]


data_Ms_pop_run = np.zeros((N_runs, len(temp_range), len(n_range), 2, 2)) # Population proportion
data_Ms_pop = np.zeros((len(temp_range), len(n_range), 2, 2)) # Population proportion

for r in range(N_runs):
    for i, temp in enumerate(temp_range):
        for j, n in enumerate(n_range):
            for k in range(2):
                for l in range(2):
                    #data_Ms_pop_run[r, i, j, k, l] = np.sum(np.argmax(data_Ls[r, i, j, k], axis=-1) == selected_digits[l], axis=-1) # not strict
                    data_Ms_pop_run[r, i, j, k, l] = np.sum( (data_Ls[r, i, j, k, :, selected_digits[l]] >= 0.95), axis=-1 ) # stricter


# Standard mean
data_Ms_pop = np.mean(data_Ms_pop_run, axis=0)


tab10_cmap = matplotlib.colormaps["tab10"]
tab10_norm = matplotlib.colors.Normalize(0, 10)


data_Ms_coefs = data_Ms[0]@data_T_inv
print(np.shape(data_Ms_coefs))

print(n_range[18])
print(temp_range[-1])

fig, ax = plt.subplots(1, 2, figsize=(16, 9))
for d_i, d in enumerate([1, 7]):
    mask = data_Ls[0, -1, 18, 0, :, d] > 0.5

    data_Ms_coefs_d = np.mean(data_Ms_coefs[-1, 18, 0, mask], axis=0) # data_Ms_coefs[-1, 18, 0, mask][0] # The first or mean
    i_sort = np.argsort(np.abs(data_Ms_coefs_d))[::-1]

    ax[d_i].scatter(range(200), data_Ms_coefs_d[i_sort], c=tab10_cmap(tab10_norm(i_sort//20)), s=20)

ax[0].set_ylabel(r"$\alpha$")
ax[0].set_title("1s")
ax[1].set_title("7s")
plt.show()


