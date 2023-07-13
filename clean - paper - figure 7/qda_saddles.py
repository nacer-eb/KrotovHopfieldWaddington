import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *

temp_range = np.arange(500, 800, 20)
n_range = np.arange(2, 31, 2)

data_dir = "data_17_saddles/"

saving_dir= data_dir+"run_1_n"+str(n_range[0])+"_T"+str(temp_range[0])+".npz"
data = np.load(saving_dir)
data_T = data['miniBatchs_images'][0]
data_T_inv = np.linalg.pinv(data_T)

data_M = data['M'][-1]

first_run = False

if first_run:
    
    data_Ms = np.zeros((len(temp_range), len(n_range), len(data_M), 784))
    data_coefs = np.zeros((len(temp_range), len(n_range), len(data_M), len(data_T)))

    for k, run in enumerate([1, 7]):
        for  i, temp in enumerate(temp_range):
            for j, n in enumerate(n_range):
                saving_dir=data_dir+"run_"+str(run)+"_n"+str(n)+"_T"+str(temp)+".npz"
            
                data = np.load(saving_dir)
            
                data_Ms[i, j] = data['M'][-1]          
                data_coefs[i, j] = data_Ms[i, j] @ data_T_inv

                print(i, j, k)
            
        np.save(data_dir+"data_Ms_"+str(run)+".npy", data_Ms)
        np.save(data_dir+"data_coefs_"+str(run)+".npy", data_coefs)
    
if not first_run:

    fig = plt.figure()
    axs = fig.subplot_mosaic(
        """
        ABC
        """
    )
    
    ax_a1 = np.asarray([axs['A'], axs['B']])
    ax_dist = axs['C']

    extent = [min(n_range), max(n_range), max(temp_range), min(temp_range)]
    aspect = (min(n_range) - max(n_range))/(min(temp_range) - max(temp_range))

    axs['A'].set_ylabel("Temperature"); axs['B'].set_yticks([]); axs['C'].set_yticks([])
    axs['A'].set_xlabel(r"$n$"); axs['B'].set_xlabel(r"$n$"); axs['C'].set_xlabel(r"$n$")
    
    for k, run in enumerate([1, 7]):
        data_Ms = np.load(data_dir+"data_Ms_"+str(run)+".npy")
        data_coefs = np.load(data_dir+"data_coefs_"+str(run)+".npy")
        
        
        ax_a1[k].set_title(r"$\alpha_1("+str(run)+")$")
        im = ax_a1[k].imshow(data_coefs[:, :, 0, 0], cmap="bwr", vmin=-1, vmax=1, extent=extent, aspect=aspect)


    data_coefs_1 = np.load(data_dir+"data_coefs_"+str(1)+".npy"); data_coefs_7 = np.load(data_dir+"data_coefs_"+str(7)+".npy")
    ax_dist.imshow((data_coefs_1[:, :, 0, 0] - data_coefs_7[:, :, 0, 0]), cmap="bwr", vmin=-1, vmax=1, extent=extent, aspect=aspect)
    
    ax_dist.set_title(r"$\alpha_1(1) - \alpha_1(7)$")
    plt.show()
    

            
