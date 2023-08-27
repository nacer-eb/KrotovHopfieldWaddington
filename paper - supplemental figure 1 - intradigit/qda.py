import sys
sys.path.append('../')

import numpy as np
import sys

from main_module.KrotovV2 import *


data_dir = "data_4_100/"

temp_range = np.arange(400, 900, 20)
n_range = np.arange(2, 31, 1)

print(np.shape(temp_range))
print(np.shape(n_range))

data_Ms = np.zeros((len(temp_range), len(n_range), 1, 784))

data_T = np.load(data_dir+"n"+str(n_range[0])+"_T"+str(temp_range[0])+".npz")["miniBatchs_images"][0]
data_T_inv = np.linalg.pinv(data_T)


first_run = False

if first_run:

    for i, temp in enumerate(temp_range):
        for j, n in enumerate(n_range):
            saving_dir=data_dir+"n"+str(n)+"_T"+str(temp)+".npz"
            
            data = np.load(saving_dir)
            data_Ms[i, j] = data['M']
    
        print(i)
    np.save(data_dir + "data_Ms.npy", data_Ms)




    
if not first_run:
    data_Ms = np.load(data_dir + "data_Ms.npy")
    data_coefs = data_Ms @ data_T_inv
    

    aspect = (np.min(n_range) - np.max(n_range))/(np.min(temp_range) - np.max(temp_range))
    extent = [np.min(n_range), np.max(n_range), np.max(temp_range), np.min(temp_range)]

    
    fig, ax = plt.subplots(1, 2)

    ax[0].imshow(merge_data(data_Ms[::2, ::2, 0, :].reshape(len(n_range[::2])*len(temp_range[::2]), 784), len(n_range[::2]), len(temp_range[::2])  ), cmap="bwr", vmin=-1, vmax=1, extent=extent, aspect=aspect)
    
    im = ax[1].imshow(data_coefs[:, :, 0, 0], cmap="Reds", vmin=0.5, vmax=1, extent=extent, aspect=aspect) #

    n = np.arange(np.min(n_range), np.max(n_range), 0.01)
    T_calc = (data_T[0]@data_T[0] + data_T[0]@data_T[1])/( 2 * ( np.arctanh( 1 - (1.0/2.0)**(1.0/(2.0*n)) ) )**(1.0/n) )
    ax[1].scatter(n, T_calc, s=1, color="k")
    ax[1].set_ylim(max(temp_range), min(temp_range))

    ax[1].set_yticks([])
    ax[0].set_xlabel(r"$n$", labelpad=10); ax[1].set_xlabel(r"$n$", labelpad=10)
    ax[0].set_ylabel("Temperature", labelpad=10)
    
    t, b = 0.855, 0.145 
    cb_ax = fig.add_axes([0.91, b, 0.02, t-b])
    
    fig.colorbar(im, cax=cb_ax)
    cb_ax.set_ylabel(r"$\alpha_{4, 1}$ coefficients")
    
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, hspace=0.2, wspace=0.2)
    plt.show()
