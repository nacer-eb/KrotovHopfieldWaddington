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

data_coefs_flat = data_coefs.reshape(Nt, Nn, N_mem, 200)

#
#
#
#


import scipy.signal as sig

# Average maximum (absolute) alpha per n
if True:
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    colors = ["blue", "orange", "red"]
    window_size=np.asarray([9, 9, 9])-2#[25, 27, 19]
    for h, temp in enumerate(temp_range):
        data_coefs_abs_max = np.mean(np.max(np.abs(data_coefs_flat[h]), axis=-1), axis=-1)       

        
        ma = np.zeros_like(data_coefs_abs_max)
        ma[:(window_size[h]-1)//2] = data_coefs_abs_max[:(window_size[h]-1)//2]
        
        for i in range((window_size[h]-1)//2, len(ma)-(window_size[h]-1)//2):
            ma[i] = np.mean(data_coefs_abs_max[i-(window_size[h]-1)//2:i+(window_size[h]-1)//2])

        for i in range(len(ma)-(window_size[h]-1)//2, len(ma)):
            ma[i] = np.mean(data_coefs_abs_max[i-window_size[h]:i])
            
        ma = sig.savgol_filter(ma, window_size[h], 3)

        ax.plot(n_range, ma, label="T="+str(temp), color=colors[h])    
        
        ax.scatter(n_range, data_coefs_abs_max, marker=".", color=colors[h], alpha=0.3)

        #delta = np.abs(filtered - data_coefs_abs_max)
        #delta = sig.savgol_filter(delta, 19, 3)
        #ax.fill_between(n_range, filtered + delta, filtered-delta, color=colors[h], alpha=0.3)

        

    ax.set_xlabel("n-power")
    ax.set_ylabel("Maximum Absolute Alpha")
    ax.legend()
    plt.savefig("max_abs_alpha_multiT.png")




