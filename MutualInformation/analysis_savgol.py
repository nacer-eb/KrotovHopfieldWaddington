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
if False:
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    colors = ["blue", "orange", "red"]
    window_size=[25, 27, 19]
    for h, temp in enumerate(temp_range):
        data_coefs_abs_max = np.mean(np.max(np.abs(data_coefs_flat[h]), axis=-1), axis=-1)
        data_coefs_abs_max_std = np.std(np.max(np.abs(data_coefs_flat[h]), axis=-1), axis=-1)    
      
        filtered = sig.savgol_filter(data_coefs_abs_max, window_size[h], 5)
        filtered = sig.savgol_filter(filtered, window_size[h], 5)
        ax.plot(n_range, filtered, label="T="+str(temp), color=colors[h])
        
        ax.scatter(n_range, data_coefs_abs_max, marker=".", color=colors[h], alpha=0.3, edgecolor="k")

        delta = data_coefs_abs_max_std
        delta = sig.savgol_filter(delta, 21, 3)
        
        ax.fill_between(n_range, filtered + delta, filtered-delta, alpha=0.2, facecolor=colors[h])

        

    ax.set_xlabel("n-power")
    ax.set_ylabel("Maximum Absolute Alpha")
    ax.legend()
    plt.savefig("max_abs_alpha_multiT.png")


# Paul Entropy is interesting and non-monotonic
if False:
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    colors = ["blue", "orange", "red"]
    window_size=[33, 33, 33]
    for h, temp in enumerate(temp_range):
        p_i = np.zeros((Nn, N_mem, 10))
        entropy = np.zeros((Nn, N_mem))
        
        window_start = 0
        window_end = 30 #21 is rly good # 30 is good
        
        for i, n in enumerate(n_range):
            for j in range(N_mem):
                i_sort = np.argsort(np.abs(data_coefs_flat[h, i, j]), axis=-1)[::-1]
                i_sort_index = i_sort[window_start:window_end]//20
            
                for d in range(10):
                    p_i[i, j, d] = np.sum(i_sort_index==d)/(window_end - window_start)
                    
                    if p_i[i, j, d] > 0:
                        entropy[i, j] += -p_i[i, j, d]*np.log10(p_i[i, j, d])

                    
        mean_entropy_per_n = np.mean(entropy, axis=-1)
        mean_entropy_per_n_std = np.std(entropy, axis=-1)*0.1
      
        filtered = sig.savgol_filter(mean_entropy_per_n, window_size[h], 3)
        filtered = sig.savgol_filter(filtered, window_size[h], 3)

        
        ax.plot(n_range, filtered, label="T="+str(temp), color=colors[h])
        ax.scatter(n_range, mean_entropy_per_n, marker=".", color=colors[h], alpha=0.73) #, edgecolor="k"
        
        delta = mean_entropy_per_n_std
        delta = sig.savgol_filter(delta, 21, 3)
        
        ax.fill_between(n_range, filtered + delta, filtered-delta, alpha=0.2, facecolor=colors[h])
        
    ax.set_xlabel("n-power")
    ax.set_ylabel("Paul's Entropy")
    ax.legend()
    plt.savefig("First_20_dominating_samples_entropy_PaulsEntropy_multiT.png")



# sum of alpha = 0.5 thing
window = np.zeros((Nt, Nn, N_mem)) # reconstruct size
alpha_tol = 0.2
for h, temp in enumerate(temp_range):
    for i, n in enumerate(n_range):
   
        for j in range(N_mem):
            i_sort = np.argsort(np.abs(data_coefs_flat[h, i, j]), axis=-1)[::-1]
            i_sort_sum = np.cumsum(data_coefs_flat[h, i, j, i_sort])

            if np.max(data_coefs_flat[h, i, j]) >= alpha_tol:
                window[h, i, j] = np.argwhere(i_sort_sum >= alpha_tol)[0] + 1
            else:
                window[h, i, j] = len(data_coefs_flat[h, i, j])


window_mean = np.mean(window, axis=-1)
window_std = np.std(window, axis=-1)*0.5

fig, ax = plt.subplots(1, 1, figsize=(16, 9))

colors = ["blue", "orange", "red"]
window_size = np.asarray([11, 11, 11])+6
for h, temp in enumerate(temp_range):
    #ax.scatter(n_range, window_mean[h], label="T="+str(temp))
      
    filtered = sig.savgol_filter(window_mean[h], window_size[h], 4)
    filtered = sig.savgol_filter(filtered, 7, 1)
    
    ax.plot(n_range, filtered, label="T="+str(temp), color=colors[h])
    ax.scatter(n_range, window_mean[h], marker=".", color=colors[h], alpha=0.73) #, edgecolor="k"
    
    delta = window_std[h]
    delta = sig.savgol_filter(delta, 21, 3)
    
    ax.fill_between(n_range, filtered + delta, filtered-delta, alpha=0.2, facecolor=colors[h])
    
ax.set_xlabel("n-power")
ax.set_ylabel("Number of useful samples")
ax.legend()
plt.savefig("tmp.png")



# Paul Entropy is interesting and non-monotonic
if True:
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    colors = ["blue", "orange", "red"]
    window_size=[33, 33, 33]
    for h, temp in enumerate(temp_range):
        p_i = np.zeros((Nn, N_mem, 10))
        entropy = np.zeros((Nn, N_mem))
        
       
        for i, n in enumerate(n_range):
            window_start = 0
            window_end = int(window_mean[h, i]) #21 is rly good # 30 is good

            for j in range(N_mem):
                i_sort = np.argsort(np.abs(data_coefs_flat[h, i, j]), axis=-1)[::-1]
                i_sort_index = i_sort[window_start:window_end]//20
            
                for d in range(10):
                    p_i[i, j, d] = np.sum(i_sort_index==d)/(window_end - window_start)
                    
                    if p_i[i, j, d] > 0:
                        entropy[i, j] += -p_i[i, j, d]*np.log10(p_i[i, j, d])

                    
        mean_entropy_per_n = np.mean(entropy, axis=-1)
        mean_entropy_per_n_std = np.std(entropy, axis=-1)*0.1
      
        filtered = sig.savgol_filter(mean_entropy_per_n, window_size[h], 3)
        filtered = sig.savgol_filter(filtered, window_size[h], 3)

        
        #ax.plot(n_range, filtered, label="T="+str(temp), color=colors[h])
        ax.scatter(n_range, mean_entropy_per_n, marker=".", color=colors[h], alpha=0.73) #, edgecolor="k"
        
        delta = mean_entropy_per_n_std
        delta = sig.savgol_filter(delta, 21, 3)
        
        #ax.fill_between(n_range, filtered + delta, filtered-delta, alpha=0.2, facecolor=colors[h])
        
    ax.set_xlabel("n-power")
    ax.set_ylabel("Paul's Entropy")
    ax.legend()
    plt.savefig("tmp.png")
