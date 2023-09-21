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

n_range = np.arange(1, 41, 1)
print(n_range)

Nn = len(n_range)

temp_range = [550, 650, 750]

data_Ms = np.zeros((len(temp_range), Nn, N_mem, 784))
data_Ls = np.zeros((len(temp_range), Nn, N_mem, 10))

data_T = np.load(data_dir + "miniBatchs_images.npy")[0]
data_T_inv = np.linalg.pinv(data_T)


if isFirstRun:
    for h, temp in enumerate(temp_range):
        for i, n in enumerate(n_range):
            print(n, temp)
        
            saving_dir = data_dir+prefix+"trained_net_end_n"+str(n)+"_T"+str(temp)+".npz"
            
            data = np.load(saving_dir)
            data_Ms[i] = data['M'][-1]
            data_Ls[i] = data['L'][-1]
            
    data_coefs = (data_Ms@data_T_inv).reshape(Nn, N_mem, 10, 20)
    
    np.save(data_dir + prefix + "data_Ms_T.npy", data_Ms)
    np.save(data_dir + prefix + "data_Ls_T.npy", data_Ls)
    np.save(data_dir + prefix + "data_Coefs_T.npy", data_coefs)

data_Ms = np.load(data_dir + prefix + "data_Ms_T.npy")
data_Ls = np.load(data_dir + prefix + "data_Ls_T.npy")
data_coefs = np.load(data_dir + prefix + "data_Coefs_T.npy")

data_coefs_flat = data_coefs.reshape(Nn, N_mem, 200)
exit()


#
#
#
#




# Average maximum (absolute) alpha per n
if True:
    data_coefs_abs_max = np.mean(np.max(np.abs(data_coefs_flat), axis=-1), axis=-1)
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    ax.scatter(n_range, data_coefs_abs_max, marker=".")
    ax.set_xlabel("n-power")
    ax.set_ylabel("Maximum Absolute Alpha")
    plt.savefig("max_abs_alpha_T"+str(temp)+".png")




# Paul Entropy is interesting and non-monotonic
if True:
    p_i = np.zeros((Nn, N_mem, 10))
    entropy = np.zeros((Nn, N_mem))
    
    window_start = 0
    window_end = 20
    
    for i, n in enumerate(n_range):
        for j in range(N_mem):
            i_sort = np.argsort(np.abs(data_coefs_flat[i, j]), axis=-1)[::-1]
            i_sort_index = i_sort[window_start:window_end]//20
            
            for d in range(10):
                p_i[i, j, d] = np.sum(i_sort_index==d)/(window_end - window_start)
                
                if p_i[i, j, d] > 0:
                    entropy[i, j] += -p_i[i, j, d]*np.log10(p_i[i, j, d])

                    
    mean_entropy_per_n = np.mean(entropy, axis=-1)

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))

    ax.scatter(n_range, mean_entropy_per_n)
    ax.set_xlabel("n-power")
    ax.set_ylabel("Paul's Entropy")
    plt.savefig("First_20_dominating_samples_entropy_PaulsEntropy_T"+str(temp)+".png")





# Combine number of reconstruction with Paul's Entropy
tol = 10 # 1 maybe too strong
rs = np.zeros((Nn, N_mem)) # reconstruct size

for i, n in enumerate(n_range):
    print(i)
    
    for j in range(N_mem):
        i_sort = np.argsort(np.abs(data_coefs_flat[i, j]), axis=-1)[::-1]
        
        for window in range(1, 200):
            rM = data_coefs_flat[i, j, i_sort[:window]]@data_T[i_sort[:window]]
            err = np.sum(np.abs(rM - data_Ms[i, j]))
            
            rs[i, j] = window
            if err < tol:
                break;

rs_mean = np.mean(rs, axis=-1)

fig, ax = plt.subplots(1, 1, figsize=(16, 9))
plt.scatter(n_range, rs_mean)
ax.set_xlabel("n-power")
ax.set_ylabel("Number of useful samples")
plt.savefig("NumberOfUsefulSamplesPerN_T"+str(temp)+".png")

p_i = np.zeros((Nn, N_mem, 10))
entropy = np.zeros((Nn, N_mem))

for i, n in enumerate(n_range):
    for j in range(N_mem):
        i_sort = np.argsort(np.abs(data_coefs_flat[i, j]), axis=-1)[::-1]
        i_sort_index = i_sort[:int(rs[i, j])]//20 # 20 here is number of samples per class used to check if index corresponds to a certain digit class
        
        for d in range(10):
            p_i[i, j, d] = np.sum(i_sort_index==d)/int(rs[i, j])
            
            if p_i[i, j, d] > 0:
                entropy[i, j] += -p_i[i, j, d]*np.log10(p_i[i, j, d])
                
                
mean_entropy_per_n = np.mean(entropy, axis=-1)

fig, ax = plt.subplots(1, 1, figsize=(16, 9))

ax.scatter(n_range, mean_entropy_per_n)
ax.set_xlabel("n-power")
ax.set_ylabel("Variable Binsize Paul Entropy")
plt.savefig("VariablePaulsEntropy_T"+str(temp)+".png")