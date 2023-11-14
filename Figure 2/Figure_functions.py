import sys
sys.path.append('../')

import numpy as np

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *

import scipy.signal as sig

import matplotlib.pyplot as plt


data_dir = "data/"
selected_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
prefix = str(selected_digits)+"/" # I used main,and momentum #"main"#

tmax = 10000
N_mem = 100

n_range = np.arange(2, 61, 1)
Nn = len(n_range)

temp_range = [450, 650, 750]
Nt = len(temp_range)

data_Ms = np.zeros((Nt, Nn, N_mem, 784))
data_Ls = np.zeros((Nt, Nn, N_mem, 10))


dataset = "../defaults/miniBatchs_images.npy"
umap_model = "../defaults/umap_model_correlation.sav"

data_T = np.load(dataset)[0]
data_T_inv = np.linalg.pinv(data_T)

isFirstRun = True # If you turn this to false it uses the preloaded data (faster)
if isFirstRun:
    for h, temp in enumerate(temp_range):
        for i, n in enumerate(n_range):
            print(n, temp)
        
            saving_dir = data_dir+prefix+"trained_net_end_n"+str(n)+"_T"+str(temp)+".npz"
            
            data = np.load(saving_dir)
            data_Ms[h, i] = data['M'][-1]
            data_Ls[h, i] = data['L'][-1]
            
    data_coefs = (data_Ms@data_T_inv).reshape(Nt, Nn, N_mem, 10, 20)

    # Creates/Saves preloaded data
    np.save(data_dir + prefix + "data_Ms_T.npy", data_Ms)
    np.save(data_dir + prefix + "data_Ls_T.npy", data_Ls)
    np.save(data_dir + prefix + "data_Coefs_T.npy", data_coefs)

# Loads "preloaded" data
data_Ms = np.load(data_dir + prefix + "data_Ms_T.npy")
data_Ls = np.load(data_dir + prefix + "data_Ls_T.npy")
data_coefs = np.load(data_dir + prefix + "data_Coefs_T.npy")

data_coefs_flat = data_coefs.reshape(Nt, Nn, N_mem, 200)



colors = ["blue", "orange", "red"]

# Average maximum (absolute) alpha per n
def plot_max_alpha(ax):
    window_size=[41, 27, 19]
    for h, temp in enumerate(temp_range):
        data_coefs_abs_max = np.mean(np.max(np.abs(data_coefs_flat[h]), axis=-1), axis=-1)
        data_coefs_abs_max_std = np.std(np.max(np.abs(data_coefs_flat[h]), axis=-1), axis=-1)    
        
        filtered = sig.savgol_filter(data_coefs_abs_max, window_size[h], 5)
        filtered = sig.savgol_filter(filtered, window_size[h], 5)
        ax.plot(n_range, filtered, label="T="+str(temp), color=colors[h], lw=10)
        
        ax.scatter(n_range, data_coefs_abs_max, marker=".", color=colors[h], alpha=0.3, edgecolor="k", s=150)
        
        delta = data_coefs_abs_max_std
        delta = sig.savgol_filter(delta, 21, 3)
        
        ax.fill_between(n_range, filtered + delta, filtered-delta, alpha=0.2, facecolor=colors[h])      
    ax.set_ylabel(r"Maximum " + r'$ | \alpha | $ ', labelpad=10)
    ax.set_xlabel("n-power")



def plot_reconstruction_samples(ax, isFirstRun = True):
    
    # # Reconstruction Samples
    if isFirstRun:
        rs = np.zeros((Nt, Nn, N_mem)) # reconstruct size
        for h, temp in enumerate(temp_range):
            for i, n in enumerate(n_range):
                print(i)
                
                tol = 784*0.05*np.mean(np.mean(np.abs(data_Ms[h, i]), axis=-1), axis=-1) # Variable tolerance
                print(temp, n, tol)

                for j in range(N_mem):
                    i_sort = np.argsort(np.abs(data_coefs_flat[h, i, j]), axis=-1)[::-1]
                    
                    for window in range(1, 200):
                        rM = data_coefs_flat[h, i, j, i_sort[:window]]@data_T[i_sort[:window]]
                        err = np.sum(np.abs(rM - data_Ms[h, i, j]))
                        
                        rs[h, i, j] = window
                        if err < tol:
                            break;
    
        np.save(data_dir+prefix+"rs_relevant_samples.npy", rs)
    rs = np.load(data_dir+prefix+"rs_relevant_samples.npy")
    rs_mean = np.mean(rs, axis=-1)
    rs_std = np.std(rs, axis=-1)

    
    window_size=[11, 13, 9]
    for h, temp in enumerate(temp_range):
        
        filtered = sig.savgol_filter(rs_mean[h], window_size[h], 3)
        filtered = sig.savgol_filter(filtered, window_size[h]*2-1, 3)
    
        ax.plot(n_range, filtered, color=colors[h], lw=10)
        ax.scatter(n_range, rs_mean[h], marker=".", color=colors[h], alpha=0.73, s=150) #, edgecolor="k"
    
        delta = rs_std[h]
        delta = sig.savgol_filter(delta, 21, 3)
    
        ax.fill_between(n_range, filtered + delta, filtered-delta, alpha=0.2, facecolor=colors[h])

    #ax.set_ylim(0, 190)
    ax.set_xlabel("n-power")
    ax.set_ylabel("Reconstruction Samples")
                  


#Nt, Nn, N_mem, 784
def plot_mem_samples(ax, t_i, n_i, samples=np.random.randint(0, 100, 8)):
    print(temp_range[t_i], n_range[n_i])
    ax.imshow(merge_data(data_Ms[t_i, n_i, samples], 8, 1), cmap="bwr", vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])






import pickle
import umap

M = len(data_T)
keys = np.zeros((M))
for d in range(0, 10):
    keys[d*M//10:(d+1)*M//10] = d

mapper = pickle.load((open(umap_model, 'rb')))
embedding = mapper.transform(data_T)


props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)

def plot_UMAP(ax, t_i, n_i):
    ax.scatter(embedding[:, 0], embedding[:, 1], c=keys, cmap="tab10", s=210, marker="*") # Plotting the UMAP training data

    data_M = data_Ms[t_i, n_i]
    M_embedding = mapper.transform(data_M)

    ax.plot(M_embedding[:, 0], M_embedding[:, 1], marker="o",
                               linestyle="", alpha=1, markeredgecolor="k", markerfacecolor="white", markeredgewidth=7, markersize=23)
    
    # Time stamps / Cosmetics
    ax.text(0.95, 0.95, r"$n=$"+str(n_range[n_i]), transform=ax.transAxes, fontsize=63, verticalalignment='top', ha='right', bbox=props)
