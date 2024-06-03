import argparse
import sys
sys.path.append('../')

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *

import matplotlib
fontsize = 34
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : fontsize}
matplotlib.rc('font', **font)



# set n when running script
#n = 3

temp = 670
data_dir = "data/"

selected_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
subdir = str(selected_digits)+"/"



umap_model_path = "../defaults/umap_model_correlation.sav"
dataset = "../defaults/miniBatchs_images.npy"
data_T = np.load(dataset)[0]
data_T_inv = np.linalg.pinv(data_T)


import pickle
import umap

mapper = pickle.load((open(umap_model_path, 'rb')))

embedding = mapper.transform(data_T)

mosaic_layout =  ("1111111111111111111111111111111111111111x....2222222222222222222222222222222222222222X\n"*2*20)

fig = plt.figure(figsize=(20+4+20+2, 20), dpi=300)
axs = fig.subplot_mosaic(mosaic_layout)

ax = [axs['1'], axs['2']]
for n_i, n in enumerate([3, 30]):
    print(n)
    M_embedding = np.load(data_dir+subdir+"/memory_umap_embed_correlation_n"+str(n)+"_T"+str(temp)+".npy")

    tmax, K, Nd = M_embedding.shape
    M, Nd = embedding.shape

    """
    distances = np.zeros((tmax, K, M))
    for i in range(M):
        distances[:, :, i] += (M_embedding[:, :, 0] - embedding[i, 0])**2
        distances[:, :, i] += (M_embedding[:, :, 1] - embedding[i, 1])**2
    distances = np.sqrt(distances)
    nn = np.argmin(distances, axis=-1)//20
    """
    
    matrix = np.zeros((10, 10))
    
    for t in range(tmax-1):
        #for i in range(100):
        matrix[nn[t], nn[t+1]] -= 1
        matrix[nn[t+1], nn[t]] += 1
    
    for d in range(10):
        ax[n_i].axhline(d-0.5)
        ax[n_i].axvline(d-0.5)
    #    matrix[d, d] = 0

    
    
    ax[n_i].imshow(matrix, cmap="bwr")

    ax[n_i].set_xticks(np.arange(0, 10, 1, dtype=int))
    ax[n_i].set_yticks(np.arange(0, 10, 1, dtype=int))

    #ax[n_i].set_title("n="+str(n))



ax_cb = axs['x']
tab10_cmap = matplotlib.cm.bwr
tab10_norm = matplotlib.colors.Normalize(vmin=-np.max(np.abs(matrix)), vmax=np.max(np.abs(matrix)))
cb = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=tab10_cmap, norm=tab10_norm, orientation='vertical')
cb.set_ticks([-np.max(np.abs(matrix)), 0, np.max(np.abs(matrix))]) # Finally found how to center these things 
cb.set_ticklabels([-1, 0, 1])
cb.set_label("Transition rate", labelpad=10)

ax_cb = axs['X']
tab10_cmap = matplotlib.cm.bwr
tab10_norm = matplotlib.colors.Normalize(vmin=-np.max(np.abs(matrix)), vmax=np.max(np.abs(matrix)))
cb = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=tab10_cmap, norm=tab10_norm, orientation='vertical')
cb.set_ticks([-np.max(np.abs(matrix)), 0, np.max(np.abs(matrix))]) # Finally found how to center these things 
cb.set_ticklabels([-1, 0, 1])
cb.set_label("Transition rate", labelpad=10)

plt.subplots_adjust(wspace=0.01)
plt.savefig("matrix_tmp.png")

