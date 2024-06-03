import sys
sys.path.append('../')

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from main_module.KrotovV2 import *

import matplotlib.pyplot as plt

import umap
from sklearn.manifold import TSNE
import trimap
import pacmap

import scipy.interpolate as interpolate

import matplotlib.animation as anim


fig, axs = plt.subplots(2, 2, figsize=(16, 9), layout="constrained")

for ax in axs.ravel():
    ax.set_xticks([])
    ax.set_yticks([])



selected_digits = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
data_dir = "../Figure 3/data/[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]/" # _400_1000

T = 670
for n in [3, 30]: # 3, 30, 15, 40
    savefile = data_dir+"trained_net_n" + str(n) + "_T" + str(T) + ".npz"
    
    data = np.load(savefile)

    print(data['M'].shape)
    if n == 3:
        data_M = data['M'][::10]
        data_L = np.argmax(data['L'][::10], axis=-1)
        
    if n == 30:
        data_M = data['M'][::10]
        data_L = np.argmax(data['L'][::10], axis=-1)
    
    
    data_T = data['miniBatchs_images']
    data_T_inv = np.linalg.pinv(data_T)

    alphas = data_M@data_T_inv
    tmax, K, N = np.shape(alphas)

    data_L_flat = data_L.reshape(tmax*K)
    
    

    tab, norm = get_tab10()

    axs[0, 0].set_title("UMAP")
    axs[0, 1].set_title("TSNE")
    
    axs[1, 0].set_title("\nTRIMAP")
    axs[1, 1].set_title("\nPaCMAP")


    neighbors = 400 # was 200
    reducer = umap.UMAP(n_components=2, n_neighbors=neighbors, verbose=True, low_memory=False) #800 for high n
    embedding = reducer.fit_transform(alphas.reshape(tmax*K, N))
    axs[0, 0].scatter(embedding[:, 0], embedding[:, 1], cmap=tab, norm=norm, c=data_L_flat, s=1)

    reducer = TSNE(perplexity=neighbors,  verbose=1) #800 for high n
    embedding = reducer.fit_transform(alphas.reshape(tmax*K, N))
    axs[0, 1].scatter(embedding[:, 0], embedding[:, 1], cmap=tab, norm=norm, c=data_L_flat, s=1)


    reducer = trimap.TRIMAP(n_inliers=neighbors, verbose=True) #800 for high n
    embedding = reducer.fit_transform(alphas.reshape(tmax*K, N))
    axs[1, 0].scatter(embedding[:, 0], embedding[:, 1], cmap=tab, norm=norm, c=data_L_flat, s=1)


    reducer = pacmap.PaCMAP(n_neighbors=neighbors, verbose=True) #800 for high n
    embedding = reducer.fit_transform(alphas.reshape(tmax*K, N))
    axs[1, 1].scatter(embedding[:, 0], embedding[:, 1], cmap=tab, norm=norm, c=data_L_flat, s=1)


plt.savefig("DifferentiationFigure_ManyMethods.pdf")
