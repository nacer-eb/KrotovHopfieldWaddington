import argparse
import sys
sys.path.append('../')

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from main_module.KrotovV2 import *
from main_module.KrotovV2_utils import *

import umap

import matplotlib
fontsize = 20
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : fontsize}
matplotlib.rc('font', **font)


data_dir = "data/"

# The digit classes to include in the training
selected_digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] 
prefix = str(selected_digits)+"/"


n, temp = 30, 670

fig = plt.figure(layout="constrained", figsize=(18, 10), dpi=200)
axs = fig.subplots(3, 3)


for ax in axs[-1]:
    ax.set_xlabel("UMAP 1")

for ax in axs[:, 0]:
    ax.set_ylabel("UMAP 2")



tab10_cmap = matplotlib.cm.tab10
tab10_norm = matplotlib.colors.Normalize(vmin=0, vmax=10)

saving_dir = data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+"_run"+str(0)+".npz"
data = np.load(saving_dir)
data_M = data['M']
data_T = data['miniBatchs_images'][0]

max_epoch = data_M.shape[0]
epoch_range = np.arange(1, max_epoch, 10)

net = KrotovNet(M=len(data_T), nbMiniBatchs=1) # The rest will be filled in by the next line load-net
net.load_net(saving_dir, epoch=0)
for i in range(9):
    classification_score = np.zeros((len(epoch_range), 10))

    ax = axs.ravel()[i]
    ax.text(0.02, 0.95, chr(ord('A')+i), transform=ax.transAxes, fontsize=30, verticalalignment='top', ha='left')

    data = np.load(data_dir+prefix+"trained_net_n"+str(n)+"_T"+str(temp)+"_run"+str(i)+".npz")
    data_M = data['M'][400:2300:10]
    data_L = data['L'][400:2300:10]

    data_L_flat = np.argmax(data_L, axis=-1).ravel()

    data_T = data['miniBatchs_images'][0]
    data_T_inv = np.linalg.pinv(data_T)
    
    """
    # Alphas
    alphas = data_M@data_T_inv
    tmax, K, N = np.shape(alphas)

    # aggregated alphas
    #alphas = alphas.reshape(tmax, K, N//20, 20).sum(axis=-1)
    #tmax, K, N = np.shape(alphas)
    """

    tmax, K, N = data_M.shape

    neighbors = 200 #500 # was 200
    reducer = umap.UMAP(n_components=2, n_neighbors=neighbors, verbose=True, low_memory=False) #800 for high n #random_state=42,
    embedding = reducer.fit_transform(data_M.reshape(tmax*K, N))
    ax.scatter(embedding[:, 0], embedding[:, 1], cmap=tab10_cmap, norm=tab10_norm, c=data_L_flat, s=1)

    ax.set_xticks([])
    ax.set_yticks([])

plt.savefig("layout_umaps.png")
