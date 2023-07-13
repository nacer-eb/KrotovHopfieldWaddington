import sys
sys.path.append('../')

import numpy as np

import matplotlib
font = {'family' : 'Times New Roman',
        'weight' : 'normal',
        'size'   : 18}
matplotlib.rc('font', **font)

import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.axes_grid1 import make_axes_locatable

import pickle

import umap
import umap.plot

from main_module.KrotovV2_utils import *

data_dir = "data_100_10_200/"


data_T = np.load(data_dir+"miniBatchs_images.npy")[0]
mapper = pickle.load((open(data_dir+"/umap_model_correlation.sav", 'rb')))


M = len(data_T)
keys = np.zeros((M))
for d in range(0, 10):
    keys[d*M//10:(d+1)*M//10] = d
    
embedding = mapper.transform(data_T)
M_embedding = np.load(data_dir+"/memory_umap_embed_correlation_n30_momentum.npy")


tmax, N_mem = np.shape(M_embedding)[0], np.shape(M_embedding)[1]
print("------>", tmax)

t_s = [20, 90, 125, 153, 180, 280, 344]

fig, ax = plt.subplots(2, 3, figsize=(18, 10), sharey=True, sharex=True)
for t_i in range(0, len(t_s)-1):
    im = ax[t_i//3, t_i%3].scatter(embedding[:, 0], embedding[:, 1], c=keys, cmap="tab10", s=10, marker="*")
    
    for i in range(0, N_mem-1):
        data_pnts = M_embedding[t_s[0]:t_s[t_i+1], i, :]
        ax[t_i//3, t_i%3].plot(data_pnts[:, 0], data_pnts[:, 1], linewidth=1, alpha=0.3, color="k")
        ax[0, 0].set_ylabel("UMAP 2"); ax[1, 0].set_ylabel("UMAP 2"); ax[1, t_i%3].set_xlabel("UMAP 1")

    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    ax[t_i//3, t_i%3].text(0.83, 0.97, r"$t=$"+str(t_s[t_i+1]*10), transform=ax[t_i//3, t_i%3].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    
    ax[t_i//3, t_i%3].plot(M_embedding[t_s[t_i+1], :, 0], M_embedding[t_s[t_i+1], :, 1], marker="o",
                           linestyle="", alpha=1, markeredgecolor="k", markerfacecolor="white", markeredgewidth=1, markersize=4)
                           
        

    

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=plt.cm.tab10(0.9), marker="*", linestyle="", ms=13),
                Line2D([0], [0], marker="o", markeredgecolor="k", markerfacecolor="white", markeredgewidth=4, linestyle="", ms=13),
                Line2D([0], [0], color="k", lw=2, marker="", linestyle="-")]

fig.legend(custom_lines, ['Training Data', 'Memory (Visible layer)', 'Memory trajectory/trail'], loc='upper center', ncol=4)
plt.subplots_adjust(top=0.93, bottom=0.09, left=0.052, right=0.92, hspace=0.02, wspace=0.02)

cbar_ax = fig.add_axes([0.925, 0.09, 0.02, 0.84])
cb = fig.colorbar(im, cax=cbar_ax)
cb.ax.set_ylabel("Training data class")

plt.show()



    
