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

fig = plt.figure()

axs = fig.subplot_mosaic(
    """
    EEEEEEEE.AAAAAAAABBBBBBBB
    EEEEEEEE.AAAAAAAABBBBBBBB
    EEEEEEEE.AAAAAAAABBBBBBBB
    EEEEEEEE.CCCCCCCCDDDDDDDD
    EEEEEEEE.CCCCCCCCDDDDDDDD
    EEEEEEEE.CCCCCCCCDDDDDDDD
    .........................
    1111122222333334444455555
    1111122222333334444455555
    """
)

ax_umap = np.asarray([axs['A'], axs['B'], axs['C'], axs['D']])
ax_alpha = axs['E']
ax_mem = np.asarray([axs['1'], axs['2'], axs['3'], axs['4'], axs['5']])


ax_umap[1].yaxis.tick_right()
ax_umap[1].yaxis.set_label_position("right")
ax_umap[3].yaxis.tick_right()
ax_umap[3].yaxis.set_label_position("right")

ax_umap[0].set_xticks([]); ax_umap[0].set_yticks([])
ax_umap[1].set_xticks([]);
ax_umap[2].set_yticks([]);


ax_umap[1].set_ylabel("UMAP 2"); ax_umap[3].set_ylabel("UMAP 2")
ax_umap[2].set_xlabel("UMAP 1"); ax_umap[3].set_xlabel("UMAP 1")


ax_alpha.set_xlabel(r"$\alpha_1$"); ax_alpha.set_ylabel(r"$\alpha_7$");

for ax in ax_mem:
    ax.set_xticks([]); ax.set_yticks([])


import umap
import umap.plot

from main_module.KrotovV2_utils import *

data_dir = "data_100_10_200/"



data_T_full = np.load(data_dir+"miniBatchs_images.npy")[0]

mapper = pickle.load((open(data_dir+"/umap_model_correlation.sav", 'rb')))


M = len(data_T_full)
keys = np.zeros((M))
for d in range(0, 10):
    keys[d*M//10:(d+1)*M//10] = d
    
embedding = mapper.transform(data_T_full)

temp = 800
n = 3

data = np.load(data_dir+"run_[1, 7, 9]_n"+str(n)+"_T"+str(temp)+".npz")
data_M = data['M']
data_T = data['miniBatchs_images'][0]
data_T_inv = np.linalg.pinv(data_T)
data_L = data['L']
data_L_key = np.argmax(data_L[-1], axis=-1)

selected_digits = [1, 7, 9]
M_embedding = np.load(data_dir+"/memory_umap_embed_correlation_n"+str(n)+str(selected_digits)+".npy")

tmax, N_mem = np.shape(M_embedding)[0], np.shape(M_embedding)[1]

t_s = [20, 64, 116, 172, 400]


if n == 3:
    t_s = [2, 9, 19, 35, 114]


for t_i, t in enumerate(t_s):
    if t_i < 4:
        im = ax_umap[t_i].scatter(embedding[:, 0], embedding[:, 1], c=keys, cmap="tab10", s=10, marker="*")
        
        for i in range(0, N_mem-1):
            data_pnts = M_embedding[t_s[0]:t_s[t_i+1], i, :]
            ax_umap[t_i].plot(data_pnts[:, 0], data_pnts[:, 1], linewidth=1, alpha=0.3, color="k")
            
            
            
        props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
        ax_umap[t_i].text(0.8, 0.94, r"$t=$"+str(t_s[t_i+1]*10), transform=ax_umap[t_i].transAxes, fontsize=14, verticalalignment='top', bbox=props)
    
        ax_umap[t_i].plot(M_embedding[t_s[t_i+1], :, 0], M_embedding[t_s[t_i+1], :, 1], marker="o", linestyle="",
                          alpha=1, markeredgecolor="k", markerfacecolor="white", markeredgewidth=1, markersize=4)


    # Fixed time labels on bottom
    props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
    ax_mem[t_i].text(0.5, 1.2, r"$t=$"+str(t_s[t_i]*10), transform=ax_mem[t_i].transAxes, fontsize=14, verticalalignment='top', horizontalalignment='center', bbox=props)
    ax_mem[t_i].imshow(merge_data(data_M[t*10, 0:8], 4, 2), cmap="bwr", vmin=-1, vmax=1)


coefs = np.sum((data_M@data_T_inv).reshape(len(data_M), 100, 3, 20), axis=-1)

tmin = 450
if n==3:
    tmin = 50

ax_alpha.plot(coefs[tmin, :, 0], coefs[tmin, :, 1], linestyle="", lw=1, marker="*", ms=10, color="gold")

tab_10 = matplotlib.cm.tab10
normalizer = matplotlib.colors.Normalize(vmin=0, vmax=9)

for i in range(100):
    ax_alpha.plot(coefs[tmin:1900, i, 0], coefs[tmin:1900, i, 1], linestyle="-", lw=1, color=tab_10(normalizer(data_L_key[i])))



#
bot = 0.065 #+ (3.0/9.0) * (0.95-0.05)
cbar_ax = fig.add_axes([0.91, bot, 0.02, 0.95-bot]) # left, bottom, width, height
cb = matplotlib.colorbar.ColorbarBase(cbar_ax, cmap=matplotlib.cm.tab10, norm=matplotlib.colors.Normalize(vmin=0, vmax=9))
cb.ax.set_ylabel("Digit class")

plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.85, hspace=0.2, wspace=0.2)
plt.show()

# First legend
from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], linestyle="", lw=1, marker="*", ms=10, color="gold"),
                Line2D([0], [0], linestyle="-", lw=3, color=tab_10(normalizer(9)))]
fig = plt.figure()
fig.legend(custom_lines, ['Starting point', 'Memory trajectory'], loc='upper center', ncol=3)
plt.show()

# Second legend
custom_lines = [Line2D([0], [0], color=plt.cm.tab10(0.9), marker="*", linestyle="", ms=13),
                Line2D([0], [0], marker="o", markeredgecolor="k", markerfacecolor="white", markeredgewidth=4, linestyle="", ms=13),
                Line2D([0], [0], color="k", lw=2, marker="", linestyle="-")]

fig = plt.figure()
fig.legend(custom_lines, ['Training Data', 'Memory (Visible layer)', 'Memory trajectory/trail'], loc='upper center', ncol=4)
plt.show()
