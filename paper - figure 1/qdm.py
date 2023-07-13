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

temp = 670
n = 30

data_T = np.load(data_dir+"miniBatchs_images.npy")[0]
mapper = pickle.load((open(data_dir+"/umap_model_correlation.sav", 'rb')))

embedding = mapper.transform(data_T)

M = len(data_T)
keys = np.zeros((M))
for d in range(0, 10):
    keys[d*M//10:(d+1)*M//10] = d

    
M_embedding = np.load(data_dir+"/memory_umap_embed_correlation_n"+str(n)+".npy")

data = np.load(data_dir+"run_0_n"+str(n)+"_T"+str(temp)+".npz")
data_M = data['M']

tmax = len(data_M)
tmax_umap = len(M_embedding)

print(tmax, tmax_umap)


fig, ax = plt.subplots(1, 2, figsize=(18, 9))

# Background
im_bkgd = ax[1].scatter(embedding[:, 0], embedding[:, 1], c=keys, cmap='tab10', s=10, marker="*")

p, = ax[1].plot(M_embedding[0, :, 0], M_embedding[0, :, 1], marker=".", linestyle="", alpha=1, markeredgecolor="k", markerfacecolor="white", markeredgewidth=1, markersize=9)

im = ax[0].imshow(merge_data(data_M[-1], 10, 10), cmap="bwr", vmin=-1, vmax=1)

ax[0].set_xticks([]); ax[0].set_yticks([])
ax[1].set_xlabel("UMAP 1"); ax[1].set_ylabel("UMAP 2")

props = dict(boxstyle='round', facecolor='whitesmoke', alpha=0.5)
text = ax[1].text(0.83, 0.97, r"$t=$"+str(1000), transform=ax[1].transAxes, fontsize=18, verticalalignment='top', bbox=props)

cbar_ax = fig.add_axes([0.93, 0.12, 0.02, 0.88-0.12])
cb = fig.colorbar(im_bkgd, cax=cbar_ax)
cb.ax.set_ylabel("Training data class")




norm = matplotlib.colors.Normalize(vmin=0, vmax=9)
cmap = matplotlib.cm.get_cmap("tab10")

def update(t_):
    t = t_
    print(t)
    d1 = M_embedding[t//10]
    d2 = M_embedding[t//10+1]

    c = (t%10)/10.0
    print(c)

    d = d1 * (1-c) + d2*c
    
    p.set_data(d[:, 0], d[:, 1])

    text.set_text("$t=$"+str(t))
    
    im.set_data(merge_data(data_M[t], 10, 10))

    return p, im, text,


from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], color=plt.cm.tab10(0.9), marker="*", linestyle="", ms=13),
                Line2D([0], [0], marker="o", markeredgecolor="k", markerfacecolor="white", markeredgewidth=4, linestyle="", ms=13)]

fig.legend(custom_lines, ['Training Data', 'Memory (Visible layer)'], loc='upper center', ncol=4)
plt.subplots_adjust(top=0.88, bottom=0.12, left=0.01, right=0.92, hspace=0.2, wspace=0.15)

ani = anim.FuncAnimation(fig, update, frames=tmax, interval=10, blit=True)
ani.save('2osc.mp4', writer="ffmpeg")
#plt.show()
